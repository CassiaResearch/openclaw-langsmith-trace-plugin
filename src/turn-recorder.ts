import { Client, RunTree } from "langsmith";
import type { KVMap } from "langsmith/schemas";
import type {
  PluginHookAfterToolCallEvent,
  PluginHookAgentContext,
  PluginHookBeforeMessageWriteEvent,
  PluginHookBeforeToolCallEvent,
  PluginHookLlmInputEvent,
  PluginHookSubagentEndedEvent,
  PluginHookSubagentSpawnedEvent,
  PluginHookToolContext,
} from "openclaw/plugin-sdk/plugin-runtime";
import type { PluginConfig } from "./config.js";
import type { Log } from "./log.js";
import {
  baseRunMetadata,
  shapeUsage,
  type ProviderModel,
  type ShapedUsage,
} from "./langsmith-bridge.js";

type InboundMessage = PluginHookBeforeMessageWriteEvent["message"];
type AssistantLike = Extract<InboundMessage, { role: "assistant" }>;

const ROOT_RUN_NAME = "agent_turn";
const COMPACTION_RETRY_ERROR = "Compacted and retried";
const SHUTDOWN_ERROR = "Gateway shutdown";
const SESSION_ENDED_ERROR = "Session ended";
const ORPHAN_TOOL_ERROR = "Tool run orphaned at turn end";

/**
 * Provider → LangChain chat-class name. LangSmith's UI keys on these names
 * to render the chat-model card (icon, token panel, message viewer). Falls
 * back to `ChatModel` for providers we don't recognise.
 */
const CHAT_CLASS_BY_PROVIDER: Record<string, string> = {
  anthropic: "ChatAnthropic",
  openai: "ChatOpenAI",
  azure: "AzureChatOpenAI",
  google: "ChatGoogleGenerativeAI",
  bedrock: "ChatBedrock",
  mistral: "ChatMistralAI",
  cohere: "ChatCohere",
  groq: "ChatGroq",
  deepseek: "ChatDeepSeek",
  xai: "ChatXAI",
  ollama: "ChatOllama",
};

/**
 * Snapshot of a `subagent_spawned` event, kept until the matching
 * `subagent_ended` fires so the final RunTree can carry spawn-time fields
 * (`agentId`, `label`, `mode`) that the ended event doesn't include.
 */
interface SubagentStart {
  startedAt: number;
  agentId: string;
  label?: string;
  mode: "run" | "session";
  childSessionKey: string;
}

interface ActiveTurn {
  root: RunTree;
  providerModel: ProviderModel;
  ctx: PluginHookAgentContext;
  /** Pre-built from (ctx, providerModel) — spread into every child run's metadata. */
  baseMetadata: KVMap;
  contextBuffer: unknown[];
  tools: Map<string, RunTree>;
  /** runId → spawn snapshot for subagents spawned in this turn. */
  subagentStarts: Map<string, SubagentStart>;
  totals: UsageTotals;
  innerLlmCalls: number;
  mostRecentLlm?: RunTree;
  /** Start timestamp to use for the next inner-LLM child run. */
  nextLlmStartMs: number;
  /** Latest assistant text block; surfaced as root `outputs.output` for trace previews. */
  lastAssistantText?: string;
}

interface UsageTotals {
  input: number;
  output: number;
  total: number;
  cost: number;
  hadUsage: boolean;
  hadCost: boolean;
}

export class TurnRecorder {
  private readonly active = new Map<string, ActiveTurn>();

  constructor(
    private readonly client: Client,
    private readonly cfg: PluginConfig,
    private readonly log: Log,
  ) {}

  async onSessionEnd(sessionKey: string): Promise<void> {
    const turn = this.active.get(sessionKey);
    if (!turn) return;
    this.log.debug(`session_end ${sessionKey} — closing open turn`);
    this.active.delete(sessionKey);
    await this.forceCloseTurn(turn, SESSION_ENDED_ERROR);
  }

  async onTurnStart(
    sessionKey: string,
    event: PluginHookLlmInputEvent,
    ctx: PluginHookAgentContext,
  ): Promise<void> {
    const existing = this.active.get(sessionKey);
    if (existing) {
      this.log.debug(`llm_input replacing existing turn for ${sessionKey} (compaction retry)`);
      this.active.delete(sessionKey);
      await this.forceCloseTurn(existing, COMPACTION_RETRY_ERROR);
      // A concurrent onTurnStart may have registered a fresher turn while we
      // were awaiting the close. If so, abort — the newer turn wins.
      if (this.active.has(sessionKey)) return;
    }

    const providerModel: ProviderModel = { provider: event.provider, model: event.model };
    const contextBuffer = seedContext(event);
    const startMs = Date.now();

    // TODO: once OpenClaw exposes the resolved tool catalog on
    // `PluginHookLlmInputEvent` (tracked upstream — the runner already has
    // the list at the moment `llm_input` fires, it just isn't on the event
    // payload today), attach it here as `rootMetadata.available_tools` and
    // as `inputs.tools` on each inner LLM child. Until then the trace shows
    // tool *invocations* (as child runs) but not the *catalog* the model
    // was choosing from — so LangSmith's chat-model viewer won't render the
    // "tools the LLM could see", which LangGraph traces usually include.
    const baseMetadata = baseRunMetadata(ctx, providerModel);
    const rootMetadata: KVMap = { ...baseMetadata };
    if (event.imagesCount > 0) rootMetadata.images_count = event.imagesCount;

    const root = new RunTree({
      name: ROOT_RUN_NAME,
      run_type: "chain",
      project_name: this.cfg.projectName,
      start_time: startMs,
      inputs: { messages: [...contextBuffer] },
      metadata: rootMetadata,
      tags: buildRootTags(ctx, providerModel),
      client: this.client,
    });

    // Register synchronously so concurrent before_message_write /
    // before_tool_call hooks that fire before the network post resolves
    // can still find the active turn.
    this.active.set(sessionKey, {
      root,
      providerModel,
      ctx,
      baseMetadata,
      contextBuffer,
      tools: new Map(),
      subagentStarts: new Map(),
      totals: freshTotals(),
      innerLlmCalls: 0,
      nextLlmStartMs: startMs,
    });

    await root.postRun();
  }

  async onMessageWrite(sessionKey: string, message: InboundMessage): Promise<void> {
    const turn = this.active.get(sessionKey);
    if (!turn) return;

    turn.contextBuffer.push(message);
    if (!isAssistant(message)) return;

    turn.innerLlmCalls += 1;
    const usage = shapeUsage(message.usage);
    if (usage) accumulate(turn.totals, usage);
    const text = extractAssistantText(message);
    if (text) turn.lastAssistantText = text;

    const startMs = turn.nextLlmStartMs;
    const endMs = Date.now();

    const llmRun = turn.root.createChild({
      name: chatClassName(turn.providerModel.provider),
      run_type: "llm",
      start_time: startMs,
      inputs: { messages: turn.contextBuffer.slice(0, -1) },
      metadata: buildLlmMetadata(turn, message),
      tags: ["openclaw:inner_llm_call"],
    });

    turn.mostRecentLlm = llmRun;
    turn.nextLlmStartMs = endMs;

    await llmRun.postRun();
    await llmRun.end(buildLlmOutputs(message, usage), deriveLlmError(message), endMs);
    await llmRun.patchRun();
  }

  async onToolStart(
    sessionKey: string,
    event: PluginHookBeforeToolCallEvent,
    ctx: PluginHookToolContext,
  ): Promise<void> {
    const turn = this.active.get(sessionKey);
    if (!turn) return;

    const toolCallId = event.toolCallId ?? ctx.toolCallId;
    if (!toolCallId) {
      this.log.warn("before_tool_call missing toolCallId — tool trace skipped");
      return;
    }

    const parent = turn.mostRecentLlm ?? turn.root;
    const toolRun = parent.createChild({
      name: event.toolName,
      run_type: "tool",
      inputs: { params: event.params },
      metadata: { ...turn.baseMetadata, tool_call_id: toolCallId },
      tags: ["openclaw:tool", `tool:${event.toolName}`],
    });

    // Register before await so a racing after_tool_call isn't dropped.
    turn.tools.set(toolCallId, toolRun);
    await toolRun.postRun();
  }

  async onToolEnd(
    sessionKey: string,
    event: PluginHookAfterToolCallEvent,
    ctx: PluginHookToolContext,
  ): Promise<void> {
    const turn = this.active.get(sessionKey);
    if (!turn) return;

    const toolCallId = event.toolCallId ?? ctx.toolCallId;
    if (!toolCallId) return;

    const toolRun = turn.tools.get(toolCallId);
    if (!toolRun) return;
    turn.tools.delete(toolCallId);

    const endMs = Date.now();
    await toolRun.end({ result: event.result ?? null }, event.error, endMs);
    await toolRun.patchRun();

    // Next inner LLM child starts wherever this tool finished, so the
    // timeline has no visible gaps.
    turn.nextLlmStartMs = endMs;
  }

  onSubagentSpawned(requesterSessionKey: string, event: PluginHookSubagentSpawnedEvent): void {
    const turn = this.active.get(requesterSessionKey);
    if (!turn) return;
    turn.subagentStarts.set(event.runId, {
      startedAt: Date.now(),
      agentId: event.agentId,
      label: event.label,
      mode: event.mode,
      childSessionKey: event.childSessionKey,
    });
  }

  async onSubagent(
    requesterSessionKey: string,
    event: PluginHookSubagentEndedEvent,
  ): Promise<void> {
    const turn = this.active.get(requesterSessionKey);
    if (!turn) return;

    const endMs = event.endedAt ?? Date.now();
    const spawn = event.runId ? turn.subagentStarts.get(event.runId) : undefined;
    // Clamp to endMs in case of host-side clock skew or a stale `endedAt`.
    const startMs = spawn ? Math.min(spawn.startedAt, endMs) : endMs;
    if (event.runId) turn.subagentStarts.delete(event.runId);

    const metadata: KVMap = {
      ...turn.baseMetadata,
      subagent_session_key: event.targetSessionKey,
      subagent_kind: event.targetKind,
    };
    if (event.runId) metadata.subagent_run_id = event.runId;
    if (event.outcome) metadata.outcome = event.outcome;
    if (spawn) {
      metadata.subagent_agent_id = spawn.agentId;
      metadata.subagent_mode = spawn.mode;
      if (spawn.label) metadata.subagent_label = spawn.label;
    }

    const tags = ["openclaw:subagent", `subagent_kind:${event.targetKind}`];
    if (spawn) {
      tags.push(`subagent_agent:${spawn.agentId}`, `subagent_mode:${spawn.mode}`);
    }

    const subagentRun = turn.root.createChild({
      name: subagentRunName(event.targetSessionKey, spawn),
      run_type: "chain",
      start_time: startMs,
      inputs: { target_kind: event.targetKind, reason: event.reason },
      metadata,
      tags,
    });

    await subagentRun.postRun();
    const outputs: KVMap = {};
    if (event.outcome) outputs.outcome = event.outcome;
    await subagentRun.end(outputs, event.error, endMs);
    await subagentRun.patchRun();
  }

  async onTurnEnd(
    sessionKey: string,
    success: boolean,
    durationMs: number | undefined,
    error: string | undefined,
  ): Promise<void> {
    const turn = this.active.get(sessionKey);
    if (!turn) return;
    this.active.delete(sessionKey);

    const endMs = Date.now();
    await this.closeOrphanTools(turn, ORPHAN_TOOL_ERROR, endMs);

    const summary: KVMap = { success, llm_call_count: turn.innerLlmCalls };
    // `output` is LangChain's AgentExecutor convention — LangSmith's trace
    // list renders it as the row preview, so surfacing the final assistant
    // text here makes the trace immediately scannable.
    if (turn.lastAssistantText) summary.output = turn.lastAssistantText;
    if (durationMs !== undefined) summary.duration_ms = durationMs;
    if (turn.totals.hadUsage) {
      const usage: KVMap = {
        input_tokens: turn.totals.input,
        output_tokens: turn.totals.output,
        total_tokens: turn.totals.total,
      };
      if (turn.totals.hadCost) usage.total_cost = turn.totals.cost;
      summary.usage = usage;
    }

    await turn.root.end(summary, error, endMs);
    await turn.root.patchRun();
  }

  async shutdown(): Promise<void> {
    const entries = [...this.active.entries()];
    this.active.clear();
    for (const [sessionKey, turn] of entries) {
      this.log.debug(`shutdown closing turn ${sessionKey}`);
      await this.forceCloseTurn(turn, SHUTDOWN_ERROR);
    }
  }

  private async forceCloseTurn(turn: ActiveTurn, reason: string): Promise<void> {
    const endMs = Date.now();
    await this.closeOrphanTools(turn, reason, endMs);
    await turn.root.end({ success: false }, reason, endMs);
    await turn.root.patchRun();
  }

  private async closeOrphanTools(turn: ActiveTurn, reason: string, endMs: number): Promise<void> {
    for (const toolRun of turn.tools.values()) {
      await toolRun.end({ result: null }, reason, endMs);
      await toolRun.patchRun();
    }
    turn.tools.clear();
  }
}

function freshTotals(): UsageTotals {
  return { input: 0, output: 0, total: 0, cost: 0, hadUsage: false, hadCost: false };
}

function accumulate(acc: UsageTotals, usage: ShapedUsage): void {
  acc.input += usage.usageMetadata.input_tokens;
  acc.output += usage.usageMetadata.output_tokens;
  acc.total += usage.usageMetadata.total_tokens;
  acc.hadUsage = true;
  if (usage.totalCost !== undefined) {
    acc.cost += usage.totalCost;
    acc.hadCost = true;
  }
}

function seedContext(event: PluginHookLlmInputEvent): unknown[] {
  const seeded: unknown[] = [];
  if (event.systemPrompt) seeded.push({ role: "system", content: event.systemPrompt });
  seeded.push(...event.historyMessages);
  if (event.prompt) seeded.push({ role: "user", content: event.prompt });
  return seeded;
}

function buildRootTags(ctx: PluginHookAgentContext, pm: ProviderModel): string[] {
  const tags = ["openclaw:agent_turn", `provider:${pm.provider}`, `model:${pm.model}`];
  if (ctx.agentId) tags.push(`agent:${ctx.agentId}`);
  if (ctx.trigger) tags.push(`trigger:${ctx.trigger}`);
  if (ctx.messageProvider) tags.push(`source:${ctx.messageProvider}`);
  if (ctx.channelId) tags.push(`channel:${ctx.channelId}`);
  return tags;
}

function buildLlmMetadata(turn: ActiveTurn, message: AssistantLike): KVMap {
  const meta: KVMap = {
    ...turn.baseMetadata,
    ls_model_type: "chat",
    stop_reason: message.stopReason,
  };
  if (message.responseId) meta.response_id = message.responseId;
  return meta;
}

/**
 * LLM outputs follow LangChain's chat-model convention: top-level
 * `usage_metadata` is what LangSmith's UI and cost calculator key off of.
 */
function buildLlmOutputs(message: InboundMessage, usage: ShapedUsage | undefined): KVMap {
  const outputs: KVMap = { message };
  if (usage) outputs.usage_metadata = usage.usageMetadata;
  return outputs;
}

function deriveLlmError(message: AssistantLike): string | undefined {
  // AssistantMessage.stopReason uses a known enum; surface the error states.
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    return message.errorMessage ?? message.stopReason;
  }
  return undefined;
}

function isAssistant(message: InboundMessage): message is AssistantLike {
  return message.role === "assistant";
}

/**
 * Concatenates the text blocks of an assistant message, skipping thinking
 * and tool-call blocks. Returns `undefined` when no visible text exists
 * (e.g. a pure tool-use turn) so callers can leave the prior preview intact.
 */
function extractAssistantText(message: AssistantLike): string | undefined {
  const content = message.content;
  if (typeof content === "string") return content || undefined;
  if (!Array.isArray(content)) return undefined;
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === "object" && (block as { type?: string }).type === "text") {
      const text = (block as { text?: unknown }).text;
      if (typeof text === "string" && text.length > 0) parts.push(text);
    }
  }
  return parts.length > 0 ? parts.join("") : undefined;
}

function chatClassName(provider: string): string {
  return CHAT_CLASS_BY_PROVIDER[provider.toLowerCase()] ?? "ChatModel";
}

/**
 * Prefer the human-readable label when the host supplied one, fall back to
 * the agentId, and finally to the target session key (legacy behavior)
 * when no spawn event was observed.
 */
function subagentRunName(targetSessionKey: string, spawn: SubagentStart | undefined): string {
  if (spawn?.label) return `subagent:${spawn.label}`;
  if (spawn?.agentId) return `subagent:${spawn.agentId}`;
  return `subagent:${targetSessionKey}`;
}
