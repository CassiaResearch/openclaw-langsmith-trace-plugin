import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Client } from "langsmith";
import type {
  PluginHookAfterToolCallEvent,
  PluginHookAgentContext,
  PluginHookBeforeToolCallEvent,
  PluginHookLlmInputEvent,
  PluginHookSubagentEndedEvent,
  PluginHookToolContext,
} from "openclaw/plugin-sdk/plugin-runtime";
import { TurnRecorder } from "../src/turn-recorder.js";
import type { PluginConfig } from "../src/config.js";

type CreateRunParams = Parameters<Client["createRun"]>[0];
type UpdateRunParams = Parameters<Client["updateRun"]>[1];

interface RecordedCalls {
  creates: CreateRunParams[];
  updates: Array<{ id: string; update: UpdateRunParams }>;
}

// The spied Client never hits the network. createRun and updateRun are the
// only Client methods RunTree uses internally for post/patch cycles.
function makeSpiedClient(): { client: Client; calls: RecordedCalls } {
  const client = new Client({
    apiKey: "test-key",
    apiUrl: "https://example.invalid",
    autoBatchTracing: false,
  });
  const calls: RecordedCalls = { creates: [], updates: [] };
  vi.spyOn(client, "createRun").mockImplementation(async (run) => {
    calls.creates.push(run);
  });
  vi.spyOn(client, "updateRun").mockImplementation(async (id, update) => {
    calls.updates.push({ id, update });
  });
  return { client, calls };
}

const noopLog = {
  info: () => {},
  warn: () => {},
  error: () => {},
  debug: () => {},
};

const baseConfig: PluginConfig = {
  apiKey: "test-key",
  endpoint: "https://example.invalid",
  projectName: "openclaw-test",
  traceAgentTurns: true,
  traceToolCalls: true,
  samplingRate: 1,
  failedTracesDir: undefined,
  debug: false,
};

const ctx: PluginHookAgentContext = {
  runId: "run-1",
  agentId: "agent-1",
  sessionKey: "session-1",
  sessionId: "session-id-1",
  trigger: "discord",
  channelId: "channel-1",
  messageProvider: "discord",
  modelProviderId: "anthropic",
  modelId: "claude-opus-4-7",
};

const llmInput: PluginHookLlmInputEvent = {
  runId: "run-1",
  sessionId: "session-id-1",
  provider: "anthropic",
  model: "claude-opus-4-7",
  systemPrompt: "You are a helpful assistant.",
  prompt: "Summarize this document.",
  historyMessages: [],
  imagesCount: 0,
};

function assistantMessage(opts?: {
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
  stopReason?: string;
}) {
  return {
    role: "assistant" as const,
    content: [{ type: "text" as const, text: "Here is the summary." }],
    usage: {
      input: 100,
      output: 50,
      cacheRead: opts?.cacheRead,
      cacheWrite: opts?.cacheWrite,
      reasoning: opts?.reasoning,
    },
    stopReason: opts?.stopReason ?? "end_turn",
  } as any;
}

describe("TurnRecorder", () => {
  beforeEach(() => vi.useRealTimers());
  afterEach(() => vi.restoreAllMocks());

  it("happy path: llm_input → message → tool → agent_end produces the expected run tree", async () => {
    const { client, calls } = makeSpiedClient();
    const recorder = new TurnRecorder(client, baseConfig, noopLog);

    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);
    await recorder.onMessageWrite(
      ctx.sessionKey!,
      assistantMessage({ cacheRead: 20, cacheWrite: 5 }),
    );

    const toolCtx: PluginHookToolContext = {
      sessionKey: ctx.sessionKey,
      agentId: ctx.agentId,
      runId: ctx.runId,
      toolName: "fetch_doc",
      toolCallId: "tool-call-1",
    };
    const toolStart: PluginHookBeforeToolCallEvent = {
      toolName: "fetch_doc",
      toolCallId: "tool-call-1",
      params: { id: "doc-1" },
    };
    await recorder.onToolStart(ctx.sessionKey!, toolStart, toolCtx);

    const toolEnd: PluginHookAfterToolCallEvent = {
      toolName: "fetch_doc",
      toolCallId: "tool-call-1",
      params: { id: "doc-1" },
      result: { ok: true },
      durationMs: 12,
    };
    await recorder.onToolEnd(ctx.sessionKey!, toolEnd, toolCtx);
    await recorder.onTurnEnd(ctx.sessionKey!, true, 200, undefined);

    const runTypes = calls.creates.map((c) => c.run_type);
    expect(runTypes).toEqual(["chain", "llm", "tool"]);

    const root = calls.creates[0]!;
    expect(root.name).toBe("agent_turn");
    expect((root.extra as any)?.metadata?.thread_id).toBe(ctx.sessionKey);
    expect((root.extra as any)?.metadata?.ls_provider).toBe("anthropic");
    expect((root.extra as any)?.metadata?.ls_model_name).toBe("claude-opus-4-7");

    const llm = calls.creates[1]!;
    expect(llm.name).toBe("ChatAnthropic"); // LangChain chat-class name
    expect((llm.extra as any)?.metadata?.ls_model_type).toBe("chat");
    expect((llm.extra as any)?.metadata?.ls_model_name).toBe("claude-opus-4-7");

    // usage_metadata lives on the LLM run's outputs (the LangChain convention)
    const llmUpdate = calls.updates.find((u) => u.id === llm.id);
    expect(llmUpdate).toBeDefined();
    const llmOutputs = llmUpdate!.update.outputs as any;
    expect(llmOutputs.usage_metadata.input_tokens).toBe(100 + 20 + 5);
    expect(llmOutputs.usage_metadata.output_tokens).toBe(50);
    expect(llmOutputs.usage_metadata.input_token_details).toEqual({
      cache_read: 20,
      cache_creation: 5,
    });

    const tool = calls.creates[2]!;
    expect(tool.name).toBe("fetch_doc");
    expect(tool.parent_run_id).toBe(llm.id); // nested under LLM, not root
    expect((tool.extra as any)?.metadata?.tool_call_id).toBe("tool-call-1");

    const rootUpdate = calls.updates.find((u) => u.id === root.id);
    expect(rootUpdate).toBeDefined();
    const rootOutputs = rootUpdate!.update.outputs as any;
    // Root outputs follow LangChain's AgentExecutor convention — just
    // `output`, nothing else. Stats live in metadata; duration/usage come
    // from LangSmith aggregating the run tree.
    expect(Object.keys(rootOutputs)).toEqual(["output"]);
    expect(rootOutputs.output).toBe("Here is the summary.");
    const rootMeta = (rootUpdate!.update.extra as any)?.metadata;
    expect(rootMeta.llm_call_count).toBe(1);
  });

  it("compaction retry: second llm_input closes the first root with reason", async () => {
    const { client, calls } = makeSpiedClient();
    const recorder = new TurnRecorder(client, baseConfig, noopLog);

    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);
    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);

    const roots = calls.creates.filter((c) => c.run_type === "chain");
    expect(roots).toHaveLength(2);

    const firstUpdate = calls.updates.find((u) => u.id === roots[0]!.id);
    expect(firstUpdate).toBeDefined();
    // Forced close records failure via `run.error`; outputs stays empty so
    // LangSmith's status-from-error logic takes over.
    expect(firstUpdate!.update.error).toBe("Compacted and retried");
    expect(firstUpdate!.update.outputs).toEqual({});
  });

  it("tool without a preceding assistant message attaches to the root", async () => {
    const { client, calls } = makeSpiedClient();
    const recorder = new TurnRecorder(client, baseConfig, noopLog);

    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);

    const toolCtx: PluginHookToolContext = {
      sessionKey: ctx.sessionKey,
      toolName: "preflight",
      toolCallId: "tool-preflight",
    };
    await recorder.onToolStart(
      ctx.sessionKey!,
      { toolName: "preflight", toolCallId: "tool-preflight", params: {} },
      toolCtx,
    );

    const root = calls.creates.find((c) => c.run_type === "chain")!;
    const tool = calls.creates.find((c) => c.run_type === "tool")!;
    expect(tool.parent_run_id).toBe(root.id);
  });

  it("subagent_ended adds a chain child under the requester's root", async () => {
    const { client, calls } = makeSpiedClient();
    const recorder = new TurnRecorder(client, baseConfig, noopLog);

    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);

    const event: PluginHookSubagentEndedEvent = {
      targetSessionKey: "child-session",
      targetKind: "subagent",
      reason: "handoff",
      runId: "child-run",
      endedAt: Date.now(),
      outcome: "ok",
    };
    await recorder.onSubagent(ctx.sessionKey!, event);

    const subagent = calls.creates.find((c) => c.name === "subagent:child-session");
    expect(subagent).toBeDefined();
    expect(subagent!.run_type).toBe("chain");
    const root = calls.creates.find((c) => c.run_type === "chain" && c.name === "agent_turn")!;
    expect(subagent!.parent_run_id).toBe(root.id);
  });

  it("shutdown closes active turns with a shutdown error", async () => {
    const { client, calls } = makeSpiedClient();
    const recorder = new TurnRecorder(client, baseConfig, noopLog);

    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);
    await recorder.shutdown();

    const root = calls.creates.find((c) => c.run_type === "chain")!;
    const rootUpdate = calls.updates.find((u) => u.id === root.id);
    expect(rootUpdate!.update.error).toBe("Gateway shutdown");
  });

  it("subagent run duration comes from subagent_spawned → subagent_ended", async () => {
    const { client, calls } = makeSpiedClient();
    const recorder = new TurnRecorder(client, baseConfig, noopLog);

    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);

    const spawnedAt = 1_700_000_000_000;
    const endedAt = spawnedAt + 1_500;
    const nowSpy = vi.spyOn(Date, "now");

    nowSpy.mockReturnValue(spawnedAt);
    recorder.onSubagentSpawned(ctx.sessionKey!, {
      runId: "child-run",
      childSessionKey: "child-session",
      agentId: "child-agent",
      mode: "run",
      threadRequested: false,
    });

    nowSpy.mockReturnValue(endedAt);
    await recorder.onSubagent(ctx.sessionKey!, {
      targetSessionKey: "child-session",
      targetKind: "subagent",
      reason: "handoff",
      runId: "child-run",
      endedAt,
      outcome: "ok",
    });
    nowSpy.mockRestore();

    // Name derives from the spawn event's agentId (more informative than
    // the opaque session key).
    const subagent = calls.creates.find((c) => c.name === "subagent:child-agent")!;
    const subagentUpdate = calls.updates.find((u) => u.id === subagent.id)!;

    // start_time is serialised to a microsecond ISO string by the SDK, but
    // the numeric prefix tells us it came from our spawnedAt timestamp.
    expect(String(subagent.start_time)).toContain("2023-11-14T22:13:20.000");
    expect(subagentUpdate.update.end_time).toBe(endedAt);

    // Descriptive fields all live in metadata (filterable) — inputs stays
    // empty since the ended event doesn't carry the subagent's input payload.
    const meta = (subagent.extra as any)?.metadata;
    expect(meta.subagent_agent_id).toBe("child-agent");
    expect(meta.subagent_mode).toBe("run");
    expect(meta.subagent_outcome).toBe("ok");
    expect(subagent.inputs).toEqual({});

    const tags = (subagent as any).tags as string[];
    expect(tags).toContain("subagent_agent:child-agent");
    expect(tags).toContain("subagent_mode:run");

    // Outputs is a readable preview for the trace-list row.
    expect(subagentUpdate.update.outputs).toEqual({ output: "ok: handoff" });
  });

  it("agent_end force-closes orphan tool runs", async () => {
    const { client, calls } = makeSpiedClient();
    const recorder = new TurnRecorder(client, baseConfig, noopLog);

    await recorder.onTurnStart(ctx.sessionKey!, llmInput, ctx);
    await recorder.onMessageWrite(ctx.sessionKey!, assistantMessage());

    const toolCtx: PluginHookToolContext = {
      sessionKey: ctx.sessionKey,
      toolName: "slow_tool",
      toolCallId: "tool-orphan",
    };
    await recorder.onToolStart(
      ctx.sessionKey!,
      { toolName: "slow_tool", toolCallId: "tool-orphan", params: {} },
      toolCtx,
    );

    // Agent ends while the tool is still open — no after_tool_call ever fires.
    await recorder.onTurnEnd(ctx.sessionKey!, true, 100, undefined);

    const tool = calls.creates.find((c) => c.run_type === "tool")!;
    const toolUpdate = calls.updates.find((u) => u.id === tool.id);
    expect(toolUpdate).toBeDefined();
    expect(toolUpdate!.update.error).toBe("Tool run orphaned at turn end");
  });
});
