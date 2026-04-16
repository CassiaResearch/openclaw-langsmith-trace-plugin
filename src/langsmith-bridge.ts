import { Client } from "langsmith";
import type { KVMap } from "langsmith/schemas";
import type { PluginHookAgentContext } from "openclaw/plugin-sdk/plugin-runtime";
import type { PluginConfig } from "./config.js";
import type { Log } from "./log.js";

export interface ProviderModel {
  provider: string;
  model: string;
}

/** LangSmith's canonical `usage_metadata` shape. */
export interface UsageMetadata {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_token_details?: {
    cache_read?: number;
    cache_creation?: number;
  };
  output_token_details?: {
    reasoning?: number;
  };
}

export interface ShapedUsage {
  usageMetadata: UsageMetadata;
  totalCost?: number;
}

/** Superset of OpenClaw's assistant-message `usage` shape; all fields optional for partial data. */
export interface RawUsage {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
  totalTokens?: number;
  cost?: { total?: number };
}

const FAILED_TRACES_ENV = "LANGSMITH_FAILED_TRACES_DIR";

/**
 * Constructs the LangSmith `Client`.
 *
 * **Process-global side effect:** when `cfg.failedTracesDir` is set and the
 * env var is not already populated, this writes
 * `process.env.LANGSMITH_FAILED_TRACES_DIR`. The langsmith SDK reads that
 * env var at spool time, so there is no per-Client config for the path in
 * the current SDK (≤ 0.5.x). The write is one-way and guarded against
 * overwriting an existing value.
 */
export function buildLangsmithClient(cfg: PluginConfig, log: Log): Client {
  if (cfg.failedTracesDir && !process.env[FAILED_TRACES_ENV]) {
    process.env[FAILED_TRACES_ENV] = cfg.failedTracesDir;
    log.debug(`failed-trace spool dir = ${cfg.failedTracesDir}`);
  }

  return new Client({
    apiUrl: cfg.endpoint,
    apiKey: cfg.apiKey,
    autoBatchTracing: true,
    tracingSamplingRate: cfg.samplingRate,
  });
}

/**
 * Metadata attached to every RunTree so LangSmith's Threads view, model
 * filters, and trigger-based queries can key off consistent fields.
 */
export function baseRunMetadata(ctx: PluginHookAgentContext, pm: ProviderModel): KVMap {
  const meta: KVMap = {
    ls_provider: pm.provider,
    ls_model_name: pm.model,
  };
  if (ctx.sessionKey) meta.thread_id = ctx.sessionKey;
  if (ctx.sessionId) meta.openclaw_session_id = ctx.sessionId;
  if (ctx.runId) meta.openclaw_run_id = ctx.runId;
  if (ctx.agentId) meta.agent_id = ctx.agentId;
  if (ctx.trigger) meta.trigger = ctx.trigger;
  // OpenClaw falls back to `messageProvider` for `channelId` in DMs, so the
  // pair can match. Only emit `channel_id` when it's actually a distinct
  // channel identifier — otherwise the value is just the provider name and
  // misleads anyone filtering on it.
  if (ctx.channelId && ctx.channelId !== ctx.messageProvider) meta.channel_id = ctx.channelId;
  if (ctx.messageProvider) meta.message_provider = ctx.messageProvider;
  return meta;
}

/**
 * Translates OpenClaw's assistant-message usage into LangSmith's canonical
 * `usage_metadata`. Returns `undefined` when there are no tokens to
 * attach so callers can skip emitting a usage payload.
 */
export function shapeUsage(raw: RawUsage | undefined): ShapedUsage | undefined {
  if (!raw) return undefined;

  const input = toCount(raw.input);
  const output = toCount(raw.output);
  const cacheRead = toCount(raw.cacheRead);
  const cacheWrite = toCount(raw.cacheWrite);
  const reasoning = toCount(raw.reasoning);

  if (input + output + cacheRead + cacheWrite === 0) return undefined;

  const inputTokens = input + cacheRead + cacheWrite;
  const usageMetadata: UsageMetadata = {
    input_tokens: inputTokens,
    output_tokens: output,
    total_tokens: raw.totalTokens ?? inputTokens + output,
  };

  if (cacheRead > 0 || cacheWrite > 0) {
    usageMetadata.input_token_details = {};
    if (cacheRead > 0) usageMetadata.input_token_details.cache_read = cacheRead;
    if (cacheWrite > 0) usageMetadata.input_token_details.cache_creation = cacheWrite;
  }

  if (reasoning > 0) usageMetadata.output_token_details = { reasoning };

  return { usageMetadata, totalCost: raw.cost?.total };
}

function toCount(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : 0;
}
