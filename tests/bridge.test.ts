import { describe, expect, it } from "vitest";
import { shapeUsage, baseRunMetadata } from "../src/langsmith-bridge.js";
import type { PluginHookAgentContext } from "openclaw/plugin-sdk/plugin-runtime";

const pm = { provider: "anthropic", model: "claude-opus" };

describe("shapeUsage", () => {
  it("returns undefined when there are no tokens", () => {
    expect(shapeUsage(undefined)).toBeUndefined();
    expect(shapeUsage({ input: 0, output: 0, totalTokens: 0 })).toBeUndefined();
  });

  it("emits usage even when only cache tokens are present", () => {
    const shaped = shapeUsage({ input: 0, output: 0, cacheRead: 25 });
    expect(shaped).toBeDefined();
    expect(shaped!.usageMetadata.input_tokens).toBe(25);
    expect(shaped!.usageMetadata.input_token_details).toEqual({ cache_read: 25 });
  });

  it("packs cache tokens into input_token_details", () => {
    const shaped = shapeUsage({ input: 100, output: 40, cacheRead: 30, cacheWrite: 10 });
    expect(shaped!.usageMetadata.input_tokens).toBe(100 + 30 + 10);
    expect(shaped!.usageMetadata.input_token_details).toEqual({
      cache_read: 30,
      cache_creation: 10,
    });
    expect(shaped!.usageMetadata.output_token_details).toBeUndefined();
  });

  it("packs reasoning tokens into output_token_details", () => {
    const shaped = shapeUsage({ input: 5, output: 5, reasoning: 42 });
    expect(shaped!.usageMetadata.output_token_details).toEqual({ reasoning: 42 });
  });

  it("computes total when only input/output are present", () => {
    const shaped = shapeUsage({ input: 7, output: 3 });
    expect(shaped!.usageMetadata.total_tokens).toBe(10);
  });

  it("prefers totalTokens over computed sum", () => {
    const shaped = shapeUsage({ input: 7, output: 3, totalTokens: 99 });
    expect(shaped!.usageMetadata.total_tokens).toBe(99);
  });
});

describe("baseRunMetadata", () => {
  const ctx: PluginHookAgentContext = {
    runId: "r1",
    agentId: "a1",
    sessionKey: "s-key",
    sessionId: "s-id",
    trigger: "slack",
    channelId: "ch-1",
    messageProvider: "slack",
  };

  it("includes thread_id for LangSmith's Threads view", () => {
    expect(baseRunMetadata(ctx, pm).thread_id).toBe("s-key");
  });

  it("includes ls_provider and ls_model_name", () => {
    const meta = baseRunMetadata(ctx, pm);
    expect(meta.ls_provider).toBe("anthropic");
    expect(meta.ls_model_name).toBe("claude-opus");
  });

  it("namespaces OpenClaw's sessionId to avoid shadowing thread_id in LangSmith's Threads view", () => {
    const meta = baseRunMetadata(ctx, pm);
    // LangSmith's thread list prefers metadata.session_id over thread_id, so
    // we expose the OpenClaw UUID under a namespaced key instead.
    expect(meta.session_id).toBeUndefined();
    expect(meta.openclaw_session_id).toBe("s-id");
  });

  it("skips empty context fields", () => {
    expect(baseRunMetadata({ sessionKey: "k" }, pm)).toEqual({
      thread_id: "k",
      ls_provider: "anthropic",
      ls_model_name: "claude-opus",
    });
  });
});
