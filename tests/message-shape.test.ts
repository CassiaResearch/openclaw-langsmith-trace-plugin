import { describe, expect, it } from "vitest";
import { extractToolCalls, normalizeMessage } from "../src/message-shape.js";

describe("normalizeMessage", () => {
  it("rewrites camelCase `toolCall` blocks to Anthropic-canonical `tool_use`", () => {
    const input = {
      role: "assistant",
      content: [
        { type: "text", text: "Let me check." },
        { type: "toolCall", id: "t1", name: "fetch", arguments: { url: "https://x" } },
      ],
    };
    const out = normalizeMessage(input) as any;
    expect(out.role).toBe("assistant");
    expect(out.content[0]).toEqual({ type: "text", text: "Let me check." });
    expect(out.content[1]).toEqual({
      type: "tool_use",
      id: "t1",
      name: "fetch",
      input: { url: "https://x" },
    });
  });

  it("leaves already-canonical tool_use blocks alone", () => {
    const block = { type: "tool_use", id: "t1", name: "fetch", input: { url: "x" } };
    const out = normalizeMessage({ role: "assistant", content: [block] }) as any;
    expect(out.content[0]).toEqual(block);
  });

  it("converts `role: toolResult` messages to LangChain's `role: tool` shape", () => {
    const input = {
      role: "toolResult",
      content: [{ type: "text", text: "ok" }],
      toolCallId: "t1",
      toolName: "fetch",
      isError: false,
    };
    const out = normalizeMessage(input) as any;
    expect(out.role).toBe("tool");
    expect(out.tool_call_id).toBe("t1");
    expect(out.name).toBe("fetch");
    expect(out.content).toBe("ok"); // flattened single text block → string
    expect(out).not.toHaveProperty("toolCallId");
    expect(out).not.toHaveProperty("toolName");
    expect(out).not.toHaveProperty("isError");
  });

  it("flags errored tool results with status=error so the viewer can render them", () => {
    const out = normalizeMessage({
      role: "toolResult",
      content: [{ type: "text", text: "boom" }],
      toolCallId: "t1",
      toolName: "fetch",
      isError: true,
    }) as any;
    expect(out.status).toBe("error");
    expect(out.content).toBe("boom");
  });

  it("passes non-assistant / non-toolResult messages through unchanged", () => {
    const msg = { role: "user", content: "hi" };
    expect(normalizeMessage(msg)).toBe(msg);
  });
});

describe("extractToolCalls", () => {
  it("returns undefined when the message has no tool-call blocks", () => {
    expect(
      extractToolCalls({
        role: "assistant",
        content: [{ type: "text", text: "hello" }],
      }),
    ).toBeUndefined();
  });

  it("pulls id / name / args from camelCase toolCall blocks", () => {
    const calls = extractToolCalls({
      role: "assistant",
      content: [
        { type: "text", text: "checking" },
        { type: "toolCall", id: "t1", name: "fetch", arguments: { url: "x" } },
        { type: "toolCall", id: "t2", name: "search", arguments: { q: "y" } },
      ],
    });
    expect(calls).toEqual([
      { id: "t1", name: "fetch", args: { url: "x" } },
      { id: "t2", name: "search", args: { q: "y" } },
    ]);
  });

  it("also reads canonical `tool_use` blocks with `input`", () => {
    const calls = extractToolCalls({
      role: "assistant",
      content: [{ type: "tool_use", id: "t1", name: "fetch", input: { url: "x" } }],
    });
    expect(calls).toEqual([{ id: "t1", name: "fetch", args: { url: "x" } }]);
  });
});
