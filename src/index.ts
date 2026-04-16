import type { OpenClawPluginApi, OpenClawPluginDefinition } from "openclaw/plugin-sdk/plugin-entry";
import { readPluginConfig } from "./config.js";
import { attachLog } from "./log.js";
import { buildLangsmithClient } from "./langsmith-bridge.js";
import { TurnRecorder } from "./turn-recorder.js";

const PLUGIN_ID = "openclaw-langsmith-trace";
const DEFAULT_SESSION_KEY = "default";
const FLUSH_TIMEOUT_MS = 5_000;

const plugin: OpenClawPluginDefinition = {
  id: PLUGIN_ID,
  name: "LangSmith Tracing",
  description:
    "LangSmith tracing for OpenClaw agent turns, inner LLM calls, tool calls, and subagents. Built on the official langsmith SDK.",

  register(api: OpenClawPluginApi): void {
    const cfg = readPluginConfig(api.pluginConfig);
    const log = attachLog(api.logger, cfg.debug);

    if (!cfg.apiKey) {
      log.warn("no LangSmith API key set — tracing disabled");
      return;
    }

    const client = buildLangsmithClient(cfg, log);
    const recorder = new TurnRecorder(client, cfg, log);

    if (cfg.traceAgentTurns) {
      api.on("session_end", (_event, ctx) => {
        void recorder.onSessionEnd(ctx.sessionKey ?? DEFAULT_SESSION_KEY);
      });

      api.on("llm_input", (event, ctx) => {
        void recorder.onTurnStart(ctx.sessionKey ?? DEFAULT_SESSION_KEY, event, ctx);
      });

      api.on("before_message_write", (event, ctx) => {
        void recorder.onMessageWrite(ctx.sessionKey ?? DEFAULT_SESSION_KEY, event.message);
      });

      api.on("agent_end", (event, ctx) => {
        void recorder.onTurnEnd(
          ctx.sessionKey ?? DEFAULT_SESSION_KEY,
          event.success,
          event.durationMs,
          event.error,
        );
      });

      api.on("subagent_spawned", (event, ctx) => {
        recorder.onSubagentSpawned(ctx.requesterSessionKey ?? DEFAULT_SESSION_KEY, event);
      });

      api.on("subagent_ended", (event, ctx) => {
        void recorder.onSubagent(ctx.requesterSessionKey ?? DEFAULT_SESSION_KEY, event);
      });
    }

    if (cfg.traceToolCalls) {
      api.on("before_tool_call", (event, ctx) => {
        void recorder.onToolStart(ctx.sessionKey ?? DEFAULT_SESSION_KEY, event, ctx);
      });

      api.on("after_tool_call", (event, ctx) => {
        void recorder.onToolEnd(ctx.sessionKey ?? DEFAULT_SESSION_KEY, event, ctx);
      });
    }

    api.registerService({
      id: PLUGIN_ID,
      start: () => {
        // No-op: the tracer is ready as soon as register() completes.
      },
      stop: async () => {
        log.info("service stop — flushing pending traces");
        await recorder.shutdown();
        try {
          // Bound the flush so a stuck upload can't block gateway shutdown.
          await Promise.race([
            client.awaitPendingTraceBatches(),
            new Promise<void>((_, reject) =>
              setTimeout(
                () => reject(new Error(`flush timed out after ${FLUSH_TIMEOUT_MS}ms`)),
                FLUSH_TIMEOUT_MS,
              ).unref(),
            ),
          ]);
        } catch (err) {
          log.warn(`flush failed: ${err instanceof Error ? err.message : String(err)}`);
        }
        // Release SDK-owned timers (prompt-cache refresh) so the process can exit.
        client.cleanup();
      },
    });

    log.info(`ready — project=${cfg.projectName} sampling=${cfg.samplingRate}`);
  },
};

export default plugin;
