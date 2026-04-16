import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";

const TAG = "[langsmith]";

export interface Log {
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
  /** No-op unless the plugin's `debug` config is true. */
  debug(msg: string): void;
}

/**
 * Wraps the host's logger so every line gets a `[langsmith]` tag and
 * `debug()` is silenced unless the plugin has debug mode enabled.
 */
export function attachLog(host: PluginLogger, verbose: boolean): Log {
  const debugSink = host.debug?.bind(host) ?? host.info.bind(host);
  const tag = (msg: string): string => `${TAG} ${msg}`;
  return {
    info: (msg) => host.info(tag(msg)),
    warn: (msg) => host.warn(tag(msg)),
    error: (msg) => host.error(tag(msg)),
    debug: (msg) => {
      if (verbose) debugSink(tag(msg));
    },
  };
}
