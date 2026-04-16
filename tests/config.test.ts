import { describe, expect, it } from "vitest";
import { readPluginConfig } from "../src/config.js";

describe("readPluginConfig", () => {
  it("applies defaults for missing fields", () => {
    const cfg = readPluginConfig({ langsmithApiKey: "k" });
    expect(cfg.endpoint).toBe("https://api.smith.langchain.com");
    expect(cfg.projectName).toBe("openclaw");
    expect(cfg.traceAgentTurns).toBe(true);
    expect(cfg.traceToolCalls).toBe(true);
    expect(cfg.samplingRate).toBe(1);
    expect(cfg.debug).toBe(false);
  });

  it("clamps sampling rate into [0, 1]", () => {
    expect(readPluginConfig({ langsmithApiKey: "k", tracingSamplingRate: 2 }).samplingRate).toBe(1);
    expect(readPluginConfig({ langsmithApiKey: "k", tracingSamplingRate: -3 }).samplingRate).toBe(
      0,
    );
    expect(readPluginConfig({ langsmithApiKey: "k", tracingSamplingRate: 0.25 }).samplingRate).toBe(
      0.25,
    );
  });

  it("prefers the config api key over LANGSMITH_API_KEY", () => {
    const prev = process.env.LANGSMITH_API_KEY;
    process.env.LANGSMITH_API_KEY = "env-key";
    try {
      expect(readPluginConfig({ langsmithApiKey: "config-key" }).apiKey).toBe("config-key");
      expect(readPluginConfig({}).apiKey).toBe("env-key");
    } finally {
      if (prev === undefined) delete process.env.LANGSMITH_API_KEY;
      else process.env.LANGSMITH_API_KEY = prev;
    }
  });
});
