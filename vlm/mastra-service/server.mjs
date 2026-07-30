// Local Mastra service for VLM feature extraction.
// The Python eval backend POSTs {model, prompt, images[]} to /extract and gets
// back {text, usage}. This is the only layer that talks to the model provider,
// all evaluation, metrics and UI stay in Python.
import { config } from "dotenv";
import { fileURLToPath } from "url";
import path from "path";
import express from "express";
import { Agent } from "@mastra/core/agent";
import { createAnthropic } from "@ai-sdk/anthropic";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Reuse the SAME key the Python side uses: vlm/.env (one directory up).
config({ path: path.join(__dirname, "..", ".env") });

const KEY = process.env.ANTHROPIC_API_KEY;
if (!KEY) {
  console.error("[mastra] ANTHROPIC_API_KEY missing in vlm/.env — cannot start.");
  process.exit(1);
}

const anthropic = createAnthropic({ apiKey: KEY });

// Map the friendly default "claude" to a concrete Anthropic model id.
function resolveModel(m) {
  if (!m || m === "claude") return "claude-sonnet-4-5";
  return m;
}

// Turn a data URL (or http URL) into an AI SDK image content part. Parsing the
// data URL ourselves and passing raw bytes + explicit mediaType avoids the
// provider guessing the wrong media type.
function toImagePart(url) {
  const m = /^data:(.+?);base64,(.*)$/s.exec(url);
  if (m) {
    return { type: "image", image: Buffer.from(m[2], "base64"), mediaType: m[1] };
  }
  return { type: "image", image: new URL(url) };
}

// Normalize usage across AI SDK v4/v5 field names.
function normUsage(u = {}) {
  const inTok = u.inputTokens ?? u.promptTokens ?? null;
  const outTok = u.outputTokens ?? u.completionTokens ?? null;
  const total = u.totalTokens ?? (inTok != null && outTok != null ? inTok + outTok : null);
  return { input_tokens: inTok, output_tokens: outTok, total_tokens: total };
}

const app = express();
app.use(express.json({ limit: "64mb" }));

app.get("/health", (_req, res) => res.json({ ok: true, service: "vlm-mastra" }));

app.post("/extract", async (req, res) => {
  const { model, prompt, images = [], maxTokens = 1500 } = req.body || {};
  if (!prompt) return res.status(400).json({ error: "prompt is required" });

  try {
    const agent = new Agent({
      name: "vlm-extractor",
      instructions:
        "You extract structured data from 2D mechanical engineering drawings. " +
        "Follow the user's instructions exactly and output only what they ask for.",
      model: anthropic(resolveModel(model)),
    });

    const content = [
      ...images.map(toImagePart),
      { type: "text", text: prompt },
    ];

    const result = await agent.generate([{ role: "user", content }], {
      modelSettings: { maxOutputTokens: maxTokens },
    });

    res.json({
      text: result?.text ?? "",
      usage: normUsage(result?.usage),
      model: resolveModel(model),
    });
  } catch (e) {
    res.json({ error: String(e?.message || e) });
  }
});

const PORT = process.env.MASTRA_PORT || 8787;
app.listen(PORT, "127.0.0.1", () =>
  console.log(`[mastra] VLM service listening on http://127.0.0.1:${PORT}`)
);
