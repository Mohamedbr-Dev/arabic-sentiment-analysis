// Runs the model in the browser with transformers.js, as src/day3_api.py does on the server.
import { AutoTokenizer, AutoModelForSequenceClassification, env }
  from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

export const MAX_TOKENS = 64;        // what the model was trained on
export const MIN_CONFIDENCE = 0.6;   // below this the model was right only 58% of the time
const LABELS = ["negative", "neutral", "positive"];

env.allowRemoteModels = false;       // only the files next to this page
env.allowLocalModels = true;
env.localModelPath = new URL(".", import.meta.url).href;

export async function loadModel(folder, onProgress) {
  const options = { progress_callback: onProgress };
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(folder, options),
    AutoModelForSequenceClassification.from_pretrained(folder, { ...options, dtype: "q8" }),
  ]);
  return { tokenizer, model };
}

// `text` must already be cleaned (text_clean.js).
export async function classify(tokenizer, model, text) {
  const full = tokenizer(text);
  const truncated = full.input_ids.dims[1] > MAX_TOKENS;
  const inputs = truncated ? tokenizer(text, { truncation: true, max_length: MAX_TOKENS }) : full;
  if (truncated) {
    // transformers.js cuts off the closing [SEP]; Python keeps it, and the model was trained so
    const ids = inputs.input_ids.data;
    ids[ids.length - 1] = typeof ids[0] === "bigint" ? BigInt(tokenizer.sep_token_id)
                                                     : tokenizer.sep_token_id;
  }
  const logits = Array.from((await model(inputs)).logits.data);
  const top = Math.max(...logits);
  const exp = logits.map(v => Math.exp(v - top));
  const sum = exp.reduce((a, b) => a + b, 0);
  const probabilities = Object.fromEntries(LABELS.map((label, i) => [label, exp[i] / sum]));
  const sentiment = LABELS[exp.indexOf(Math.max(...exp))];
  const confidence = probabilities[sentiment];
  return { sentiment, confidence, uncertain: confidence < MIN_CONFIDENCE, truncated, probabilities };
}
