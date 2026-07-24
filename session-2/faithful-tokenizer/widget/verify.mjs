#!/usr/bin/env node
// Verification harness: runs the SAME JS BPE engine used in index.html
// (imported from engine.mjs) against the four corpus files and prints
// per-language token counts + score, to confirm the in-browser re-run matches
// HuggingFace `tokenizers` exactly.
//
//   node verify.mjs                       # verify corpora against targets
//   node verify.mjs --emit-stats          # also (re)write stats.json
//   node verify.mjs --emit-edges          # also write _edges.json for python xcheck
//   node verify.mjs --strings <file.json> # encode/decode an array of strings (json)
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildEngine, faithfulUnits } from "./engine.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const tk = JSON.parse(readFileSync(join(HERE, "tokenizer.json"), "utf8"));
const engine = buildEngine(tk);
const nonspace = (s) => s.replace(/\s+/g, "");

// ---- --strings mode: emit ids + decode for a list of strings (python xcheck) ----
const sArgIdx = process.argv.indexOf("--strings");
if (sArgIdx !== -1) {
  const strs = JSON.parse(readFileSync(process.argv[sArgIdx + 1], "utf8"));
  const out = strs.map((s) => {
    const { ids } = engine.encode(s);
    return { s, ids, decode: engine.decode(ids) };
  });
  process.stdout.write(JSON.stringify(out));
  process.exit(0);
}

const LANGS = ["en", "hi", "te", "mr"];
const NAMES = { en: "English", hi: "Hindi", te: "Telugu", mr: "Marathi" };
const TARGET = {
  en: { tokens: 120409, units: 186426 },
  hi: { tokens: 56158, units: 88359 },
  te: { tokens: 22146, units: 36293 },
  mr: { tokens: 18870, units: 29766 },
};

const rows = {};
let allMatch = true;
for (const l of LANGS) {
  const text = readFileSync(join(HERE, "corpus", `${l}.faithful.txt`), "utf8");
  const { ids } = engine.encode(text);
  const units = faithfulUnits(text);
  const tokens = ids.length;
  const rt = nonspace(text) === nonspace(engine.decode(ids));
  const okTok = tokens === TARGET[l].tokens;
  const okUnit = units === TARGET[l].units;
  if (!okTok || !okUnit || !rt) allMatch = false;
  rows[l] = { tokens, units, fertility: tokens / units, roundtrip: rt };
  console.log(
    `${NAMES[l].padEnd(8)} tokens=${tokens} (target ${TARGET[l].tokens} ${okTok ? "OK" : "MISMATCH"})` +
    `  units=${units} (target ${TARGET[l].units} ${okUnit ? "OK" : "MISMATCH"})` +
    `  fertility=${(tokens / units).toFixed(4)}  roundtrip=${rt ? "PASS" : "FAIL"}`
  );
}

const ferts = LANGS.map((l) => rows[l].fertility);
const min = Math.min(...ferts), max = Math.max(...ferts);
const spread = max - min;
const score = 1000 / spread;
const hiPen = Math.exp(Math.max(0, rows.hi.fertility / 1.2 - 1));
console.log("\nspread =", spread.toFixed(6), " score =", score.toFixed(4),
  " hindi_penalty =", hiPen.toFixed(6), " adjusted =", (score / hiPen).toFixed(4));
console.log("ALL COUNTS MATCH TARGETS + ROUNDTRIP:", allMatch ? "YES" : "NO");

// ---- edge strings (JS) ----
const EDGE = [
  "India, officially the Republic of India [1]",
  "India's population is 1,428,627,663.",
  "See [https://example.com] (ref). {x} <y> #tag *em* `code` | table |",
  "emoji 😀 test",
  "CJK 中文 test",
  "rare ℤ ∑ char",
  "tab\there\nnewline",
  "  leading spaces",
  "▁ literal metaspace",
  "Café naïve résumé — em–dash",
  "money £ € ¥ ₹ ½ × § … ‰ †",
  "math 𝕳𝖊𝖑𝖑𝖔 ℤ ∑ ∫ √ ≈",
  "accents ï ã ó Þ ø å ç ñ",
  "emoji 😀🎉👨‍👩‍👧 survives",
];
if (process.argv.includes("--emit-edges")) {
  const edgeOut = {};
  for (const s of EDGE) edgeOut[s] = { ids: engine.encode(s).ids };
  writeFileSync(join(HERE, "_edges.json"), JSON.stringify(edgeOut));
  console.log("wrote _edges.json");
}

if (process.argv.includes("--emit-stats")) {
  const stats = {
    languages: NAMES,
    per_language: Object.fromEntries(LANGS.map((l) => [l, {
      tokens: rows[l].tokens, faithful_units: rows[l].units,
      fertility: rows[l].fertility, roundtrip_nonspace_preserved: rows[l].roundtrip,
    }])),
    sorted_fertilities: LANGS.slice().sort((a, b) => rows[a].fertility - rows[b].fertility)
      .map((l) => ({ lang: l, fertility: rows[l].fertility })),
    min_fertility: min, max_fertility: max, spread,
    score, hindi_penalty_factor: hiPen, adjusted_score: score / hiPen,
    vocab_size: engine.VOCAB.size, merges: engine.model.merges.length,
    all_counts_match_targets: allMatch,
    computed_by: "widget/verify.mjs via engine.mjs (same JS BPE engine as index.html)",
  };
  writeFileSync(join(HERE, "stats.json"), JSON.stringify(stats, null, 2));
  console.log("wrote stats.json");
}
