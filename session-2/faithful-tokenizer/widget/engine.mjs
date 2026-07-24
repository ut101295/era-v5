// ======================================================================
// Faithful BPE tokenizer engine — a JS port of HuggingFace `tokenizers`.
// This is the SHARED SOURCE OF TRUTH. The identical code is inlined into
// index.html (self-contained widget); verify.mjs imports it for testing.
//
// Tokenizer config (from tokenizer.json):
//   model:        BPE, byte_fallback=true, fuse_unk=false, unk="[UNK]"
//   normalizer:   NFC
//   pre_tokenizer/decoder: Metaspace(replacement=▁, prepend_scheme=never)
//   NOTE: the vocab contains all 256 <0xHH> byte tokens, so byte_fallback IS
//   functional — any symbol not in vocab is emitted as its UTF-8 bytes and
//   round-trips exactly on decode (emoji, math letters, £/€/…, accents, etc.).
//   [UNK] therefore never appears for real text. (10000 vocab = base + 9441
//   merges + 256 byte tokens.)
// ======================================================================
export const METASPACE = "▁"; // ▁

export function buildEngine(tk) {
  const model = tk.model;
  const VOCAB = new Map(Object.entries(model.vocab));
  const RANK = new Map();
  model.merges.forEach((m, i) => {
    const p = Array.isArray(m) ? m : m.split(" ");
    RANK.set(p[0] + " " + p[1], i);
  });
  const UNK = model.unk_token;
  const UNK_ID = VOCAB.get(UNK);
  const BYTE_FALLBACK = !!model.byte_fallback;
  const FUSE_UNK = !!model.fuse_unk;
  const enc = new TextEncoder();

  const SPECIAL_IDS = new Set(
    (tk.added_tokens || []).filter((t) => t.special).map((t) => t.id)
  );
  const ID2TOK = new Map();
  for (const [t, i] of VOCAB) ID2TOK.set(i, t);

  // --- Heap ordered by (rank asc, position asc) => leftmost tie-break ---
  const lt = (x, y) => x[0] < y[0] || (x[0] === y[0] && x[2] < y[2]);
  class Heap {
    constructor() { this.a = []; }
    push(x) {
      const a = this.a; a.push(x); let i = a.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (!lt(a[i], a[p])) break; [a[p], a[i]] = [a[i], a[p]]; i = p; }
    }
    pop() {
      const a = this.a; const t = a[0]; const l = a.pop();
      if (a.length) {
        a[0] = l; let i = 0; const n = a.length;
        for (;;) {
          let L = 2 * i + 1, R = L + 1, s = i;
          if (L < n && lt(a[L], a[s])) s = L;
          if (R < n && lt(a[R], a[s])) s = R;
          if (s === i) break; [a[s], a[i]] = [a[i], a[s]]; i = s;
        }
      }
      return t;
    }
    get size() { return this.a.length; }
  }

  // BPE over an array of symbol strings -> merged symbol strings.
  function bpe(symbols) {
    const N = symbols.length;
    if (N <= 1) return symbols;
    const val = symbols.slice();
    const prev = new Int32Array(N), next = new Int32Array(N), ver = new Int32Array(N);
    const alive = new Uint8Array(N).fill(1);
    for (let i = 0; i < N; i++) { prev[i] = i - 1; next[i] = i + 1 < N ? i + 1 : -1; }
    const heap = new Heap();
    const add = (i) => {
      const j = next[i]; if (j === -1) return;
      const r = RANK.get(val[i] + " " + val[j]);
      if (r !== undefined) heap.push([r, ver[i], i]);
    };
    for (let i = 0; i < N; i++) add(i);
    while (heap.size) {
      const [r, seq, i] = heap.pop();
      if (!alive[i] || ver[i] !== seq) continue;
      const j = next[i];
      if (j === -1 || !alive[j]) continue;
      if (RANK.get(val[i] + " " + val[j]) !== r) continue;
      val[i] = val[i] + val[j]; alive[j] = 0;
      const k = next[j]; next[i] = k; if (k !== -1) prev[k] = i;
      ver[i]++; if (prev[i] !== -1) { ver[prev[i]]++; add(prev[i]); } add(i);
    }
    const out = [];
    for (let i = 0; i !== -1 && i < N; i = next[i]) if (alive[i]) out.push(val[i]);
    return out;
  }

  // Metaspace pre-tokenization (replacement ▁, prepend_scheme never, split true).
  // Only U+0020 space -> ▁. Each ▁ begins a new pre-token; the run before the
  // first ▁ is its own pre-token. Newlines/tabs are NOT replaced.
  function preTokenize(text) {
    const s = text.replace(/ /g, METASPACE);
    const pre = [];
    let cur = "";
    for (const ch of s) {
      if (ch === METASPACE) { if (cur !== "") pre.push(cur); cur = METASPACE; }
      else cur += ch;
    }
    if (cur !== "") pre.push(cur);
    return pre;
  }

  // One merged symbol -> ids (byte_fallback then unk). fuse_unk handled by caller.
  function symbolToIds(sym, outIds, outToks) {
    const id = VOCAB.get(sym);
    if (id !== undefined) { outIds.push(id); outToks.push(sym); return; }
    if (BYTE_FALLBACK) {
      const bytes = enc.encode(sym);
      const btoks = [];
      let ok = true;
      for (const b of bytes) {
        const bt = "<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">";
        if (VOCAB.has(bt)) btoks.push(bt); else { ok = false; break; }
      }
      if (ok) { for (const bt of btoks) { outIds.push(VOCAB.get(bt)); outToks.push(bt); } return; }
    }
    outIds.push(UNK_ID); outToks.push(UNK);
  }

  function encode(text) {
    const norm = text.normalize("NFC");
    const pre = preTokenize(norm);
    const ids = [], toks = [];
    for (const pt of pre) {
      const merged = bpe(Array.from(pt));
      for (const sym of merged) {
        if (FUSE_UNK && VOCAB.get(sym) === undefined) {
          const before = ids.length;
          symbolToIds(sym, ids, toks);
          if (ids[before] === UNK_ID && before > 0 && ids[before - 1] === UNK_ID) {
            ids.splice(before, 1); toks.splice(before, 1);
          }
        } else {
          symbolToIds(sym, ids, toks);
        }
      }
    }
    return { ids, tokens: toks };
  }

  // decode: ids -> tokens -> ByteFallback -> Metaspace(▁->space)
  function decode(ids, skipSpecial = true) {
    const toks = [];
    for (const id of ids) {
      if (skipSpecial && SPECIAL_IDS.has(id)) continue;
      toks.push(ID2TOK.get(id) ?? "");
    }
    const out = [];
    let buf = [];
    const flush = () => {
      if (buf.length) { out.push(new TextDecoder().decode(Uint8Array.from(buf))); buf = []; }
    };
    for (const t of toks) {
      const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(t);
      if (m) buf.push(parseInt(m[1], 16));
      else { flush(); out.push(t); }
    }
    flush();
    return out.join("").replace(/▁/g, " ");
  }

  return { encode, decode, VOCAB, RANK, model, ID2TOK, SPECIAL_IDS };
}

// Faithful-unit regex. Uses \p{White_Space} (NOT \s) so U+FEFF is treated as a
// visible unit, matching Python's `regex` module exactly (JS \s matches U+FEFF).
export const FAITHFUL_UNIT_RE = /[\p{L}\p{M}\p{N}]+|[^\p{White_Space}\p{L}\p{M}\p{N}]/gu;
export const faithfulUnits = (t) => (t.match(FAITHFUL_UNIT_RE) || []).length;
