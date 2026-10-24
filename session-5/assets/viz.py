"""
viz.py — generate static SVG charts for the V5 mixture plan (stdlib only, deterministic).
Outputs into this folder: mixture.svg, supply_gap.svg, indic_tiers.svg,
proxy.svg, curriculum.svg. Colorblind-safe palette; violet = scarce lanes.
"""
import os, math

OUT = os.path.dirname(__file__)

# colorblind-safe palette; violet for the scarce Indic lane (session's scarcity marker)
C = {
    "web": "#4E79A7", "code": "#59A14F", "math": "#E15759", "indic": "#B07AA1",
    "agentic": "#F28E2B", "longctx": "#76B7B2", "other": "#9C9C9C",
    "ink": "#222", "grid": "#ddd", "bg": "#ffffff", "muted": "#666",
}
FONT = 'font-family="ui-sans-serif,Segoe UI,Helvetica,Arial,sans-serif"'


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{title}">'
            f'<rect width="{w}" height="{h}" fill="{C["bg"]}"/>{body}</svg>')


def txt(x, y, s, size=13, fill=None, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill or C["ink"]}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {FONT}>{s}</text>')


# ---------------------------------------------------------------- 1. mixture
def mixture():
    lanes = [("web", "General web"), ("code", "Code"), ("math", "Math/reason"),
             ("indic", "Indic"), ("agentic", "Agentic"), ("longctx", "Long-ctx"),
             ("other", "Other")]
    main = {"web": 45, "code": 20, "math": 8, "indic": 13, "agentic": 1.5, "longctx": 2.5, "other": 10}
    anneal = {"web": 15, "code": 22, "math": 22, "indic": 25, "agentic": 10, "longctx": 4, "other": 2}
    W, H = 720, 340
    x0, y0, bw, bh = 90, 40, 200, 230
    b = [txt(W / 2, 24, "V5 mixture — main pretraining vs annealing preset", 15, C["ink"], "middle", "bold")]
    for i, (col, (k, label)) in enumerate([(0, l) for l in lanes]):
        pass
    for bi, (data, bx, name) in enumerate([(main, x0, "Main (13.6T)"), (anneal, x0 + 260, "Anneal (~480B)")]):
        y = y0
        b.append(txt(bx + bw / 2, y0 - 8, name, 13, C["muted"], "middle", "bold"))
        for k, label in lanes:
            frac = data[k] / 100.0
            seg = frac * bh
            b.append(f'<rect x="{bx}" y="{y:.1f}" width="{bw}" height="{seg:.1f}" '
                     f'fill="{C[k]}" stroke="#fff" stroke-width="1"/>')
            if seg > 12:
                b.append(txt(bx + bw / 2, y + seg / 2 + 4, f"{data[k]:g}%", 11, "#fff", "middle", "bold"))
            y += seg
    # legend
    ly = y0 + bh + 30
    lx = x0
    for k, label in lanes:
        b.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{C[k]}"/>')
        b.append(txt(lx + 16, ly, label, 11, C["ink"]))
        lx += 92
    return svg(W, H, "".join(b), "V5 mixture main vs anneal")


# ---------------------------------------------------------------- 2. supply gap (log)
def supply_gap():
    # (label, need tokens, have-unique tokens)
    rows = [("Indic (target 13% = 2.08T)", 2.08e12, 64.3e9, "indic"),
            ("Agentic (target 1.5% = 240B)", 240e9, 0.01e9, "agentic"),
            ("Code (target 20% = 3.2T)", 3.2e12, 1.0e12, "code"),
            ("Math (target 8% = 1.28T)", 1.28e12, 90e9, "math"),
            ("General web (45% = 7.2T)", 7.2e12, 25e12, "web")]
    W, H = 760, 300
    x0, y0, rw = 320, 50, 380
    lo, hi = math.log10(1e7), math.log10(3e13)

    def sx(v):
        return x0 + rw * (math.log10(max(v, 1e7)) - lo) / (hi - lo)
    b = [txt(W / 2, 24, "Supply vs target (log scale) — need ▮  vs  unique supply ●", 15, C["ink"], "middle", "bold")]
    # gridlines at 1e8..1e13
    for e in range(8, 14):
        gx = sx(10 ** e)
        b.append(f'<line x1="{gx:.1f}" y1="{y0-6}" x2="{gx:.1f}" y2="{y0+len(rows)*44}" stroke="{C["grid"]}"/>')
        lbl = {8: "100M", 9: "1B", 10: "10B", 11: "100B", 12: "1T", 13: "10T"}[e]
        b.append(txt(gx, y0 + len(rows) * 44 + 16, lbl, 10, C["muted"], "middle"))
    y = y0 + 18
    for label, need, have, col in rows:
        b.append(txt(x0 - 12, y + 4, label, 11, C["ink"], "end"))
        b.append(f'<rect x="{x0:.1f}" y="{y-8}" width="{sx(need)-x0:.1f}" height="16" fill="{C[col]}" opacity="0.30"/>')
        b.append(f'<line x1="{sx(need):.1f}" y1="{y-10}" x2="{sx(need):.1f}" y2="{y+6}" stroke="{C[col]}" stroke-width="2"/>')
        b.append(f'<circle cx="{sx(have):.1f}" cy="{y-1}" r="5" fill="{col if False else C[col]}" stroke="#fff"/>')
        gap = need / have if have else 0
        if gap > 5:
            b.append(txt(sx(need) + 6, y + 3, f"{gap:.0f}× gap", 10, C["muted"]))
        y += 44
    return svg(W, H, "".join(b), "supply vs target")


# ---------------------------------------------------------------- 3. indic tiers
def indic_tiers():
    tiers = [("Verified native (A)", 22, "~64B, ~7 epochs", "#6a3d9a"),
             ("Unverified crawl (B)", 18, "~90B, 2-4 epochs", "#9b6fb0"),
             ("Translated (C→B)", 40, "IndicTrans2 MT", "#c3a0d0"),
             ("Synthetic (C)", 20, "teacher-LLM gen", "#e0cce8")]
    W, H = 720, 170
    x0, y0, bw = 40, 70, 640
    b = [txt(W / 2, 26, "Indic lane (2.08T) split across provenance tiers", 15, C["ink"], "middle", "bold"),
         txt(W / 2, 46, "only ~22% verified-native — ~60% translated+synthetic (stated in data card)", 11, C["muted"], "middle")]
    x = x0
    for label, pct, note, col in tiers:
        seg = bw * pct / 100
        b.append(f'<rect x="{x:.1f}" y="{y0}" width="{seg:.1f}" height="46" fill="{col}" stroke="#fff" stroke-width="1"/>')
        b.append(txt(x + seg / 2, y0 + 22, f"{pct}%", 13, "#fff" if pct > 20 else C["ink"], "middle", "bold"))
        b.append(txt(x + seg / 2, y0 + 62, label, 10, C["ink"], "middle"))
        b.append(txt(x + seg / 2, y0 + 76, note, 9, C["muted"], "middle"))
        x += seg
    return svg(W, H, "".join(b), "indic tiers")


# ---------------------------------------------------------------- 4. proxy
def proxy():
    lanes = ["indic", "math", "code", "en"]
    arms = [("A naive", {"indic": 48.28, "math": 12.76, "code": 17.37, "en": 7.32}, "#c9c9c9"),
            ("B V5-main", {"indic": 21.42, "math": 5.63, "code": 15.48, "en": 8.29}, "#B07AA1"),
            ("C V5-anneal", {"indic": 19.26, "math": 3.70, "code": 15.54, "en": 8.93}, "#6a3d9a")]
    W, H = 720, 340
    x0, y0, gh = 70, 60, 220
    maxv = 50
    gw = (W - x0 - 40) / len(lanes)
    b = [txt(W / 2, 24, "Proxy ablation — held-out perplexity (lower = better)", 15, C["ink"], "middle", "bold"),
         txt(W / 2, 42, "real char-LM, equal compute; 3 mixtures on real data", 11, C["muted"], "middle")]
    b.append(f'<line x1="{x0}" y1="{y0+gh}" x2="{W-30}" y2="{y0+gh}" stroke="{C["ink"]}"/>')
    for gi, lane in enumerate(lanes):
        gx = x0 + gi * gw
        bw = gw / (len(arms) + 1)
        for ai, (aname, data, col) in enumerate(arms):
            v = data[lane]
            hh = gh * v / maxv
            bx = gx + bw * (ai + 0.5)
            b.append(f'<rect x="{bx:.1f}" y="{y0+gh-hh:.1f}" width="{bw*0.9:.1f}" height="{hh:.1f}" fill="{col}"/>')
            b.append(txt(bx + bw * 0.45, y0 + gh - hh - 4, f"{v:.0f}", 9, C["ink"], "middle"))
        b.append(txt(gx + gw / 2, y0 + gh + 18, lane, 12, C["ink"], "middle", "bold"))
    # legend
    lx, ly = x0, y0 + gh + 44
    for aname, _, col in arms:
        b.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{col}"/>')
        b.append(txt(lx + 16, ly, aname, 11, C["ink"]))
        lx += 150
    b.append(txt(x0, ly + 24, "Indic 48→21 (−21%), math 13→6 (−32%); English 7.3→8.3 (+6% cost, proxy-inflated)", 11, C["muted"]))
    return svg(W, H, "".join(b), "proxy result")


# ---------------------------------------------------------------- 5. curriculum
def curriculum():
    # phases across training progress; stacked shares (web,code,math,indic,agentic,longctx,other)
    order = ["web", "code", "math", "indic", "agentic", "longctx", "other"]
    pts = [  # (progress%, shares)
        (0,   {"web": 70, "code": 10, "math": 4, "indic": 8, "agentic": 0.5, "longctx": 0, "other": 7.5}),
        (30,  {"web": 55, "code": 16, "math": 6, "indic": 11, "agentic": 1, "longctx": 0, "other": 11}),
        (70,  {"web": 33, "code": 24, "math": 12, "indic": 15, "agentic": 2, "longctx": 4, "other": 10}),
        (88,  {"web": 28, "code": 22, "math": 12, "indic": 15, "agentic": 2, "longctx": 13, "other": 8}),
        (100, {"web": 15, "code": 22, "math": 22, "indic": 25, "agentic": 10, "longctx": 4, "other": 2}),
    ]
    W, H = 760, 330
    x0, y0, pw, ph = 50, 50, 660, 220
    b = [txt(W / 2, 24, "Curriculum — lane share over training progress (→ anneal)", 15, C["ink"], "middle", "bold")]

    def X(p): return x0 + pw * p / 100

    def Y(v): return y0 + ph * (1 - v / 100)
    # build stacked polygons
    cum_prev = {p: 0 for p, _ in pts}
    base = {p: 0 for p, _ in pts}
    for k in order:
        top = []
        bot = []
        for p, sh in pts:
            b0 = base[p]
            b1 = b0 + sh[k]
            base[p] = b1
            top.append((X(p), Y(b1)))
            bot.append((X(p), Y(b0)))
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in top) + " " + \
                  " ".join(f"{x:.1f},{y:.1f}" for x, y in reversed(bot))
        b.append(f'<polygon points="{pts_str}" fill="{C[k]}" stroke="#fff" stroke-width="0.5" opacity="0.9"/>')
    # phase markers
    for p, lbl in [(2, "seed"), (30, "general"), (70, "capability"), (88, "long-ctx"), (100, "anneal")]:
        b.append(f'<line x1="{X(p):.1f}" y1="{y0}" x2="{X(p):.1f}" y2="{y0+ph}" stroke="#fff" stroke-dasharray="3 3" opacity="0.6"/>')
        b.append(txt(X(p), y0 + ph + 16, lbl, 10, C["muted"], "middle"))
    b.append(txt(x0, y0 - 6, "100%", 10, C["muted"]))
    b.append(txt(x0, y0 + ph + 34, "violet Indic band held ~8-10% floor throughout; scarce lanes surge only in anneal", 11, C["muted"]))
    return svg(W, H, "".join(b), "curriculum")


for name, fn in [("mixture", mixture), ("supply_gap", supply_gap), ("indic_tiers", indic_tiers),
                 ("proxy", proxy), ("curriculum", curriculum)]:
    open(os.path.join(OUT, f"{name}.svg"), "w").write(fn())
    print("wrote", name + ".svg")
