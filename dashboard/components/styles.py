"""Global CSS for the dashboard (dark nautical theme)."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #070d1a; --panel: rgba(15, 26, 46, 0.78); --panel-2: rgba(22, 36, 62, 0.85);
  --line: rgba(148, 163, 184, 0.14); --text: #e2e8f0; --muted: #94a3b8; --accent: #22d3ee;
  --accent-2: #38bdf8; --good: #22c55e; --warn: #f59e0b; --bad: #ef4444;
}
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3, h4, .sc-title { font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; }
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(34, 211, 238, 0.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(99, 102, 241, 0.10), transparent 60%),
    linear-gradient(180deg, #070d1a 0%, #0a1426 100%);
}
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1528 0%, #081120 100%); border-right: 1px solid var(--line); }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; font-size: 0.95rem; padding: 0.55rem 0.9rem; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; }
div[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, #06b6d4, #6366f1) !important; }

/* header */
.sc-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  padding: 1rem 1.25rem; border-radius: 18px; border:1px solid var(--line);
  background: linear-gradient(120deg, rgba(8,47,73,.85) 0%, rgba(30,27,75,.75) 100%); margin-bottom: 0.9rem;
  box-shadow: 0 10px 30px rgba(0,0,0,.25); }
.sc-brand { display:flex; align-items:center; gap:.85rem; }
.sc-logo { width:46px; height:46px; border-radius:12px; display:grid; place-items:center; font-size:26px;
  background: linear-gradient(135deg, #06b6d4, #6366f1); box-shadow: 0 6px 20px rgba(34,211,238,.35); }
.sc-name { font-family:'Space Grotesk'; font-size: 1.65rem; font-weight:700; line-height:1; color:#f8fafc; }
.sc-name span { background: linear-gradient(90deg,#67e8f9,#a5b4fc); -webkit-background-clip:text; color:transparent; }
.sc-tag { color: var(--muted); font-size:.86rem; margin-top:.2rem; }
.sc-meta { display:flex; gap:.5rem; flex-wrap:wrap; }
.sc-pill { padding:.35rem .7rem; border-radius:999px; border:1px solid var(--line); background: rgba(2,6,23,.45);
  font-size:.8rem; color:#cbd5e1; white-space:nowrap; }
.sc-pill b { color:#f8fafc; }

/* cards */
.sc-card { background: var(--panel); border:1px solid var(--line); border-radius:16px; padding: 1rem 1.1rem;
  box-shadow: 0 8px 24px rgba(0,0,0,.22); backdrop-filter: blur(6px); height: 100%; }
.sc-card h4 { margin: 0 0 .15rem 0; font-size: 1.08rem; color:#f8fafc; }
.sc-sub { color: var(--muted); font-size: .8rem; }
.sc-hero { background: linear-gradient(120deg, rgba(6,182,212,.18), rgba(99,102,241,.16)); border:1px solid rgba(34,211,238,.35); }
.sc-hero .big { font-family:'Space Grotesk'; font-size: 2.6rem; font-weight:700; line-height:1; }
.sc-kpi { text-align:left; }
.sc-kpi .lab { color: var(--muted); font-size:.75rem; text-transform: uppercase; letter-spacing:.06em; }
.sc-kpi .val { font-family:'Space Grotesk'; font-size:1.7rem; font-weight:700; color:#f8fafc; line-height:1.15; }
.sc-kpi .note { color: var(--muted); font-size:.74rem; }
.sc-chips { display:flex; flex-wrap:wrap; gap:.35rem; margin:.55rem 0; }
.sc-chip { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .55rem; border-radius:10px;
  font-size:.78rem; font-weight:600; border:1px solid rgba(255,255,255,.08); color:#0b1220; }
.sc-chip .n { font-family:'Space Grotesk'; font-size:.92rem; }
.sc-chip .t { font-weight:500; opacity:.8; }
/* spot cards: consistent spacing between rows */
.sc-spot { margin-bottom: 1.25rem; height: auto; position: relative; }
.sc-spot:hover, .sc-spot:focus-within { z-index: 60; }
/* species detail pop-up (hover / tap / keyboard focus) */
.sc-chips { position: relative; }
.sc-tip { position: static; display: inline-flex; outline: none; cursor: help; }
.sc-tip .sc-chip { transition: transform .12s ease, box-shadow .12s ease; }
.sc-tip:hover .sc-chip, .sc-tip:focus .sc-chip { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,.4); }
.sc-pop { display:none; position:absolute; top: calc(100% + 6px); left: -4px; right: -4px; z-index: 1000; min-width: 280px;
  background: #0b1528; border: 1px solid rgba(34,211,238,.35); border-radius: 14px; padding: .75rem .85rem;
  box-shadow: 0 18px 40px rgba(0,0,0,.6); color: #e2e8f0; font-size: .78rem; line-height: 1.4; font-weight: 400;
  text-align: left; white-space: normal; cursor: default; }
.sc-pop::before { content:""; position:absolute; top:-12px; left:0; right:0; height:12px; }
.sc-tip:hover .sc-pop, .sc-tip:focus .sc-pop, .sc-tip:focus-within .sc-pop { display:block; }
.sc-pop .pt { display:block; font-family:'Space Grotesk'; font-size:.92rem; font-weight:700; color:#f8fafc;
  border-left: 3px solid; padding-left: .45rem; margin-bottom: .3rem; }
.sc-pop .pb { display:inline-block; font-size:.66rem; font-weight:700; padding:.08rem .4rem; border-radius:5px; margin-bottom:.2rem; }
.sc-pop .pb.ok { background:#14532d; color:#bbf7d0; }
.sc-pop .pb.no { background:#7f1d1d; color:#fecaca; }
.sc-pop .ph { color:#67e8f9; font-size:.68rem; text-transform:uppercase; letter-spacing:.07em; font-weight:700; margin:.55rem 0 .2rem; }
.sc-pop .pw { padding:.12rem 0; }
.sc-pop .pg { display:flex; gap:.4rem; }
.sc-pop .pg span { color:#94a3b8; min-width:78px; }
.sc-pop ul { margin:.1rem 0 .2rem; padding-left:1rem; }
.sc-pop li { margin:.1rem 0; }
.sc-pop .pn { color:#94a3b8; font-size:.72rem; margin-top:.25rem; }
/* surf block on spot cards */
.sc-surf { margin-top:.55rem; padding:.5rem .6rem; border-radius:10px; background: rgba(2,6,23,.45);
  border:1px solid rgba(96,165,250,.28); font-size:.8rem; color:#cbd5e1; }
.sc-surf-top { display:flex; flex-wrap:wrap; align-items:baseline; gap:.35rem .5rem; }
.sc-surf-top .lab { color:#93c5fd; font-weight:600; }
.sc-surf-top .v { font-family:'Space Grotesk'; font-size:1.05rem; color:#f8fafc; }
.sc-surf-top .v.na { font-family:'Inter'; font-size:.85rem; color:#fca5a5; }
.sc-surf-top .rng { color:#94a3b8; font-size:.76rem; }
.sc-surf-grid { display:grid; grid-template-columns: 1fr; gap:.12rem; margin-top:.3rem; }
.sc-surf-grid > div { display:flex; gap:.45rem; }
.sc-surf-grid > div > span:first-child { color: var(--muted); min-width: 82px; flex: 0 0 82px; }
.sc-surf-grid i { color:#86efac; font-style:normal; font-size:.7rem; }
.sc-surf-note { color:#94a3b8; font-size:.74rem; margin-top:.15rem; }
.sc-surf-flag { color:#fbbf24; font-size:.72rem; margin-top:.2rem; }
.sc-conf, .sc-trend, .sc-safe { font-size:.7rem; font-weight:700; padding:.05rem .4rem; border-radius:6px; cursor:help;
  position:relative; outline:none; }
.sc-conf { color: var(--c); border:1px solid color-mix(in srgb, var(--c) 45%, transparent); }
.sc-trend { color:#e2e8f0; border:1px solid rgba(148,163,184,.3); }
.sc-safe.s-ok { color:#86efac; border:1px solid rgba(134,239,172,.35); }
.sc-safe.s-caution { color:#fbbf24; border:1px solid rgba(251,191,36,.45); }
.sc-safe.s-dangerous { color:#fca5a5; border:1px solid rgba(252,165,165,.5); background: rgba(127,29,29,.35); }
/* hover / focus tooltips for measurement labels */
.sc-h { border-bottom:1px dotted rgba(148,163,184,.7); cursor:help; position:relative; outline:none; }
.sc-h:hover::after, .sc-h:focus::after, .sc-conf:hover::after, .sc-conf:focus::after,
.sc-trend:hover::after, .sc-trend:focus::after, .sc-safe:hover::after, .sc-safe:focus::after {
  content: attr(data-tip); position:absolute; left:0; top: calc(100% + 6px); z-index: 1200; width: 270px;
  background:#0b1528; color:#e2e8f0; border:1px solid rgba(96,165,250,.45); border-radius:10px; padding:.5rem .6rem;
  font-size:.74rem; font-weight:400; line-height:1.4; white-space:normal; box-shadow: 0 14px 30px rgba(0,0,0,.55); }
.sc-cond { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:.3rem .8rem; font-size:.82rem; color:#cbd5e1; margin-top:.4rem; }
.sc-cond span { color: var(--muted); }
.sc-window { margin-top:.55rem; padding:.45rem .6rem; border-radius:10px; background: rgba(2,6,23,.5);
  border:1px dashed rgba(34,211,238,.35); font-size:.8rem; color:#e2e8f0; }
.sc-badge { display:inline-block; padding:.12rem .45rem; border-radius:6px; font-size:.7rem; font-weight:700; }
.sc-alert { padding:.6rem .9rem; border-radius:12px; border:1px solid rgba(245,158,11,.45);
  background: rgba(120,53,15,.35); color:#fde68a; font-size:.86rem; margin-bottom:.6rem; }
.sc-kind { font-size:.68rem; padding:.08rem .4rem; border-radius:5px; font-weight:700; text-transform:uppercase; }
.k-measured { background:#14532d; color:#bbf7d0; } .k-predicted { background:#1e3a8a; color:#bfdbfe; }
.k-model { background:#4c1d95; color:#ddd6fe; } .k-interpolated { background:#164e63; color:#a5f3fc; }
.k-computed { background:#334155; color:#e2e8f0; } .k-heuristic { background:#78350f; color:#fde68a; }
.k-missing { background:#7f1d1d; color:#fecaca; }
.sc-legend { display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; font-size:.78rem; color:#cbd5e1; }
.sc-legend i { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:.3rem; vertical-align:-1px; }
.sc-foot { color:#64748b; font-size:.75rem; margin-top:2rem; border-top:1px solid var(--line); padding-top:.8rem; }
.sc-foot a { color:#7dd3fc; }
.sc-explain { font-size:.92rem; line-height:1.5; color:#e2e8f0; }
.sc-target { display:flex; gap:.6rem; align-items:flex-start; padding:.45rem 0; border-bottom:1px solid var(--line); font-size:.84rem; }
.sc-target .num { flex:0 0 26px; height:26px; border-radius:50%; display:grid; place-items:center; font-weight:700;
  background:#22d3ee; color:#082f49; }
</style>
"""
