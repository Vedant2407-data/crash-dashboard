"""
Crash severity factors over time - dashboard
MSE907 Industry-based Capstone · Vedant Dave · 270779539

Shows the results of the analysis in plain language, in three pages:
    Overview          the short answer and how good the models are
    Factors (RQ1)     what the models rely on most, period by period
    Temporal (RQ2)    which changes are bigger than chance

The dashboard only READS files the pipeline already wrote. It trains nothing.

Run it from the project folder (the one that holds outputs/ and data/):
    pip install streamlit
    streamlit run app.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crash severity factors, 2006–2025",
                   page_icon=":material/traffic:", layout="wide")

PERIODS = ["2006-10", "2011-15", "2016-20", "2021-25"]
SEVERE = ["Serious Crash", "Fatal Crash"]
BREAK_AFTER = "2011-15"          # the 2015/16 boundary sits after this period


def nice_period(p: str) -> str:
    """2006-10 -> 2006–10 (en dash, for display only)."""
    return p.replace("-", "–")


# --------------------------------------------------------------------------
# Plain-English names for the 30 predictors (labels only - no results here)
# --------------------------------------------------------------------------
PREDICTORS = {
    "speedLimit": ("Speed limit", "Posted speed limit where the crash happened"),
    "motorcycle": ("Motorcycle involved", "A motorcycle was involved in the crash"),
    "pedestrian": ("Pedestrian involved", "A pedestrian was involved in the crash"),
    "struck_natural": ("Hit a natural object", "Hit a tree, bank, ditch, water or an animal"),
    "light": ("Light conditions", "Daylight, dark or twilight"),
    "bicycle": ("Bicycle involved", "A bicycle was involved in the crash"),
    "roadLane": ("Lane layout", "How the lanes are laid out (e.g. one-way, two-way)"),
    "region": ("Region", "Which region of Aotearoa New Zealand the crash was in"),
    "struck_roadside": ("Hit roadside furniture", "Hit a guard rail, post, sign or kerb"),
    "urban": ("Urban or rural", "Whether the crash was in an urban or rural area"),
    "carStationWagon": ("Car or station wagon involved", "Cars or station wagons were involved"),
    "weatherA": ("Weather", "Main weather at the time (fine, rain, fog, snow …)"),
    "truck": ("Truck involved", "A truck was involved in the crash"),
    "NumberOfLanes": ("Number of lanes", "How many traffic lanes the road has"),
    "flatHill": ("Flat or hilly road", "Whether the road was flat or on a hill"),
    "vehicle_other": ("Other vehicle type", "Taxi, moped or school bus (grouped together)"),
    "vanOrUtility": ("Van or ute involved", "A van or utility vehicle was involved"),
    "streetLight": ("Street lighting", "Whether street lights were there and on (recorded unevenly)"),
    "suv": ("SUV involved", "An SUV was involved in the crash"),
    "crashDirectionDescription": ("Direction of travel", "Compass direction of travel"),
    "struck_not_recorded": ("No object recorded", "No struck object was recorded (a recording marker)"),
    "struck_other": ("Hit another object", "Hit debris, a parked vehicle or a thrown object"),
    "struck_structure": ("Hit a structure", "Hit a bridge, building, roadworks or a train"),
    "roadCharacter": ("Road type", "Bridge, motorway, tunnel and similar"),
    "holiday_recorded": ("Holiday period", "The crash fell in a recorded holiday period"),
    "temporarySpeedLimit_recorded": ("Temporary speed limit", "A temporary speed limit was in place"),
    "advisorySpeed_recorded": ("Advisory speed", "An advisory speed sign applied"),
    "bus": ("Bus involved", "A bus was involved in the crash"),
    "weatherB_recorded": ("Second weather condition", "A second weather condition was recorded"),
    "roadSurface": ("Road surface", "Sealed or unsealed road"),
}


def label(code: str) -> str:
    return PREDICTORS.get(code, (code, ""))[0]


def describe(code: str) -> str:
    return PREDICTORS.get(code, ("", "No description"))[1]


# --------------------------------------------------------------------------
# Colours - colour-blind-safe palette (validated), light and dark steps
# --------------------------------------------------------------------------
LIGHT = dict(text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e6e5e0",
             card="#ffffff", tint="#f3f2ee", band="#dedcd5", context="#cfcdc6",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100", s5="#e87ba4",
             neutral="#b4b2aa", bg="#fcfcfb")
DARK = dict(text="#eef0f3", text2="#b4bac4", muted="#8a919c", grid="#2b303a",
            card="#1b1f27", tint="#20252e", band="#343a46", context="#3a414d",
            s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500", s5="#d55181",
            neutral="#5d6573", bg="#13161c")
FONT = '"Source Sans Pro", "Source Sans 3", "Source Sans", sans-serif'
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def theme_tokens() -> dict:
    """Colours follow the theme set in .streamlit/config.toml (dark by default)."""
    try:
        kind = st.get_option("theme.base")
    except Exception:
        kind = None
    if kind is None:                     # no config.toml: follow the viewer's theme
        try:
            kind = st.context.theme.type
        except Exception:
            kind = None
    return LIGHT if kind == "light" else DARK


T = theme_tokens()


def css():
    st.markdown(f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1280px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.kpi {{border: 1px solid {T['grid']}; border-radius: 12px; padding: 14px 16px 12px;
      background: {T['card']}; height: 100%;}}
.kpi .v {{font-size: 1.85rem; font-weight: 650; line-height: 1.15; color: {T['text']};}}
.kpi .l {{font-size: .86rem; color: {T['text2']}; margin-top: 4px; line-height: 1.35;}}
.answer {{border: 1px solid {T['grid']}; border-left: 5px solid {T['s1']};
         border-radius: 12px; padding: 16px 18px; background: {T['card']}; height: 100%;}}
.answer.two {{border-left-color: {T['s2']};}}
.answer .q {{font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;
            color: {T['text2']}; font-weight: 600;}}
.answer .a {{font-size: 1.3rem; font-weight: 650; margin: 6px 0 8px; color: {T['text']};
            line-height: 1.3;}}
.answer .d {{font-size: .95rem; color: {T['text2']}; line-height: 1.5;}}
.term {{border: 1px solid {T['grid']}; border-radius: 12px; padding: 14px 16px;
       background: {T['card']}; height: 100%;}}
.term .t {{font-weight: 650; color: {T['text']}; margin-bottom: 4px;}}
.term .x {{font-size: .92rem; color: {T['text2']}; line-height: 1.5;}}
.notmean {{border-radius: 12px; padding: 14px 18px; background: {T['tint']};
          border: 1px solid {T['grid']}; margin-top: 8px;}}
.notmean .h {{font-weight: 650; color: {T['text']}; margin-bottom: 6px;}}
.notmean ul {{margin: 0 0 0 1.1rem; padding: 0; color: {T['text2']};}}
.notmean li {{margin: 3px 0; line-height: 1.45;}}
.step {{display: inline-block; font-size: .75rem; font-weight: 700; letter-spacing: .06em;
       text-transform: uppercase; color: {T['s1']}; margin-bottom: -6px;}}
.badge {{display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: .82rem;
        font-weight: 600; border: 1px solid {T['grid']}; color: {T['text']};
        background: {T['tint']}; margin: 2px 4px 2px 0;}}
.badge.real {{border-color: {T['s2']};}}
.badge.real::before {{content: "▲ "; color: {T['s2']};}}
.badge.calm::before {{content: "● "; color: {T['muted']};}}
.badge.unsure::before {{content: "? "; color: {T['s4']}; font-weight: 800;}}
.fcard {{border: 1px solid {T['grid']}; border-radius: 12px; padding: 16px 18px;
        background: {T['card']};}}
.fcard .n {{font-size: 1.25rem; font-weight: 650; color: {T['text']};}}
.fcard .w {{color: {T['text2']}; font-size: .93rem; margin: 2px 0 10px;}}
.fcard table {{width: 100%; border-collapse: collapse; font-size: .92rem; margin: 6px 0 10px;}}
.fcard th {{text-align: left; color: {T['text2']}; font-weight: 500; padding: 4px 0;
           border-bottom: 1px solid {T['grid']};}}
.fcard td {{padding: 5px 0; color: {T['text']}; border-bottom: 1px solid {T['grid']};
           font-variant-numeric: tabular-nums;}}
.fcard .s {{font-size: .9rem; color: {T['text2']}; line-height: 1.5; margin-top: 6px;}}
.small {{font-size: .88rem; color: {T['text2']}; line-height: 1.5;}}
.mini-h {{font-weight: 650; color: {T['text']}; margin: 0 0 2px; font-size: 1rem;}}
button[data-baseweb="tab"] p {{font-size: 1.02rem; font-weight: 600;}}
button[data-baseweb="tab"] {{padding: 10px 14px;}}
.mini-s {{font-size: .86rem; color: {T['text2']}; margin: 0 0 4px; line-height: 1.4;}}
</style>""", unsafe_allow_html=True)


def kpi(value: str, text: str):
    st.markdown(f'<div class="kpi"><div class="v">{value}</div>'
                f'<div class="l">{text}</div></div>', unsafe_allow_html=True)


def not_mean(points: list[str]):
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(f'<div class="notmean"><div class="h">⚠️ What this does not mean</div>'
                f'<ul>{items}</ul></div>', unsafe_allow_html=True)


def step(n: int, title: str, sub: str | None = None):
    st.markdown(f'<span class="step">Step {n}</span>', unsafe_allow_html=True)
    st.subheader(title, anchor=False)
    if sub:
        st.markdown(f'<div class="small">{sub}</div>', unsafe_allow_html=True)


def show(chart):
    """Draw an Altair chart with this dashboard's styling, on any Streamlit version."""
    chart = (chart.configure(background="transparent", font=FONT)
             .configure_view(stroke=None)
             .configure_axis(labelColor=T["text2"], titleColor=T["text2"],
                             gridColor=T["grid"], domainColor=T["grid"],
                             tickColor=T["grid"], labelFontSize=12, titleFontSize=12,
                             titleFontWeight="normal", labelLimit=260)
             .configure_legend(labelColor=T["text2"], titleColor=T["text2"],
                               labelFontSize=12, titleFontSize=12, orient="top",
                               symbolStrokeWidth=0, titleFontWeight="normal")
             .configure_title(color=T["text"], fontSize=14, anchor="start",
                              fontWeight=600))
    try:
        st.altair_chart(chart, theme=None, width="stretch")
    except TypeError:
        st.altair_chart(chart, theme=None, use_container_width=True)


def table(df: pd.DataFrame, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def pick(lbl: str, options: list, default, key: str, fmt=str):
    """One-row option picker; segmented control where available, radio otherwise."""
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(lbl, options, default=default, key=key,
                                 format_func=fmt, selection_mode="single")
        return v if v is not None else default
    return st.radio(lbl, options, index=options.index(default), key=key,
                    format_func=fmt, horizontal=True)


# --------------------------------------------------------------------------
# Loading the pipeline's output files
# --------------------------------------------------------------------------
def project_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [os.environ.get("CRASH_DASH_DIR"), Path.cwd(), here]:
        if cand and (Path(cand) / "outputs" / "periods").is_dir():
            return Path(cand)
    return here


def _stamp(folder: Path) -> float:
    """Changes whenever any output file changes, so the cache refreshes itself."""
    per = folder / "outputs" / "periods"
    if not per.is_dir():
        return 0.0
    return max([f.stat().st_mtime for f in per.iterdir()] + [0.0])


@st.cache_data(show_spinner=False)
def load_model(folder: str, tag: str, _stamp: float) -> dict:
    per = Path(folder) / "outputs" / "periods"
    out, missing = {"tag": tag}, []

    # rankings -> rank and share of reliance per period
    ranks, shares = {}, {}
    for p in PERIODS:
        f = per / f"{tag}_{p}_ranking.csv"
        if not f.exists():
            missing.append(f.name); continue
        d = pd.read_csv(f).set_index("predictor")
        ranks[p] = d["rank"]
        shares[p] = d["mean_abs_shap"] / d["mean_abs_shap"].sum() * 100
    out["periods"] = [p for p in PERIODS if p in ranks]
    out["ranks"] = pd.DataFrame(ranks) if ranks else None
    out["shares"] = pd.DataFrame(shares) if shares else None

    # run_period.py overwrites metrics_all.csv on every run, so if periods were
    # run one at a time it holds only the last one. The per-period JSON files
    # are always complete, so read those first and fill gaps from the CSV.
    rows = {}
    f = per / f"{tag}_metrics_all.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            rows[str(r["period"])] = r.to_dict()
    for p in PERIODS:
        j = per / f"{tag}_{p}_metrics.json"
        if j.exists():
            rows[p] = json.load(open(j))
    if rows:
        out["metrics"] = (pd.DataFrame([dict(v, period=k) for k, v in rows.items()])
                          .set_index("period").reindex([p for p in PERIODS if p in rows]))
    else:
        out["metrics"] = None
        missing.append(f"{tag}_<period>_metrics.json")

    f = per / f"{tag}_spearman.csv"
    out["spearman"] = pd.read_csv(f, index_col=0) if f.exists() else None
    if out["spearman"] is None:
        missing.append(f.name)

    # noise baseline - run for the main model only
    for key, name in [("noise_ref", "noise_reference.json"),
                      ("rank_test", "noise_vs_observed.csv"),
                      ("weight_test", "magnitude.csv")]:
        f = per / f"{tag}_{name}"
        if not f.exists():
            out[key] = None
            if tag == "primary":
                missing.append(f.name)
        elif name.endswith(".json"):
            out[key] = json.load(open(f))
        else:
            out[key] = pd.read_csv(f, index_col=0)
    out["missing"] = missing
    return out


@st.cache_data(show_spinner=False)
def load_checks(folder: str, _stamp: float) -> dict:
    """Results of baseline_check.py and class_balance_check.py (both optional)."""
    per = Path(folder) / "outputs" / "periods"
    f1, f2 = per / "baseline_comparison.csv", per / "classbalance_summary.json"
    return {"baseline": pd.read_csv(f1).set_index("period") if f1.exists() else None,
            "balance": json.load(open(f2)) if f2.exists() else None}


@st.cache_data(show_spinner="Reading the crash data once …")
def load_crashes(folder: str) -> pd.DataFrame | None:
    """Counts by period and the share of exactly repeated records (optional file).

    Reads the full model data if it is there. A shared copy of the dashboard
    (made by make_share.py) has no crash data, only the small summary
    outputs/crash_summary.csv, so that is used instead."""
    f = Path(folder) / "data" / "model" / "cas_model.csv"
    summ = Path(folder) / "outputs" / "crash_summary.csv"
    if not f.exists():
        return pd.read_csv(summ).set_index("period") if summ.exists() else None
    d = pd.read_csv(f, low_memory=False)
    feats = [c for c in d.columns if c not in ("crashSeverity", "period", "crashYear")]
    rows = []
    for p in PERIODS:
        s = d[d.period == p]
        if s.empty:
            continue
        rows.append(dict(period=p, crashes=len(s),
                         severe=int(s.crashSeverity.isin(SEVERE).sum()),
                         fatal=int((s.crashSeverity == "Fatal Crash").sum()),
                         repeated_pct=float(s.duplicated(subset=feats).mean() * 100)
                         if feats else float("nan")))
    return pd.DataFrame(rows).set_index("period")


def verdicts(m: dict) -> pd.DataFrame:
    """Per predictor: did its rank / its weight change by more than chance?"""
    idx = m["ranks"].index
    v = pd.DataFrame(index=idx)
    rt, wt = m.get("rank_test"), m.get("weight_test")
    v["rank_real"] = (rt.reindex(idx)["verdict"].eq("BEYOND NOISE")
                      if rt is not None else False)
    v["weight_real"] = (wt.reindex(idx)["verdict"].eq("BEYOND NOISE")
                        if wt is not None else False)
    v["any_real"] = v.rank_real | v.weight_real
    return v


def crosses_break(a: str, b: str) -> bool:
    i = PERIODS.index(BREAK_AFTER)
    return (PERIODS.index(a) <= i) != (PERIODS.index(b) <= i)


def break_chart(nr: dict, height: int = 250,
                title: str = "Where the disagreement is, period pair by period pair"):
    """Share of between-period comparisons below the chance floor, per pair."""
    pairs = nr["share_below_floor_by_pair_pct"]
    d = []
    for key_, val in pairs.items():
        a_, b_ = key_.split(" vs ")
        d.append(dict(pair=f"{nice_period(a_)} vs {nice_period(b_)}", pct=val,
                      kind="Crosses 2015/16" if crosses_break(a_, b_) else
                      "Same side of 2015/16",
                      rho=nr.get("between_period_rho_by_pair", {}).get(key_)))
    d = pd.DataFrame(d).sort_values("pct")
    kinds = ["Same side of 2015/16", "Crosses 2015/16"]
    base = alt.Chart(d).encode(
        y=alt.Y("pair:N", sort=list(d.pair), title=None,
                axis=alt.Axis(labelColor=T["text"], labelFontSize=12.5)),
        x=alt.X("pct:Q", title="Comparisons below the chance floor (%)",
                scale=alt.Scale(domain=[0, 108], nice=False),
                axis=alt.Axis(values=[0, 25, 50, 75, 100])),
        tooltip=[alt.Tooltip("pair:N", title="Periods"),
                 alt.Tooltip("pct:Q", title="% below floor", format=".1f"),
                 alt.Tooltip("rho:Q", title="Mean agreement", format=".3f")])
    bars = base.mark_bar(size=20, cornerRadiusTopRight=4, cornerRadiusBottomRight=4) \
        .encode(color=alt.Color("kind:N", title=None, sort=kinds,
                                scale=alt.Scale(domain=kinds, range=[T["s1"], T["s2"]])))
    lab = base.mark_text(align="left", dx=5, color=T["text2"], fontSize=12) \
        .encode(text=alt.Text("pct:Q", format=".0f"))
    exp_ = nr["expected_share_if_no_effect_pct"]
    rule = alt.Chart(pd.DataFrame({"x": [exp_]})).mark_rule(
        color=T["text2"], strokeWidth=1.5).encode(x="x:Q")
    rule_t = alt.Chart(pd.DataFrame({"x": [exp_], "t": ["← expected by chance"]})) \
        .mark_text(align="left", dx=4, dy=-6, color=T["text2"], fontSize=11.5) \
        .encode(x="x:Q", y=alt.value(0), text="t:N")
    show((bars + lab + rule + rule_t).properties(
        height=height, title=title))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
css()
FOLDER = project_dir()
MODELS = {"primary": "XGBoost (main model)", "secondary": "Random forest (check model)"}

with st.sidebar:
    st.markdown("### Crash severity factors")
    st.caption("Aotearoa New Zealand · 2006–2025")

MAIN = load_model(str(FOLDER), "primary", _stamp(FOLDER))
SEC = load_model(str(FOLDER), "secondary", _stamp(FOLDER))
BY_TAG = {"primary": MAIN, "secondary": SEC}
SHORT = {"primary": "XGBoost", "secondary": "Random forest"}
CR = load_crashes(str(FOLDER))
CHK = load_checks(str(FOLDER), _stamp(FOLDER))

with st.sidebar:
    st.divider()
    st.markdown(
        '<div class="small"><b>Data:</b> Crash Analysis System (CAS), '
        'NZ Transport Agency Waka Kotahi, used under CC BY 4.0. '
        'Crashes reported to police, 2006–2025.<br><br>'
        '<b>Study:</b> Vedant Dave (270779539), MSE907 Industry-based Capstone, '
        'Yoobee College. Supervisor: Dr Helena Bahrami.</div>',
        unsafe_allow_html=True)
    with st.expander("Files in use"):
        st.caption(f"Reading from: `{FOLDER}`")
        if SEC["missing"] or MAIN["missing"]:
            st.caption("Missing: " + ", ".join(sorted(set(SEC["missing"] + MAIN["missing"]))))
        else:
            st.caption("All expected output files found.")
        if CR is None:
            st.caption("Optional `data/model/cas_model.csv` not found - fatal counts "
                       "and the recording check are hidden.")


def need(m: dict, *keys) -> bool:
    """Stop the page politely if a file it needs is missing."""
    gone = [k for k in keys if m.get(k) is None]
    if not gone:
        return True
    st.warning(
        "Some result files are missing, so this part can't be shown yet.\n\n"
        f"Looking in: `{FOLDER / 'outputs' / 'periods'}`\n\n"
        f"Missing: {', '.join(m['missing']) or ', '.join(gone)}\n\n"
        "Run the pipeline first (README, steps 9–15), or start the dashboard from "
        "the project folder.")
    return False


# ==========================================================================
# PAGE 1 - OVERVIEW
# ==========================================================================
def page_overview():
    st.title("What makes a crash severe – and has it changed?", anchor=False)
    st.markdown(
        '<div class="small" style="font-size:1.02rem;max-width:860px">'
        "Two machine-learning models learned to tell <b>severe</b> crashes (someone "
        "killed or seriously injured) apart from the rest, using 30 facts recorded "
        "about each crash. We then asked each model <i>which facts it relied on</i>, "
        "separately for four five-year periods, and checked whether that changed by "
        "more than chance would explain.</div>", unsafe_allow_html=True)
    st.write("")
    if not need(MAIN, "ranks", "metrics"):
        return

    met = MAIN["metrics"]
    if CR is not None and len(CR):
        crashes, severe = CR.crashes.astype(float), CR.severe.astype(float)
    else:
        crashes = (met.n_train + met.n_test).astype(float)
        severe = (met.severe_train + met.severe_test).astype(float)
    cps = [p for p in PERIODS if p in crashes.index]
    crashes, severe = crashes.reindex(cps), severe.reindex(cps)
    c = st.columns(4)
    with c[0]:
        kpi(f"{int(crashes.sum()):,}", "crashes analysed, 2006–2025")
    with c[1]:
        kpi(f"{int(severe.sum()):,}",
            f"severe crashes ({severe.sum() / crashes.sum() * 100:.1f}% of all)")
    with c[2]:
        if CR is not None:
            kpi(f"{int(CR.fatal.sum()):,}", "of those were fatal")
        else:
            kpi(f"{len(MAIN['periods'])}", "five-year periods compared")
    with c[3]:
        kpi(f"{len(MAIN['ranks'])}", "facts (factors) the models could use")

    st.write("")
    # ---- the two answers -------------------------------------------------
    r = MAIN["ranks"]
    always = r[r.max(axis=1) <= 3].index.tolist()
    always = sorted(always, key=lambda q: r.loc[q].mean())
    names = ", ".join(label(q) for q in always[:-1]) + (" and " if len(always) > 1 else "") \
        + (label(always[-1]) if always else "")
    sec = load_model(str(FOLDER), "secondary", _stamp(FOLDER))
    check_line = ""
    if sec["ranks"] is not None and always:
        worst = int(sec["ranks"].reindex(always).max(axis=1).max())
        check_line = (f" The check model also puts them in its top {max(worst, 3)} "
                      f"every time." if worst <= 5 else
                      " The check model ranks them lower in some periods.")
    a, b = st.columns(2)
    with a:
        st.markdown(
            f'<div class="answer"><div class="q">Question 1 · Which factors matter most?</div>'
            f'<div class="a">{names or "No factor stayed in the top 3"}</div>'
            f'<div class="d">These {len(always)} were in the main model\'s top 3 in '
            f'<b>every</b> period.{check_line}</div></div>', unsafe_allow_html=True)
    with b:
        nr = MAIN.get("noise_ref")
        if nr:
            v = verdicts(MAIN)
            moved = v[v.any_real].index.tolist()
            moved = sorted(moved, key=lambda q: r.loc[q].mean())
            st.markdown(
                f'<div class="answer two"><div class="q">Question 2 · Do they stay the same?</div>'
                f'<div class="a">{"Not entirely – the order shifted around 2015/16" if nr["overall_unstable"] else "Yes – changes are within chance"}</div>'
                f'<div class="d">{nr["share_between_below_floor_pct"]:.1f}% of comparisons '
                f'between periods agreed less than chance allows (expected: '
                f'{nr["expected_share_if_no_effect_pct"]:.1f}%). '
                f'{len(moved)} factors really changed: '
                f'{", ".join(label(q) for q in moved)}.</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="answer two"><div class="q">Question 2</div>'
                        '<div class="a">Noise test results not found</div></div>',
                        unsafe_allow_html=True)

    nr0 = MAIN.get("noise_ref")
    if nr0:
        st.write("")
        st.subheader("The main finding in one chart", anchor=False)
        st.markdown(
            '<div class="small">Each bar compares two periods. It shows how often their '
            'rankings disagreed by more than chance allows. If nothing had changed, '
            f'every bar would sit near the grey line ({nr0["expected_share_if_no_effect_pct"]:.1f}%).'
            '</div>', unsafe_allow_html=True)
        break_chart(nr0, height=230, title="")
        st.markdown(
            '<div class="small">Periods on the <b>same side</b> of 2015/16 (blue) mostly '
            'agree. Every pair that <b>crosses</b> 2015/16 (orange) disagrees far more '
            'often. The change is a break around 2015/16, not a slow drift. The '
            '<b>Temporal</b> page shows which factors changed.</div>',
            unsafe_allow_html=True)

    st.write("")
    st.divider()
    # ---- severe rate + model quality ---------------------------------------
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("How often are crashes severe?", anchor=False)
        d = pd.DataFrame({"period": [nice_period(p) for p in cps],
                          "rate": (severe / crashes * 100).values,
                          "crashes": crashes.values, "severe": severe.values})
        d["txt"] = d.rate.map(lambda x: f"{x:.1f}%")
        base = alt.Chart(d).encode(
            x=alt.X("period:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("rate:Q", title="Severe crashes (% of all crashes)",
                    scale=alt.Scale(domain=[0, max(9, d.rate.max() + 1)])),
            tooltip=[alt.Tooltip("period:N", title="Period"),
                     alt.Tooltip("crashes:Q", title="Crashes", format=","),
                     alt.Tooltip("severe:Q", title="Severe", format=","),
                     alt.Tooltip("rate:Q", title="Severe %", format=".2f")])
        ch = (base.mark_bar(color=T["s1"], cornerRadiusTopLeft=4,
                            cornerRadiusTopRight=4, size=38)
              + base.mark_text(dy=-9, color=T["text"], fontSize=13, fontWeight=600)
              .encode(text="txt:N"))
        show(ch.properties(height=290))
        st.markdown(
            f'<div class="small">The share of severe crashes rose from '
            f'{d.rate.iloc[0]:.1f}% to {d.rate.iloc[-1]:.1f}%. Part of this may be '
            f'because fewer minor crashes are reported to police now – not '
            f'necessarily because crashes got worse.</div>', unsafe_allow_html=True)

    with right:
        st.subheader("How good are the models?", anchor=False)
        rows = []
        for t, mm in [("primary", MAIN), ("secondary", load_model(str(FOLDER), "secondary", _stamp(FOLDER)))]:
            if mm["metrics"] is None:
                continue
            for p, x in mm["metrics"].iterrows():
                rows.append(dict(period=nice_period(p), series=MODELS[t].split(" (")[0],
                                 score=x.pr_auc))
        bl = CHK["baseline"]
        if bl is not None:
            for p, x in bl.iterrows():
                rows.append(dict(period=nice_period(p), series="Logistic regression",
                                 score=x.lr_pr_auc))
        for p in cps:
            rows.append(dict(period=nice_period(p), series="Random guessing",
                             score=float(severe[p] / crashes[p])))
        d = pd.DataFrame(rows)
        order = ["XGBoost", "Random forest", "Logistic regression", "Random guessing"]
        if bl is None:
            order.remove("Logistic regression")
        colours = {"XGBoost": T["s1"], "Random forest": T["s2"],
                   "Logistic regression": T["s3"], "Random guessing": T["neutral"]}
        ch = alt.Chart(d).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                   size=18 if bl is not None else 22).encode(
            x=alt.X("period:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("series:N", sort=order),
            y=alt.Y("score:Q", title="Score (PR-AUC, higher is better)",
                    scale=alt.Scale(domain=[0, 0.4])),
            color=alt.Color("series:N", title=None, sort=order,
                            scale=alt.Scale(domain=order,
                                            range=[colours[k] for k in order])),
            tooltip=[alt.Tooltip("period:N", title="Period"),
                     alt.Tooltip("series:N", title="Model"),
                     alt.Tooltip("score:Q", title="PR-AUC", format=".3f")])
        show(ch.properties(height=290))
        x_mean = d[d.series == "XGBoost"].score.mean()
        lr_line = ""
        if bl is not None and bl.xgb_pr_auc.notna().all():
            wins = int((bl.xgb_pr_auc > bl.lr_pr_auc).sum())
            lr_line = (f" A simple logistic regression (green) is the baseline: XGBoost "
                       f"beats it in {wins} of {len(bl)} periods, by "
                       f"{bl.xgb_vs_lr_pr_auc_pct.mean():.0f}% on average.")
        g_mean = d[d.series == "Random guessing"].score.mean()
        st.markdown(
            f'<div class="small">PR-AUC measures how well a model picks out the rare '
            f'severe crashes (1.0 = perfect). Random guessing scores about '
            f'{g_mean:.2f}; the main model scores about {x_mean:.2f} – roughly '
            f'{x_mean / g_mean:.0f}× better. Both models score about the same in every '
            f'period, so comparing what they rely on is fair.{lr_line}</div>',
            unsafe_allow_html=True)

    st.divider()
    st.subheader("Three ideas you need to read this dashboard", anchor=False)
    t = st.columns(3)
    terms = [
        ("Rank", "Puts the 30 factors in order of how much the model leans on them. "
                 "Rank 1 = the factor it relies on most. Rank tells you <i>which</i> "
                 "factor leads, not by how much."),
        ("Share of reliance", "How much of the model's total reliance one factor "
                              "carries, as a percentage. If speed limit has 13%, it "
                              "carries 13% of what the model uses to make its call. "
                              "This tells you <i>by how much</i>."),
        ("Chance allowance", "Even with nothing changing, two random halves of the "
                             "same period give slightly different results. We measured "
                             "how big that wobble is. A change only counts as "
                             "<b>real</b> if it is bigger than this allowance."),
    ]
    for col, (h, x) in zip(t, terms):
        with col:
            st.markdown(f'<div class="term"><div class="t">{h}</div>'
                        f'<div class="x">{x}</div></div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("The data and the method, in brief"):
        rows = []
        for p in cps:
            row = {"Period": nice_period(p), "Crashes": int(crashes[p]),
                   "Severe": int(severe[p]),
                   "Severe %": round(float(severe[p] / crashes[p] * 100), 2)}
            if CR is not None and p in CR.index:
                row["Fatal"] = int(CR.loc[p, "fatal"])
            rows.append(row)
        table(pd.DataFrame(rows))
        st.markdown(
            "- **Data:** every crash in the Crash Analysis System from 2006 to 2025, "
            "split into four five-year periods. 2026 is left out because it is incomplete.\n"
            "- **Target:** severe (fatal or serious injury) versus not severe.\n"
            "- **Models:** XGBoost (main) and random forest (check). The same settings "
            "were used in every period, so only the data changes.\n"
            "- **Explanation:** SHAP values show how much each factor pushed each "
            "prediction. Averaging them gives each factor's rank and share.\n"
            "- **Chance allowance:** each period was split into two halves with no "
            "shared crashes, 30 times over (240 models). Halves of the same period "
            "differ only by chance; that sets the bar a real change must clear.")

    not_mean([
        "A factor the model relies on is not proven to <b>cause</b> severe crashes. "
        "It means the factor helps the model tell severe crashes apart.",
        "It is not a ranking of how dangerous regions, roads or drivers are.",
        "The models are not meant for predicting individual crashes; they are tools "
        "for seeing what the data says, period by period.",
    ])


# ==========================================================================
# PAGE 2 - WHICH FACTORS MATTER (RQ1)
# ==========================================================================
def page_factors():
    st.title("Which factors matter most?", anchor=False)
    st.markdown('<div class="small" style="font-size:1rem">Research question 1 · '
                'What each model leans on most to tell severe crashes apart. Switch '
                'model to see where the two agree and where they differ.</div>',
                unsafe_allow_html=True)
    st.write("")
    c0, c1, c2 = st.columns([1.25, 1.6, 1])
    with c0:
        tag = pick("Model", ["primary", "secondary"], "primary", "f_model",
                   lambda k: {"primary": "XGBoost (main)", "secondary": "Random forest (check)"}[k])
    M = BY_TAG[tag]
    other = "secondary" if tag == "primary" else "primary"
    O = BY_TAG[other]
    if not need(M, "ranks", "shares"):
        return
    r, s, ps = M["ranks"], M["shares"], M["periods"]
    always = set(r[r.max(axis=1) <= 3].index)
    with c1:
        period = pick("Period", ps, ps[-1], "f_period", nice_period)
    with c2:
        n = pick("How many factors", [10, 15, 30], 15, "f_n",
                 lambda k: "All 30" if k == 30 else f"Top {k}")

    met = M["metrics"]
    if met is not None and period in met.index:
        x = met.loc[period]
        n_all = int(x.n_train + x.n_test)
        guess = float((x.severe_train + x.severe_test) / n_all)
        k = st.columns(3)
        with k[0]:
            kpi(f"{n_all:,}", f"crashes in {nice_period(period)} "
                f"({guess * 100:.1f}% severe)")
        with k[1]:
            kpi(f"{x.pr_auc:.3f}", f"PR-AUC on held-out crashes – how well the model "
                f"finds severe ones (random guessing ≈ {guess:.2f})")
        with k[2]:
            base_f1 = (2 * (1 - guess) / (2 - guess)) / 2
            kpi(f"{x.macro_f1:.3f}", f"macro-F1 on held-out crashes – severe and "
                f"non-severe count equally (always saying “not severe” ≈ {base_f1:.2f})")
        st.write("")

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        d = pd.DataFrame({"code": s.index, "share": s[period].values,
                          "rank": r[period].reindex(s.index).values})
        d = d.sort_values("rank").head(n)
        d["name"] = d.code.map(label)
        d["what"] = d.code.map(describe)
        d["group"] = np.where(d.code.isin(always), "In the top 3 in every period",
                              "Other factors")
        d["txt"] = d.share.map(lambda x: f"{x:.1f}%")
        groups = ["In the top 3 in every period", "Other factors"]
        base = alt.Chart(d).encode(
            y=alt.Y("name:N", sort=list(d.name), title=None,
                    axis=alt.Axis(labelFontSize=12.5, labelColor=T["text"])),
            x=alt.X("share:Q", title="Share of the model's reliance (%)",
                    scale=alt.Scale(domain=[0, max(16, d.share.max() * 1.15)])),
            tooltip=[alt.Tooltip("name:N", title="Factor"),
                     alt.Tooltip("what:N", title="What it is"),
                     alt.Tooltip("rank:Q", title="Rank"),
                     alt.Tooltip("share:Q", title="Share %", format=".1f")])
        bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                             size=14 if n > 15 else 18).encode(
            color=alt.Color("group:N", title=None, sort=groups,
                            scale=alt.Scale(domain=groups,
                                            range=[T["s1"], T["neutral"]])))
        txt = base.mark_text(align="left", dx=5, color=T["text2"], fontSize=12) \
            .encode(text="txt:N")
        show((bars + txt).properties(
            height=max(300, 26 * len(d)),
            title=f"What {SHORT[tag]} relied on most, {nice_period(period)}"))
        top = d.iloc[0]
        st.markdown(
            f'<div class="small">In {nice_period(period)}, <b>{top["name"]}</b> '
            f'carried the most weight ({top.share:.1f}% of the total). The top 3 '
            f'together carried {d.head(3).share.sum():.0f}%. Hover a bar for '
            f'details.</div>', unsafe_allow_html=True)
        # where the two models disagree, for this period
        if O["ranks"] is not None and period in O["ranks"].columns:
            ro = O["ranks"][period].reindex(r.index)
            gap = (r[period] - ro).abs().sort_values(ascending=False).head(3)
            same3 = set(r[period].nsmallest(3).index) == set(ro.nsmallest(3).index)
            items = "".join(
                f"<li><b>{label(q)}</b>: #{int(r.loc[q, period])} in {SHORT[tag]}, "
                f"#{int(ro[q])} in {SHORT[other]}</li>" for q in gap.index)
            st.markdown(
                f'<div class="notmean" style="margin-top:14px"><div class="h">'
                f'{SHORT[tag]} vs {SHORT[other]}, {nice_period(period)}</div>'
                f'<div class="small" style="margin-bottom:4px">'
                f'{"Both models have the same top 3 in this period. " if same3 else "The two models have a different top 3 in this period. "}'
                f'Biggest disagreements:</div><ul>{items}</ul></div>',
                unsafe_allow_html=True)

    with right:
        options = list(r.loc[r[period].sort_values().index].index)
        code = st.selectbox("Look up one factor", options, index=0,
                            format_func=label, key="f_factor")
        ro_all = O["ranks"] if O["ranks"] is not None else None

        def orank(p):
            if ro_all is None or p not in ro_all.columns or code not in ro_all.index:
                return "–"
            return f"#{int(ro_all.loc[code, p])}"
        rows = "".join(
            f"<tr><td>{nice_period(p)}</td><td>#{int(r.loc[code, p])}</td>"
            f"<td>{s.loc[code, p]:.1f}%</td><td style='color:{T['text2']}'>{orank(p)}</td></tr>"
            for p in ps)
        badges = ""
        note = ""
        if tag == "primary" and M.get("rank_test") is not None:
            v = verdicts(M).loc[code]
            badges = (
                f'<span class="badge {"real" if v.rank_real else "calm"}">Rank: '
                f'{"real change" if v.rank_real else "steady (within chance)"}</span>'
                f'<span class="badge {"real" if v.weight_real else "calm"}">Share: '
                f'{"real change" if v.weight_real else "steady (within chance)"}</span>')
            ch = (s.loc[code, ps[-1]] / s.loc[code, ps[0]] - 1) * 100
            if not v.weight_real and abs(ch) >= 20:
                badges += '<span class="badge unsure">Share change unconfirmed</span>'
                note = (f"Its share moved {ch:+.0f}% from first to last period, but "
                        f"the chance test does not confirm it, so treat that as "
                        f"possible noise.")
        elif tag == "secondary":
            note = "The chance test was run on the main model only, so no verdict here."
        place = ("It was in the top 3 in every period." if code in always else
                 f"Its best rank was #{int(r.loc[code].min())}, its lowest "
                 f"#{int(r.loc[code].max())}.")
        st.markdown(
            f'<div class="fcard"><div class="n">{label(code)}</div>'
            f'<div class="w">{describe(code)}</div>{badges}'
            f'<table><tr><th>Period</th><th>Rank</th><th>Share</th><th>Rank in {SHORT[other]}</th></tr>{rows}</table>'
            f'<div class="s">{place} {note}</div></div>', unsafe_allow_html=True)
        st.caption("To see how this factor moved over time, open the Temporal page.")

    st.write("")
    with st.expander("See every factor in a table"):
        tb = pd.DataFrame({"Factor": [label(q) for q in r.index],
                           "What it is": [describe(q) for q in r.index]})
        for p in ps:
            tb[f"Rank {nice_period(p)}"] = r[p].astype(int).values
        for p in ps:
            tb[f"Share {nice_period(p)} (%)"] = s[p].reindex(r.index).round(1).values
        tb = tb.sort_values(f"Rank {nice_period(ps[-1])}")
        table(tb)
        st.download_button("Download this table (CSV)", tb.to_csv(index=False),
                           file_name=f"{tag}_factors_by_period.csv", mime="text/csv")

    not_mean([
        "A high rank means the model <b>uses</b> the factor to tell severe crashes "
        "apart, not that the factor causes severity. For example, <i>Pedestrian "
        "involved</i> ranks high because crashes with pedestrians are more often "
        "severe – the model is spotting that link.",
        "Factors that travel together can stand in for each other. If two factors "
        "carry similar information, the model may lean on one and rank the other low.",
        "The share is about this model on this data. A different model can spread its "
        "reliance differently – switch the model at the top of this page to compare.",
    ])


# ==========================================================================
# PAGE 3 - HAS IT CHANGED? (RQ2)
# ==========================================================================
def page_temporal():
    st.title("Have the important factors changed over time?", anchor=False)
    st.markdown('<div class="small" style="font-size:1rem">Research question 2 · '
                'Every result here is checked against the chance allowance, so only '
                'changes bigger than random wobble are called real.</div>',
                unsafe_allow_html=True)
    st.caption("Five steps, one per tab. Steps 1–3 use XGBoost (the main model), "
               "because the chance test was run on it. Step 4 lets you pick either "
               "model. Step 5 checks the results hold up.")
    st.write("")
    if not need(MAIN, "ranks", "shares", "noise_ref", "rank_test", "weight_test"):
        return
    r, s, ps = MAIN["ranks"], MAIN["shares"], MAIN["periods"]
    nr, v = MAIN["noise_ref"], verdicts(MAIN)

    moved = sorted(v[v.any_real].index, key=lambda q: -abs(
        math.log(s.loc[q, ps[-1]] / s.loc[q, ps[0]])))
    tabs = st.tabs(['1 · Is it changing?', '2 · What changed?', '3 · Follow a factor', '4 · When?', '5 · Does it hold up?'])
    with tabs[0]:
        # ---- STEP 1 --------------------------------------------------------
        step(1, "Is the order of factors changing more than chance would?",
             "We cut each period into two halves that share no crashes, 30 times over. "
             "Two halves of the <b>same</b> period show how much results wobble by chance. "
             "Then we compared halves of <b>different</b> periods. If time made no "
             "difference, only 2.5% of those comparisons should fall below the chance "
             "floor.")
        st.write("")
        k = st.columns(4)
        with k[0]:
            kpi(f'{nr["within_period_rho_mean"]:.3f}', "agreement between two halves of "
                "the same period (1 = identical order)")
        with k[1]:
            kpi(f'{nr["within_period_rho_p2_5"]:.3f}', "chance floor – agreement below "
                "this is more change than chance explains")
        with k[2]:
            kpi(f'{nr["between_period_rho_mean"]:.3f}', "agreement between different "
                "periods")
        with k[3]:
            kpi(f'{nr["share_between_below_floor_pct"]:.1f}%',
                f'of comparisons fell below the floor – {nr["share_ratio"]:.1f}× the '
                f'{nr["expected_share_if_no_effect_pct"]:.1f}% expected by chance')

        st.write("")
        break_chart(nr)
        st.markdown(
            '<div class="small">Pairs on the same side of 2015/16 (blue) mostly agree. '
            'Every pair that crosses it (orange) disagrees far more often. That looks '
            'like a <b>break around 2015/16</b>, not a slow, steady drift.</div>',
            unsafe_allow_html=True)
    with tabs[1]:
        # ---- STEP 2 --------------------------------------------------------
        step(2, f"Which factors really changed? ({len(moved)} of {len(r)})",
             "Each small chart shows one factor's share of the model's reliance in each "
             "period. These are the factors whose change beat the chance allowance.")
        st.write("")
        ymax = float(np.ceil(s.loc[moved, ps].values.max() + 1.5)) if moved else 15
        for row_start in range(0, len(moved), 3):
            cols = st.columns(3, gap="medium")
            for col, q in zip(cols, moved[row_start:row_start + 3]):
                with col:
                    a0, a1 = s.loc[q, ps[0]], s.loc[q, ps[-1]]
                    r0, r1 = int(r.loc[q, ps[0]]), int(r.loc[q, ps[-1]])
                    what = ("rank and share" if v.loc[q, "rank_real"] and v.loc[q, "weight_real"]
                            else "share only – its rank barely moved"
                            if v.loc[q, "weight_real"] else "rank only")
                    arrow = "rose" if a1 > a0 else "fell"
                    st.markdown(
                        f'<div class="mini-h">{label(q)} {"↑" if a1 > a0 else "↓"}</div>'
                        f'<div class="mini-s">Share {arrow} {a0:.1f}% → {a1:.1f}% · rank '
                        f'#{r0} → #{r1}<br>Real change in: {what}</div>',
                        unsafe_allow_html=True)
                    dd = pd.DataFrame({"period": [nice_period(p) for p in ps],
                                       "share": s.loc[q, ps].values,
                                       "rank": r.loc[q, ps].astype(int).values})
                    dd["txt"] = dd.share.map(lambda x: f"{x:.1f}")
                    colr = T["s2"] if a1 > a0 else T["s1"]
                    b2 = alt.Chart(dd).encode(
                        x=alt.X("period:N", title=None, axis=alt.Axis(labelAngle=0,
                                                                      labelFontSize=11)),
                        y=alt.Y("share:Q", title="Share (%)", scale=alt.Scale(domain=[0, ymax])),
                        tooltip=[alt.Tooltip("period:N", title="Period"),
                                 alt.Tooltip("share:Q", title="Share %", format=".1f"),
                                 alt.Tooltip("rank:Q", title="Rank")])
                    ch = (b2.mark_line(color=colr, strokeWidth=2.5)
                          + b2.mark_point(color=colr, filled=True, size=70)
                          + b2.mark_text(dy=-11, color=T["text2"], fontSize=11)
                          .encode(text="txt:N"))
                    show(ch.properties(height=170))
        st.markdown(
            '<div class="small">↑ orange = the model relies on it more over time; '
            '↓ blue = less. Motorcycle is the case a rank-only study would miss: it '
            'stayed near the top, yet its share grew beyond chance.</div>'
            if "motorcycle" in moved and not v.loc["motorcycle", "rank_real"] else
            '<div class="small">↑ orange = the model relies on it more over time; '
            '↓ blue = less.</div>', unsafe_allow_html=True)

        st.write("")
        # ---- change vs allowance, all factors ------------------------------
        st.markdown("##### Every factor: how much it moved, against what chance allows")
        measure = pick("Measure", ["rank", "share"], "rank", "t_measure",
                       lambda k: "Rank (places moved)" if k == "rank"
                       else "Share (% change)")
        if measure == "rank":
            t_ = MAIN["rank_test"].reindex(r.index)
            dd = pd.DataFrame({"code": r.index, "obs": t_.observed_median_shift.values,
                               "allow": t_.noise_p95_shift.values,
                               "real": v.rank_real.values})
            unit, xt = "places", "Typical movement between periods (rank places)"
        else:
            t_ = MAIN["weight_test"].reindex(r.index)
            dd = pd.DataFrame({"code": r.index,
                               "obs": (np.exp(t_.observed_median.values) - 1) * 100,
                               "allow": (np.exp(t_.noise_p95.values) - 1) * 100,
                               "real": v.weight_real.values})
            unit, xt = "%", "Typical change in share between periods (%)"
        dd["name"] = dd.code.map(label)
        dd["status"] = np.where(dd.real, "Real change (beyond chance)", "Within chance")
        dd["ratio"] = dd.obs / dd.allow.replace(0, np.nan)
        dd["tag"] = np.where(dd.real, "real", "")
        dd = dd.sort_values("ratio", ascending=False)
        n_all = len(dd)
        if not st.toggle(f"Show all {n_all} factors", value=False, key="t_all"):
            dd = dd.head(max(10, int(dd.real.sum())))
        sts = ["Real change (beyond chance)", "Within chance"]
        base = alt.Chart(dd).encode(
            y=alt.Y("name:N", title=None, sort=list(dd.name),
                    axis=alt.Axis(labelColor=T["text"], labelFontSize=12)),
            tooltip=[alt.Tooltip("name:N", title="Factor"),
                     alt.Tooltip("obs:Q", title=f"Moved ({unit})", format=".1f"),
                     alt.Tooltip("allow:Q", title=f"Chance allows ({unit})", format=".1f"),
                     alt.Tooltip("status:N", title="Verdict")])
        band = base.mark_bar(size=11, color=T["band"], cornerRadius=3).encode(
            x=alt.X("allow:Q", title=xt), x2=alt.datum(0))
        dots = base.mark_point(filled=True, size=90, opacity=1,
                               stroke=T["bg"], strokeWidth=1.5).encode(
            x="obs:Q",
            color=alt.Color("status:N", title=None, sort=sts,
                            scale=alt.Scale(domain=sts, range=[T["s2"], T["muted"]])))
        lab = base.mark_text(align="left", dx=9, color=T["text"], fontSize=11.5,
                             fontWeight=600).encode(x="obs:Q", text="tag:N")
        show((band + dots + lab).properties(height=22 * len(dd) + 40))
        st.markdown(
            f'<div class="small"><b>Grey bar</b> = the most chance alone moves that factor '
            f'(in 95% of tries). <b>Dot</b> = how much it actually moved between periods. '
            f'A dot past the end of its bar is a real change. Factors are sorted with the '
            f'clearest changes at the top{"" if len(dd) == n_all else f"; showing the top {len(dd)} of {n_all}"}. '
            f'A big raw move is not enough on its own: a factor that normally wobbles a lot '
            f'needs to move even more.</div>',
            unsafe_allow_html=True)
    with tabs[2]:
        # ---- STEP 3 - explore one factor -----------------------------------
        step(3, "Follow any factor through time")
        q = st.selectbox("Factor", list(r.index), index=list(r.index).index(moved[0])
                         if moved else 0, format_func=label, key="t_factor")
        a, b = st.columns(2, gap="large")
        with a:
            long = r.reset_index(names="code").melt("code", var_name="period", value_name="rank")
            long["period"] = long.period.map(nice_period)
            long["name"] = long.code.map(label)
            ctx = alt.Chart(long[long.code != q]).mark_line(
                color=T["context"], strokeWidth=1, opacity=.7).encode(
                x=alt.X("period:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("rank:Q", title="Rank (1 = relied on most)",
                        scale=alt.Scale(domain=[1, len(r)], reverse=True, nice=False),
                        axis=alt.Axis(values=[1, 5, 10, 15, 20, 25, 30])),
                detail="code:N",
                tooltip=[alt.Tooltip("name:N", title="Factor"), alt.Tooltip("rank:Q")])
            fo = long[long.code == q].assign(txt=lambda x: "#" + x["rank"].astype(int).astype(str))
            fb = alt.Chart(fo).encode(x="period:N", y="rank:Q",
                                      tooltip=[alt.Tooltip("period:N"), alt.Tooltip("rank:Q")])
            focus = (fb.mark_line(color=T["s1"], strokeWidth=3)
                     + fb.mark_point(color=T["s1"], filled=True, size=90)
                     + fb.mark_text(dy=-13, color=T["text"], fontSize=12.5, fontWeight=600)
                     .encode(text="txt:N"))
            show((ctx + focus).properties(height=320, title=f"{label(q)} – rank"))
        with b:
            fs = pd.DataFrame({"period": [nice_period(p) for p in ps],
                               "share": s.loc[q, ps].values})
            fs["txt"] = fs.share.map(lambda x: f"{x:.1f}%")
            sb = alt.Chart(fs).encode(
                x=alt.X("period:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("share:Q", title="Share of the model's reliance (%)",
                        scale=alt.Scale(domain=[0, max(14, fs.share.max() * 1.25)])),
                tooltip=[alt.Tooltip("period:N"), alt.Tooltip("share:Q", format=".1f")])
            ch = (sb.mark_line(color=T["s1"], strokeWidth=3)
                  + sb.mark_point(color=T["s1"], filled=True, size=90)
                  + sb.mark_text(dy=-13, color=T["text"], fontSize=12.5, fontWeight=600)
                  .encode(text="txt:N"))
            show(ch.properties(height=320, title=f"{label(q)} – share"))

        rt_, wt_ = MAIN["rank_test"].loc[q], MAIN["weight_test"].loc[q]
        w_obs = (math.exp(wt_.observed_median) - 1) * 100
        w_all = (math.exp(wt_.noise_p95) - 1) * 100
        ch_pct = (s.loc[q, ps[-1]] / s.loc[q, ps[0]] - 1) * 100
        s1 = (f"<b>Rank:</b> between periods it typically moved "
              f"{rt_.observed_median_shift:.0f} places; chance alone moves it up to "
              f"{rt_.noise_p95_shift:.0f}. → " +
              ("<b>Real change.</b>" if v.loc[q, "rank_real"] else "Within chance – not counted."))
        s2 = (f"<b>Share:</b> {s.loc[q, ps[0]]:.1f}% → {s.loc[q, ps[-1]]:.1f}% "
              f"({ch_pct:+.0f}% first to last). Typical change between periods "
              f"{w_obs:.0f}%; chance allows up to {w_all:.0f}%. → " +
              ("<b>Real change.</b>" if v.loc[q, "weight_real"] else
               "Not confirmed – could be noise." if abs(ch_pct) >= 20 else
               "Within chance – not counted."))
        st.markdown(f'<div class="fcard"><div class="n">{label(q)}</div>'
                    f'<div class="w">{describe(q)}</div>'
                    f'<div class="s">{s1}<br>{s2}</div></div>', unsafe_allow_html=True)
    with tabs[3]:
        # ---- STEP 4 - heatmap ----------------------------------------------
        step(4, "When did it change?",
             "How closely the order of all 30 factors agrees between each pair of periods. "
             "1.00 would mean exactly the same order.")
        tag = pick("Model", ["primary", "secondary"], "primary", "t_model",
                   lambda k: {"primary": "XGBoost (main)", "secondary": "Random forest (check)"}[k])
        M = BY_TAG[tag]
        sp = M["spearman"]
        if sp is not None:
            a, b = st.columns([1.1, 1], gap="large")
            with a:
                hm = sp.reset_index(names="p1").melt("p1", var_name="p2", value_name="rho")
                hm = hm[hm.p1 != hm.p2]
                lo = float(np.floor(hm.rho.min() * 20) / 20)
                hm["ink"] = np.where(hm.rho >= lo + (1 - lo) * 0.55, "#ffffff", "#0b0b0b")
                hm["p1"] = hm.p1.map(nice_period)
                hm["p2"] = hm.p2.map(nice_period)
                order = [nice_period(p) for p in ps]
                cells = alt.Chart(hm).mark_rect(cornerRadius=4, stroke=T["bg"],
                                                strokeWidth=3).encode(
                    x=alt.X("p2:N", title=None, sort=order,
                            axis=alt.Axis(labelAngle=0, orient="top")),
                    y=alt.Y("p1:N", title=None, sort=order),
                    color=alt.Color("rho:Q", title="Agreement (1 = same order)",
                                    scale=alt.Scale(domain=[lo, 1], range=BLUES[:6]),
                                    legend=alt.Legend(orient="bottom", gradientLength=220)),
                    tooltip=[alt.Tooltip("p1:N", title="Period"),
                             alt.Tooltip("p2:N", title="vs"),
                             alt.Tooltip("rho:Q", title="Agreement", format=".3f")])
                txt = alt.Chart(hm).mark_text(fontSize=13, fontWeight=600).encode(
                    x=alt.X("p2:N", sort=order), y=alt.Y("p1:N", sort=order),
                    text=alt.Text("rho:Q", format=".2f"),
                    color=alt.Color("ink:N", scale=None))
                show((cells + txt).resolve_scale(color="independent").properties(height=300))
            with b:
                same = [sp.loc[x, y] for i, x in enumerate(ps) for y in ps[i + 1:]
                        if not crosses_break(x, y)]
                cross = [sp.loc[x, y] for i, x in enumerate(ps) for y in ps[i + 1:]
                         if crosses_break(x, y)]
                st.write("")
                kpi(f"{np.mean(same):.2f}", "average agreement, periods on the same side "
                    "of 2015/16")
                st.write("")
                kpi(f"{np.mean(cross):.2f}", "average agreement, periods either side of "
                    "2015/16")
                pv = [(sp.loc[x, y], x, y) for i, x in enumerate(ps) for y in ps[i + 1:]]
                hi, lo_ = max(pv), min(pv)
                st.markdown(
                    f'<div class="small" style="margin-top:12px">Darker = more alike. '
                    f'Most alike: {nice_period(hi[1])} and {nice_period(hi[2])} '
                    f'({hi[0]:.2f}). Least alike: {nice_period(lo_[1])} and '
                    f'{nice_period(lo_[2])} ({lo_[0]:.2f}). These raw numbers use the '
                    f'full periods; the fair test against chance is Step 1.</div>',
                    unsafe_allow_html=True)
        else:
            need(M, "spearman")
    with tabs[4]:
        # ---- STEP 5 - does the check model agree? --------------------------
        step(5, "Do the results hold up?",
             "Two checks. First, a different model. Second, the same model with every "
             "period given the same share of severe crashes.")
        st.markdown("##### Check 1 · Does a different model see the same changes?")
        st.markdown('<div class="small">The random forest was run the same way. It has '
                    'no chance test of its own (too slow to compute), so this is a '
                    'direction check only.</div>', unsafe_allow_html=True)
        sec = load_model(str(FOLDER), "secondary", _stamp(FOLDER))
        if sec["ranks"] is not None and moved:
            r2 = sec["ranks"]
            rows = []
            for q in moved:
                d1 = r.loc[q, ps[-1]] - r.loc[q, ps[0]]
                d2 = r2.loc[q, ps[-1]] - r2.loc[q, ps[0]] if q in r2.index else np.nan
                same_dir = (np.sign(d1) == np.sign(d2)) if d1 != 0 else None
                rows.append({"Factor": label(q),
                             "Main model rank": f"#{int(r.loc[q, ps[0]])} → #{int(r.loc[q, ps[-1]])}",
                             "Check model rank": f"#{int(r2.loc[q, ps[0]])} → #{int(r2.loc[q, ps[-1]])}"
                             if q in r2.index else "–",
                             "Same direction?": "✓ Yes" if same_dir else
                             ("✗ No" if same_dir is False else "– rank unchanged")})
            table(pd.DataFrame(rows))
            n_same = sum(x["Same direction?"].startswith("✓") for x in rows)
            m1 = np.mean([abs(r.loc[q, ps[-1]] - r.loc[q, ps[0]]) for q in moved])
            m2 = np.mean([abs(r2.loc[q, ps[-1]] - r2.loc[q, ps[0]]) for q in moved
                          if q in r2.index])
            size = (" The check model's moves are smaller, so the <b>direction</b> holds "
                    "across models but the <b>size</b> does not." if m2 < m1 else "")
            st.markdown(f'<div class="small">{n_same} of {len(rows)} changes go the same way '
                        f'in both models.{size}</div>', unsafe_allow_html=True)
        else:
            st.caption("Check-model rankings not found.")

        # ---- class balance check -------------------------------------------
        st.write("")
        st.markdown("##### Check 2 · Is it just the share of severe crashes changing?")
        cb = CHK["balance"]
        if cb is None:
            st.caption("Run `python class_balance_check.py` to add this check.")
        else:
            tgt = cb["target_severe_rate"] * 100
            st.markdown(
                f'<div class="small">Severe crashes rise from '
                f'{min(cb["rates_before"].values()) * 100:.2f}% to {tgt:.2f}% of all crashes. '
                f'To test whether that alone explains the change, every period was given '
                f'the same rate ({tgt:.2f}%) by randomly leaving out some non-severe crashes '
                f'(never a severe one), and the main model was run again '
                f'{cb["seeds"]} times.</div>', unsafe_allow_html=True)
            st.write("")
            moved_p = [p for p, z in cb["sizes"].items() if z["rows"] != z["rows_before"]]
            lo_in = min(cb["within_period_agreement"][p]["min"] for p in moved_p) if moved_p else 1
            fac = cb.get("changed_factors", {})
            n_same = sum(v["same_direction"] for v in fac.values())
            k = st.columns(3)
            with k[0]:
                kpi(f"{lo_in:.3f}", f"lowest agreement with the original ranking – above "
                    f"the chance floor of {cb['noise_floor']:.3f}, so each period barely moves")
            with k[1]:
                kpi(f'{cb["same_side_mean"]["equalised"]:.2f} vs {cb["across_mean"]["equalised"]:.2f}',
                    "agreement on the same side vs across 2015/16 – the break "
                    + ("is still there" if cb["break_survives"] else "is no longer clear"))
            with k[2]:
                kpi(f"{n_same} of {len(fac)}", "changed factors still move the same way")
            rows = [{"Factor": label(q),
                     "Share, original": f'{v["share_original_pct"][0]:.1f}% → {v["share_original_pct"][1]:.1f}%',
                     "Share, same severe rate": f'{v["share_equalised_pct"][0]:.1f}% → {v["share_equalised_pct"][1]:.1f}%',
                     "Rank, original": f'#{v["rank_original"][0]} → #{v["rank_original"][1]}',
                     "Rank, same severe rate": f'#{v["rank_equalised"][0]} → #{v["rank_equalised"][1]}',
                     "Same direction?": "✓ Yes" if v["same_direction"] else "✗ No"}
                    for q, v in fac.items()]
            st.write("")
            table(pd.DataFrame(rows))
            st.markdown(
                '<div class="small"><b>Answer:</b> ' +
                ("the rising share of severe crashes does <b>not</b> explain the change. "
                 "Some changes get a little smaller, but none changes direction."
                 if not cb["overall_rate_explains_finding"] else
                 "the rising share of severe crashes explains part of the change – see the "
                 "rows above.") + '</div>', unsafe_allow_html=True)

        # ---- recording caveat ----------------------------------------------
        st.write("")
        with st.container(border=True):
            st.markdown("**Could part of this be how crashes are recorded?** Yes – "
                        "that can't be ruled out.")
            msg = ("Recording practice changed most at the same 2015/16 boundary. ")
            if CR is not None and CR.repeated_pct.notna().all():
                msg += ("One sign: the share of crash records whose 30 recorded facts "
                        "exactly repeat an earlier record in the same period fell from " +
                        " → ".join(f"{x:.1f}%" for x in CR.repeated_pct) +
                        ". ")
            msg += ("So the break may partly reflect <b>how</b> crashes were written "
                    "down, not only <b>how</b> they happened. The analysis shows that the "
                    "model's reliance changed; it cannot show why.")
            st.markdown(f'<div class="small">{msg}</div>', unsafe_allow_html=True)

    st.write("")
    not_mean([
        "A real change in the model's reliance does not prove the road toll changed "
        "for that reason. For example, <i>Weather</i> rising does not mean weather "
        "became more dangerous – it means the model leans on it more to tell severe "
        "crashes apart.",
        "<i>Region</i> falling is not a judgement on any region. Regions are not "
        "ranked or compared here.",
        "“Within chance” does not prove nothing changed. It means any change is too "
        "small to tell apart from random wobble with this data.",
    ])


# --------------------------------------------------------------------------
pages = [
    st.Page(page_overview, title="Overview", icon=":material/home:", default=True),
    st.Page(page_factors, title="Factors (RQ1)", icon=":material/leaderboard:",
            url_path="factors"),
    st.Page(page_temporal, title="Temporal (RQ2)", icon=":material/timeline:",
            url_path="temporal"),
]
st.navigation(pages).run()
