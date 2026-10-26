
import html
import importlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import link_checker

importlib.reload(link_checker)

from link_checker import (FREE_HOSTS, MODEL_DIR, OPEN_PLATFORMS, check_link, explain_score,  # noqa: E402
                          link_text, load_checker, split_link)

ROOT = Path(__file__).parent

# colours: blue and orange always mean legitimate and phishing (same as the notebook figures),
# teal always means trust, violet is used for page features
ORDER = ["Legitimate", "Phishing"]
COLORS = {"Legitimate": "#2a78d6", "Phishing": "#eb6834"}
VIOLET, TEAL, AMBER, NAVY = "#6b58d9", "#00968d", "#e8a127", "#15284b"
TEAL_RAMP = ["#5ac6bd", "#48b0a8", "#369a93", "#23857f", "#0a716b", "#005d58", "#004a46", "#003834"]  # trust 0 to 7
INK, INK_SOFT, GRID, AXIS = "#1b2a4a", "#4b5a78", "#e9edf4", "#cdd5e3"
PAGE_SIZES = ["tiny (<50)", "small (50-500)", "medium (500-2k)", "large (>2k)"]

# columns people can pick in the charts, under names that make sense without the notebook
FEATURES = {
    "Lines of code": "LineOfCode",
    "Links to other sites": "NoOfExternalRef",
    "Links to the same site": "NoOfSelfRef",
    "Images": "NoOfImage",
    "JavaScript files": "NoOfJS",
    "CSS files": "NoOfCSS",
    "Domain length": "domain_length",
    "URL length": "URLLength",
}
SKEWED = {"LineOfCode", "NoOfExternalRef", "NoOfSelfRef", "NoOfImage", "NoOfJS", "NoOfCSS"}   # log scale by default

# the columns checked for outliers (the same as section 8 of the notebook)
OUTLIER_COLUMNS = {
    "URLLength": "URL length",
    "domain_length": "Domain length",
    "LineOfCode": "Lines of code",
    "LargestLineLength": "Longest line of code",
    "NoOfImage": "Images",
    "NoOfCSS": "CSS files",
    "NoOfJS": "JavaScript files",
    "NoOfSelfRef": "Links to the same site",
    "NoOfExternalRef": "Links to other sites",
    "NoOfEmptyRef": "Empty links",
}

# the 7 yes/no columns that add up to trust_score
TRUST_SIGNALS = {
    "HasTitle": "A title in the browser tab",
    "HasFavicon": "A small logo in the tab",
    "HasDescription": "A description for search engines",
    "HasSocialNet": "Links to social media",
    "HasCopyrightInfo": "A copyright line",
    "IsResponsive": "A layout that works on phones",
    "Robots": "A robots.txt file",
}
TRUST_HELP = ("A checklist score made in the notebook: 1 point each for a title, a tab logo, a description, social "
              "media links, a copyright line, a phone layout and a robots.txt file. 0 = none, 7 = all.")
LOG_HELP = "Each step on the axis is 10 times bigger (1, 10, 100, 1,000), so tiny and huge values both fit."

CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap");
[data-testid="InputInstructions"], [data-testid="stHeaderActionElements"] { display: none; }
.block-container { padding-top: 4.2rem; }

/* top bar: the same navy as the sidebar, so the app has one frame */
[data-testid="stHeader"] { background: #15284b; }
[data-testid="stTopNavLink"] { border-radius: 8px; }
[data-testid="stTopNavLink"] p { color: #c3cfe8; }
[data-testid="stTopNavLink"]:hover { background: rgba(255, 255, 255, 0.08); }
[data-testid="stTopNavLink"][aria-current="page"] { background: rgba(255, 255, 255, 0.14); }
[data-testid="stTopNavLink"][aria-current="page"] p { color: #ffffff; font-weight: 600; }
[data-testid="stToolbar"] button, [data-testid="stSidebarCollapseButton"] button { color: #c3cfe8; }
[data-testid="stSidebar"] h2 { color: #ffffff; }

/* tabs as pills */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; background: #e3e8f2; padding: 4px; border-radius: 12px; width: fit-content;
                                          border: none; }
[data-testid="stTab"] { padding: 6px 16px; border-radius: 9px; margin: 0; }
[data-testid="stTab"] p { color: #4b5a78; font-weight: 500; }
[data-testid="stTab"][aria-selected="true"] { background: #ffffff; box-shadow: 0 1px 3px rgba(21, 40, 75, 0.15); }
[data-testid="stTab"][aria-selected="true"] p { color: #5a45cf; font-weight: 600; }
.react-aria-SelectionIndicator { display: none; }

/* white cards for charts and tables */
[class*="st-key-card"] { background: #ffffff; border: 1px solid #e3e8f1; border-radius: 16px; padding: 18px 20px 10px;
                         box-shadow: 0 1px 2px rgba(21, 40, 75, 0.04), 0 6px 20px rgba(21, 40, 75, 0.05); }

/* section titles with a coloured mark */
.section { display: flex; align-items: center; gap: 12px; margin: 30px 0 4px; }
.section-mark { width: 8px; height: 28px; border-radius: 4px; }
.section-title { font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-size: 1.55rem; font-weight: 700;
                 color: #1b2a4a; }
.section-note { color: #4b5a78; margin: 0 0 10px 20px; }

/* dashboard banner */
.hero { background: #15284b; border-radius: 20px; padding: 28px 30px 22px; color: #ffffff; }
.hero-title { font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-size: 2.5rem; font-weight: 700;
              line-height: 1.1; letter-spacing: -0.01em; }
.hero-sub { color: #b8c5e2; margin-top: 6px; font-size: 1.02rem; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 22px; }
.kpi { background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.10); border-top: 4px solid var(--c);
       border-radius: 12px; padding: 12px 16px 12px; }
.kpi-label { color: #b8c5e2; font-size: 0.86rem; }
.kpi-value { font-size: 2rem; font-weight: 600; line-height: 1.25; }
.kpi-note { color: #8fa1c7; font-size: 0.8rem; }

/* explainers */
.explain code { background: #ece9fb; color: #4a36b8; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }
.explain-title { font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-size: 1.25rem; font-weight: 700;
                 color: #1b2a4a; margin-bottom: 4px; }
.explain p { color: #4b5a78; margin: 0 0 12px; line-height: 1.55; }
.checklist { width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #1b2a4a; }
.checklist th { text-align: left; font-weight: 600; padding: 6px 8px; border-bottom: 2px solid #e3e8f1; vertical-align: bottom; }
.checklist th span { display: block; font-weight: 400; color: #4b5a78; font-size: 0.8rem; }
.checklist td { padding: 6px 8px; border-bottom: 1px solid #eef1f6; }
.checklist td.c, .checklist th.c { text-align: center; }
.checklist tr.total td { font-weight: 700; border-bottom: none; font-size: 1rem; }
.dot { display: inline-block; width: 14px; height: 14px; border-radius: 50%; vertical-align: middle; }
.dot.yes { background: #00968d; }
.dot.no { border: 2px solid #cdd5e3; width: 10px; height: 10px; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }

/* narrower column for the text-heavy pages */
.st-key-reading { max-width: 780px; margin-inline: auto; }
.st-key-reading li { margin-bottom: 0.75rem; }
.page-title { font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-size: 2.4rem; font-weight: 700;
              color: #1b2a4a; line-height: 1.1; }
.page-bar { width: 64px; height: 6px; border-radius: 3px; margin: 10px 0 12px; }
.page-intro { color: #4b5a78; font-size: 1.05rem; margin-bottom: 6px; }

/* signs: one colour per group */
.group { display: flex; align-items: center; gap: 10px; margin: 22px 0 6px; }
.group-mark { width: 12px; height: 12px; border-radius: 3px; }
.group-title { font-family: "IBM Plex Sans Condensed", "IBM Plex Sans Arabic", sans-serif; font-size: 1.3rem; font-weight: 700;
               color: #1b2a4a; }
[class*="st-key-g-link"] li::marker { color: #eb6834; }
[class*="st-key-g-page"] li::marker { color: #6b58d9; }
[class*="st-key-g-msg"] li::marker { color: #00968d; }
[class*="st-key-closing"] { background: #ece9fb; border-radius: 12px; padding: 12px 18px 2px; margin-top: 14px; }

/* Arabic tab reads right to left, links inside it stay left to right */
.st-key-arabic { direction: rtl; }
.st-key-arabic :is(div, h3, p, ul, li) { text-align: right; }
.st-key-arabic, .st-key-arabic strong, .st-key-arabic .group-title { font-family: "IBM Plex Sans Arabic", "IBM Plex Sans", sans-serif; }
.st-key-arabic p, .st-key-arabic li { line-height: 1.8; }
.st-key-arabic ul { padding-right: 1.25rem; padding-left: 0; }
.st-key-arabic code { direction: ltr; unicode-bidi: isolate; }

/* link checker */
.verdict { border-radius: 16px; padding: 18px 22px; margin: 18px 0 16px; border-left: 8px solid var(--c); background: var(--bg); }
.verdict-title { font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-size: 1.9rem; font-weight: 700;
                 line-height: 1.15; color: var(--ink); }
.verdict-text { color: #1b2a4a; margin-top: 4px; }
.anatomy { font-family: "IBM Plex Mono", monospace; font-size: 1.05rem; line-height: 1.7; padding: 14px 16px;
           background: #ffffff; border: 1px solid #e3e8f1; border-radius: 12px; word-break: break-all; }
.anatomy .rest { color: #8a96ad; }
.anatomy .real { color: #1b2a4a; font-weight: 600; background: #ffe9b0; border-radius: 4px; padding: 1px 4px; }
.anatomy-note { color: #4b5a78; font-size: 0.9rem; margin: 6px 0 18px; }
.meter-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; color: #1b2a4a; }
.meter-head strong { font-size: 1.5rem; font-weight: 700; }
.meter-track { position: relative; height: 14px; border-radius: 7px; }
.meter-dot { position: absolute; top: -5px; width: 24px; height: 24px; margin-left: -12px; border-radius: 50%;
             background: #ffffff; border: 4px solid #1b2a4a; box-sizing: border-box; }
.meter-scale { display: flex; font-size: 0.8rem; color: #4b5a78; margin-top: 8px; }
.meter-scale span { text-align: center; }

/* insights */
.insights { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.insight { background: #ffffff; border: 1px solid #e3e8f1; border-top: 5px solid var(--c); border-radius: 14px;
           padding: 14px 18px 6px; color: #1b2a4a; }
.insight-title { font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-size: 1.15rem; font-weight: 700; }
.insight ul { padding-left: 1.1rem; margin: 8px 0; }
.insight li { margin-bottom: 6px; line-height: 1.5; }
.insight li::marker { color: var(--c); }
.insight code { background: #ece9fb; color: #4a36b8; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }
@media (max-width: 900px) { .insights { grid-template-columns: 1fr; } }

/* confusion matrix */
.cm { width: 100%; border-collapse: separate; border-spacing: 8px; color: #1b2a4a; }
.cm th { font-weight: 600; font-size: 0.86rem; color: #4b5a78; text-align: center; padding: 2px 4px; }
.cm th.row { text-align: right; width: 26%; }
.cm td { border-radius: 12px; padding: 14px 10px; text-align: center; }
.cm td.right { background: #e3f4f2; }
.cm td.wrong { background: #fdeee6; }
.cm .n { font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
.cm td.right .n { color: #00756e; }
.cm td.wrong .n { color: #c4501d; }
.cm .what { font-size: 0.84rem; color: #4b5a78; }

</style>
"""

# one chart style for the whole app: hairline grid, text in ink colours
pio.templates["dashboard"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, sans-serif", size=13, color=INK_SOFT),
    margin=dict(l=8, r=8, t=30, b=8),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title_font_size=12, automargin=True),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title_font_size=12, automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK)),
    barcornerradius=4,
))


# ---------------------------------------------------------------- loading
@st.cache_data(show_spinner="Loading data...")
def load_data():
    cols = ["URL", "TLD", "tld_group", "is_phishing", "label_name", "URLLength", "domain_length", "LineOfCode",
            "LargestLineLength", "NoOfImage", "NoOfJS", "NoOfCSS", "NoOfSelfRef", "NoOfExternalRef", "NoOfEmptyRef",
            "trust_score", "page_size_group"]
    return pd.read_csv(ROOT / "clean_features.csv", usecols=cols + list(TRUST_SIGNALS))[cols + list(TRUST_SIGNALS)]


@st.cache_data
def load_cleaning_summary():
    return json.loads((ROOT / "before_after.json").read_text(encoding="utf-8"))


def get_checker():
    # keyed on when the model file was saved, so re-running the notebook gives the app the new model
    return load_checker_once((MODEL_DIR / "link_model.joblib").stat().st_mtime)


@st.cache_resource(show_spinner="Loading the link model...")
def load_checker_once(saved_at):
    return load_checker()


# ---------------------------------------------------------------- page building blocks
def section(title, color, note=None):
    st.html(f'<div class="section"><div class="section-mark" style="background:{color}"></div>'
            f'<div class="section-title">{title}</div></div>' + (f'<div class="section-note">{note}</div>' if note else ""))


def page_header(title, intro, color):
    st.html(f'<div class="page-title">{title}</div><div class="page-bar" style="background:{color}"></div>'
            f'<div class="page-intro">{intro}</div>')


def show(fig, height=340):
    # set on every figure: Streamlit swaps the default template and fills empty backgrounds with its own grey
    fig.update_layout(template="dashboard", height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def chart_title(title, note):
    st.markdown(f"#### {title}")
    st.caption(note)


def log_axis(values):
    """Ticks for an axis that shows log10(x + 1), labelled with the real numbers."""
    ticks = [t for t in [0, 10, 100, 1_000, 10_000, 100_000] if t <= max(values.max(), 1)]
    return dict(tickvals=[np.log10(t + 1) for t in ticks],
                ticktext=[f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])


def feature_picker(key, default="Lines of code"):
    """Selectbox + log toggle. Returns the column, its readable name and whether to use a log scale."""
    row = st.container(horizontal=True, vertical_alignment="bottom", gap="medium")
    name = row.selectbox("Feature", list(FEATURES), index=list(FEATURES).index(default), key=f"{key}_feature", width=240)
    col = FEATURES[name]
    log = row.toggle("Log scale", value=col in SKEWED, key=f"{key}_log", help=LOG_HELP)
    return col, name, log


def rate_chart(rates, x, y, overall, horizontal=False, colors=None):
    """Bar chart of 'share phishing' with a line for the overall rate."""
    fig = px.bar(rates, x=x, y=y, orientation="h" if horizontal else "v", custom_data=["n"],
                 color_discrete_sequence=[COLORS["Phishing"]])
    if colors:
        fig.update_traces(marker_color=colors)
    label = "%{y}" if horizontal else "%{x}"
    value = "%{x:.0%}" if horizontal else "%{y:.0%}"
    fig.update_traces(hovertemplate=f"{label}: {value} phishing<br>%{{customdata[0]:,}} links<extra></extra>")
    line = dict(line_color=INK, line_width=1, line_dash="dot", annotation_text=f"overall {overall:.0%}",
                annotation_font_color=INK_SOFT, annotation_font_size=11)
    if horizontal:
        fig.add_vline(x=overall, annotation_position="top", **line)
        fig.update_layout(xaxis=dict(range=[0, 1.05], tickformat=".0%", title="share phishing"),
                          yaxis=dict(title="", showgrid=False), bargap=0.35)
    else:
        fig.add_hline(y=overall, annotation_position="top right", **line)
        fig.update_layout(yaxis=dict(range=[0, 1.05], tickformat=".0%", title="share phishing"),
                          xaxis=dict(showgrid=False), bargap=0.45)
    return fig


# ---------------------------------------------------------------- page 1: dashboard
def page_dashboard():
    df = load_data()

    with st.sidebar:
        st.header("Filters")
        ending = st.selectbox("Website ending", ["All"] + df["tld_group"].value_counts().index.tolist(),
                              help="The 10 most common endings. All the others are grouped as 'other'.")
        trust = st.slider("Trust score", 0, 7, (0, 7), help=TRUST_HELP)
        classes = st.multiselect("Class", ORDER, default=ORDER)
        st.divider()
        st.caption("Data: [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)"
                   " from UCI. About 235k website links marked as phishing or legitimate, with details about the "
                   "link and the page it opens.")

    sel = df[df["trust_score"].between(*trust) & df["label_name"].isin(classes)]
    if ending != "All":
        sel = sel[sel["tld_group"] == ending]

    banner(df, sel)
    if sel.empty:
        st.warning("No links match these filters. Try a wider trust score range or another ending.")
        return

    section("What is the trust score?", TEAL, "Many charts below use it, so here is how it's made.")
    with st.container(key="card-explain-trust"):
        trust_explainer()

    section("Charts", COLORS["Phishing"], "Univariate charts look at one column at a time. Bivariate charts compare "
            "two, usually a column against phishing or legitimate. The sidebar filters change every chart.")
    uni, bi = st.tabs(["Univariate", "Bivariate"])
    with uni:
        univariate_charts(sel)
    with bi:
        bivariate_charts(sel, ending)

    section("Data preview", COLORS["Legitimate"])
    with st.container(key="card-data"):
        data_preview(sel)

    section("Model results", VIOLET, "Trained in the notebook and tested on websites the models never saw. "
            "These results don't change with the filters.")
    with st.container(key="card-models"):
        model_results()

    section("Insights", AMBER, "What I found, looking at the whole dataset.")
    insights()

    section("What the dataset means", NAVY)
    with st.container(key="card-dataset"):
        dataset_meaning()


def banner(df, sel):
    """The navy banner at the top of the dashboard with the four headline numbers."""
    empty = sel.empty
    kpis = [
        ("#5b9bf0", "Links", f"{len(sel):,}", "matching the filters"),
        ("#f59a6c", "Phishing share", "-" if empty else f"{sel['is_phishing'].mean():.1%}", "of those links"),
        ("#a397f2", "Median page size", "-" if empty else f"{sel['LineOfCode'].median():,.0f}", "lines of HTML"),
        ("#5ac6bd", "Median trust score", "-" if empty else f"{sel['trust_score'].median():.0f} of 7", "trust signals"),
    ]
    tiles = "".join(f'<div class="kpi" style="--c:{c}"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'
                    for c, label, value, note in kpis)
    st.html(f'<div class="hero"><div class="hero-title">Phishing URL dashboard</div>'
            f'<div class="hero-sub">What makes a phishing link different from a real one, based on {len(df):,} '
            f'links from the PhiUSIIL dataset.</div><div class="kpis">{tiles}</div></div>')


@st.cache_data
def trust_examples():
    """Two real rows from the data, to show how the checklist adds up."""
    df = load_data()
    real = df[df["URL"] == "https://www.aap.org"].iloc[0]
    phish = df[df["URL"] == "http://www.f0519141.xsph.ru"].iloc[0]
    shares = df.groupby("label_name")[list(TRUST_SIGNALS)].mean()
    medians = df.groupby("label_name")["trust_score"].median()
    return real, phish, shares, medians


def trust_explainer():
    real, phish, shares, medians = trust_examples()
    dot = lambda v: '<span class="dot yes"></span>' if v else '<span class="dot no"></span>'
    rows = "".join(
        f'<tr><td>{label}</td><td class="c">{dot(real[col])}</td><td class="c">{dot(phish[col])}</td>'
        f'<td class="c">{shares.loc["Legitimate", col]:.0%}</td><td class="c">{shares.loc["Phishing", col]:.0%}</td></tr>'
        for col, label in TRUST_SIGNALS.items())
    total = (f'<tr class="total"><td>Trust score</td><td class="c">{real["trust_score"]} of 7</td>'
             f'<td class="c">{phish["trust_score"]} of 7</td><td class="c">median {medians["Legitimate"]:.0f}</td>'
             f'<td class="c">median {medians["Phishing"]:.0f}</td></tr>')
    st.html(f'<div class="explain"><div class="explain-title">Trust score: a checklist of 7 things real sites have</div>'
            f'<p>It isn\'t in the original data. The notebook adds it up from 7 yes/no columns: each page gets '
            f'1 point for every item it has, so the score goes from 0 (none) to 7 (all). Real businesses tick most '
            f'boxes. Throw-away phishing pages tick almost none.</p>'
            f'<table class="checklist"><thead><tr><th>Trust signal</th><th class="c">aap.org<span>a real site</span></th>'
            f'<th class="c">f0519141.xsph.ru<span>a phishing page</span></th><th class="c">Real sites<span>that have it</span></th>'
            f'<th class="c">Phishing<span>that has it</span></th></tr></thead><tbody>{rows}{total}</tbody></table></div>')


@st.cache_data
def insight_numbers():
    df = load_data()
    med = df.groupby("label_name")[["LineOfCode", "NoOfImage", "NoOfCSS", "NoOfJS", "domain_length", "trust_score"]].median()
    tld = df.groupby("TLD")["is_phishing"].agg(["mean", "size"]).query("size >= 500").nlargest(5, "mean")
    return {"share": df["is_phishing"].mean(), "med": med, "tld": tld,
            "trust": df.groupby("trust_score")["is_phishing"].mean(),
            "size": df.groupby("page_size_group")["is_phishing"].mean()}


def link_model_line():
    try:
        s = get_checker()["test_scores"]
    except FileNotFoundError:
        return "The link checker only reads the website name, so it's much weaker than the full model."
    return (f"The link checker only reads the website name. It gets about {s['accuracy']:.0%} right and catches "
            f"{s['recall (phishing)']:.0%} of phishing links.")


def insights():
    f = insight_numbers()
    m, ph, lg = f["med"], "Phishing", "Legitimate"
    endings = ", ".join(f"<code>.{t}</code>" for t in f["tld"].index)
    groups = [
        (COLORS["Phishing"], "What phishing links look like", [
            f"Phishing pages are tiny. The typical one has {m.loc[ph, 'LineOfCode']:.0f} lines of code, while a real "
            f"site has about {m.loc[lg, 'LineOfCode']:,.0f}.",
            f"They barely have any trust signals: usually {m.loc[ph, 'trust_score']:.0f} out of 7, against "
            f"{m.loc[lg, 'trust_score']:.0f} for real sites.",
            f"Cheap endings like {endings} are basically all phishing.",
        ]),
        (TEAL, "What goes together", [
            f"More trust signals means less phishing. With none it's {f['trust'].loc[0]:.0%} phishing, with 6 it "
            f"almost never is.",
            f"Small pages are risky. Under 50 lines it's {f['size'].loc['tiny (<50)']:.0%} phishing, over 2,000 "
            f"lines only {f['size'].loc['large (>2k)']:.1%}.",
            "Oddly, phishing pages have fewer password boxes than real ones, because a lot of them were already "
            "dead when the data was collected.",
        ]),
        (AMBER, "The leakage problem", [
            "Every real link in this data is a plain <code>https://www.</code> homepage, so some columns basically "
            "give the answer away.",
            "I left 21 of those columns out of the model. With them it would score 100% here and then fail on real links.",
        ]),
        (VIOLET, "The models", [
            "Without the leaky columns, the full model still gets 99.8% right, but it needs the page itself.",
            link_model_line(),
        ]),
    ]
    cards = "".join(f'<div class="insight" style="--c:{c}"><div class="insight-title">{title}</div><ul>'
                    + "".join(f"<li>{x}</li>" for x in items) + "</ul></div>" for c, title, items in groups)
    st.html(f'<div class="insights">{cards}</div>')


# what every column of clean_features.csv means
COLUMN_GUIDE = [
    ("URL", "the link", "The full link"),
    ("URLLength", "the link", "Characters in the whole link"),
    ("Domain", "the link", "The website part of the link, e.g. www.uni-mainz.de"),
    ("DomainLength", "the link", "Characters in the domain, www. included"),
    ("IsDomainIP", "the link", "1 if the website is a number like 185.12.4.9 instead of a name"),
    ("TLD", "the link", "The ending: com, org, sa, top..."),
    ("URLSimilarityIndex", "the link", "How similar the link is to known real links, 0 to 100 (gives the answer away)"),
    ("CharContinuationRate", "the link", "How long the unbroken runs of letters, digits or symbols are"),
    ("TLDLegitimateProb", "the link", "How common the ending is among real websites, 0 to 1"),
    ("URLCharProb", "the link", "How typical the link's characters are for real links"),
    ("TLDLength", "the link", "Letters in the ending"),
    ("NoOfSubDomain", "the link", "Parts in front of the name, www counted"),
    ("HasObfuscation", "the link", "1 if the link hides characters with codes like %20"),
    ("NoOfObfuscatedChar", "the link", "How many hidden characters"),
    ("ObfuscationRatio", "the link", "Share of the link that is hidden characters"),
    ("NoOfLettersInURL", "the link", "Letters in the link"),
    ("LetterRatioInURL", "the link", "Share of the link that is letters"),
    ("NoOfDigitsInURL", "the link", "Digits in the link"),
    ("DigitRatioInURL", "the link", "Share of the link that is digits"),
    ("NoOfEqualsInURL", "the link", "= signs (they appear when a link carries data)"),
    ("NoOfQMarkInURL", "the link", "? signs"),
    ("NoOfAmpersandInURL", "the link", "& signs"),
    ("NoOfOtherSpecialCharsInURL", "the link", "Other symbols"),
    ("SpecialCharRatioInURL", "the link", "Share of the link that is symbols"),
    ("IsHTTPS", "the link", "1 if the link starts with https://"),
    ("LineOfCode", "the page", "Lines of HTML code: how big the page is"),
    ("LargestLineLength", "the page", "Characters in the longest line of code"),
    ("HasTitle", "the page", "1 if the browser tab has a title (trust signal)"),
    ("DomainTitleMatchScore", "the page", "How much the page title matches the domain, 0 to 100"),
    ("URLTitleMatchScore", "the page", "How much the page title matches the link, 0 to 100"),
    ("HasFavicon", "the page", "1 if there is a small logo in the browser tab (trust signal)"),
    ("Robots", "the page", "1 if the site has a robots.txt file for search engines (trust signal)"),
    ("IsResponsive", "the page", "1 if the page adapts to phone screens (trust signal)"),
    ("NoOfURLRedirect", "the page", "Times the page sends you to another address"),
    ("NoOfSelfRedirect", "the page", "Times the page sends you to another address on the same site"),
    ("HasDescription", "the page", "1 if the page has a description for search engines (trust signal)"),
    ("NoOfPopup", "the page", "Pop-up windows"),
    ("NoOfiFrame", "the page", "Frames that show another page inside this one"),
    ("HasExternalFormSubmit", "the page", "1 if a form sends what you type to another website"),
    ("HasSocialNet", "the page", "1 if there are links to social media (trust signal)"),
    ("HasSubmitButton", "the page", "1 if there is a submit button"),
    ("HasHiddenFields", "the page", "1 if a form has hidden fields"),
    ("HasPasswordField", "the page", "1 if there is a password box"),
    ("Bank", "the page", "1 if the page mentions banking words"),
    ("Pay", "the page", "1 if the page mentions payment words"),
    ("Crypto", "the page", "1 if the page mentions crypto words"),
    ("HasCopyrightInfo", "the page", "1 if there is a copyright line (trust signal)"),
    ("NoOfImage", "the page", "Images on the page"),
    ("NoOfCSS", "the page", "Style files"),
    ("NoOfJS", "the page", "JavaScript files"),
    ("NoOfSelfRef", "the page", "Links to other pages of the same site"),
    ("NoOfEmptyRef", "the page", "Empty links that go nowhere"),
    ("NoOfExternalRef", "the page", "Links to other websites"),
    ("is_phishing", "the answer", "1 = phishing, 0 = real (made from the original label, which was the other way round)"),
    ("label_name", "the answer", "The same as words: Phishing or Legitimate"),
    ("has_www", "added", "1 if the link has www. (only to prove the leak)"),
    ("has_path", "added", "1 if something comes after the domain, like /login (only to prove the leak)"),
    ("domain_length", "added", "Characters in the domain without www."),
    ("domain_digits", "added", "Digits in the domain"),
    ("domain_hyphens", "added", "Hyphens (-) in the domain"),
    ("domain_dots", "added", "Dots in the domain"),
    ("trust_score", "added", "How many of the 7 trust signals the page has, 0 to 7"),
    ("tld_group", "added", "The 10 most common endings, everything else called other"),
    ("page_size_group", "added", "Lines of code cut into tiny, small, medium and large"),
]


def dataset_meaning():
    real, phish, _, _ = trust_examples()
    st.html('<div class="explain"><p>The <b>PhiUSIIL Phishing URL Dataset</b> was published on the '
            '<a href="https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset">UCI Machine Learning '
            'Repository</a> in 2024 by Arvind Prasad and Shalini Chandra. It has 235,795 website links: 134,850 real '
            'ones and 100,945 phishing ones. For every link, the authors saved facts about the link text and about '
            'the web page it opened, taken from the page\'s source code.</p>'
            '<p><b>One row is one link.</b> Its columns describe the link (length, ending, digits...), the page '
            '(lines of code, images, a password box...) and the answer (phishing or not). After cleaning, '
            '235,370 links and 64 columns are left. Here are two real rows, one of each kind:</p></div>')
    fields = [("URL", "The link"), ("TLD", "Its ending"), ("domain_length", "Characters in the website name"),
              ("LineOfCode", "Lines of HTML on the page"), ("NoOfImage", "Images on the page"),
              ("NoOfExternalRef", "Links to other websites"), ("trust_score", "Trust signals, 0 to 7"),
              ("label_name", "The answer")]
    st.dataframe(pd.DataFrame([{"Column": col, "What it means": meaning, "A real site": str(real[col]),
                                "A phishing page": str(phish[col])} for col, meaning in fields]), hide_index=True)
    with st.expander("What every column means (all 64)"):
        st.dataframe(pd.DataFrame(COLUMN_GUIDE, columns=["Column", "About", "What it means"]), hide_index=True,
                     height=420)


def univariate_charts(sel):
    left, right = st.columns(2, gap="medium")
    with left, st.container(key="card-uni-balance"):
        chart_title("Class balance", "Fairly balanced, so no resampling is needed.")
        counts = sel["label_name"].value_counts().reindex(ORDER).dropna()
        fig = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h", width=0.36,
            marker_color=[COLORS[c] for c in counts.index],
            text=[f"{v:,.0f} ({v / counts.sum():.0%})" for v in counts.values], textposition="outside",
            textfont_color=INK, cliponaxis=False, hovertemplate="%{y}: %{x:,} links<extra></extra>",
        ))
        fig.update_layout(xaxis=dict(range=[0, counts.max() * 1.3], title="links"),
                          yaxis=dict(autorange="reversed", showgrid=False))
        show(fig, 290)

    with right, st.container(key="card-uni-trust"):
        chart_title("Trust score", "How many of the 7 trust signals each page has. Darker teal = more signals. "
                    "Most pages have almost none or most of them.")
        counts = sel["trust_score"].value_counts().reindex(range(8), fill_value=0).rename_axis("score").reset_index()
        fig = px.bar(counts, x="score", y="count", labels={"score": "trust score (0 to 7)", "count": "links"})
        fig.update_traces(marker_color=TEAL_RAMP, hovertemplate="score %{x}: %{y:,} links<extra></extra>")
        fig.update_layout(xaxis=dict(dtick=1, showgrid=False), bargap=0.45)
        show(fig, 290)

    with st.container(key="card-uni-dist"):
        chart_title("Distribution of a feature", "Page features have two humps: dead or tiny pages, and normal websites.")
        col, name, log = feature_picker("uni")
        values = np.log10(sel[col] + 1) if log else sel[col].clip(upper=sel[col].quantile(0.99))
        fig = px.histogram(x=values, nbins=60, color_discrete_sequence=[VIOLET], labels={"x": name.lower()})
        fig.update_traces(hovertemplate="%{y:,} links<extra></extra>")
        fig.update_layout(yaxis_title="links", bargap=0.06,
                          xaxis=dict(showgrid=False, **(log_axis(sel[col]) if log else {})))
        show(fig, 320)
        if not log:
            st.caption("The top 1% of values are cut off so the shape is visible.")


def bivariate_charts(sel, ending):
    overall = sel["is_phishing"].mean()
    left, right = st.columns(2, gap="medium")
    with left, st.container(key="card-bi-trust"):
        chart_title("Phishing rate by trust score", "The clearest pattern in the data: every extra trust signal "
                    "lowers the phishing rate.")
        rates = sel.groupby("trust_score")["is_phishing"].agg(rate="mean", n="size").reset_index()
        fig = rate_chart(rates, "trust_score", "rate", overall,
                         colors=[TEAL_RAMP[int(s)] for s in rates["trust_score"]])
        fig.update_layout(xaxis=dict(dtick=1, title="trust score (0 to 7)"))
        show(fig, 380)

    with right, st.container(key="card-bi-ending"):
        if ending == "All":
            chart_title("Phishing rate by ending", "Cheap and free endings are almost all phishing. "
                        "Showing the 12 most common endings.")
            rates = (sel.groupby("TLD")["is_phishing"].agg(rate="mean", n="size").nlargest(12, "n")
                     .sort_values("rate").reset_index())
            rates["TLD"] = "." + rates["TLD"]
            show(rate_chart(rates, "rate", "TLD", overall, horizontal=True), 380)
        else:
            chart_title(f"Phishing rate by page size for .{ending}", "Small pages are much more likely to be phishing.")
            rates = (sel.groupby("page_size_group")["is_phishing"].agg(rate="mean", n="size")
                     .reindex(PAGE_SIZES).dropna().reset_index())
            fig = rate_chart(rates, "page_size_group", "rate", overall)
            fig.update_layout(xaxis_title="page size (lines of code)")
            show(fig, 380)

    left, right = st.columns(2, gap="medium")
    with left, st.container(key="card-bi-box"):
        chart_title("A feature by class", "Real sites are bigger and have more of everything. The box holds the "
                    "middle half of the links, the line inside is the median.")
        col, name, log = feature_picker("bi")
        y = np.log10(sel[col] + 1) if log else sel[col]
        fig = px.box(sel.assign(y=y), x="label_name", y="y", color="label_name", color_discrete_map=COLORS,
                     category_orders={"label_name": ORDER}, points=False, labels={"label_name": "", "y": name.lower()})
        fig.update_layout(showlegend=False, xaxis_showgrid=False, yaxis=log_axis(sel[col]) if log else {})
        show(fig, 360)

    with right, st.container(key="card-bi-scatter"):
        chart_title("Page size vs links to other sites", "Phishing pages sit in the bottom left: small pages "
                    "with almost no links. Random sample of 2,500 links.")
        sample = sel.sample(min(2500, len(sel)), random_state=0)
        fig = px.scatter(sample, x=np.log10(sample["LineOfCode"] + 1), y=np.log10(sample["NoOfExternalRef"] + 1),
                         color="label_name", color_discrete_map=COLORS, category_orders={"label_name": ORDER},
                         opacity=0.55, custom_data=["URL", "LineOfCode", "NoOfExternalRef"],
                         labels={"x": "lines of code (log scale)", "y": "links to other sites (log scale)", "label_name": ""})
        fig.update_traces(marker=dict(size=7, line=dict(width=0.6, color="white")),
                          hovertemplate="%{customdata[0]}<br>%{customdata[1]:,} lines, "
                                        "%{customdata[2]:,} links<extra></extra>")
        fig.update_layout(xaxis=log_axis(sample["LineOfCode"]), yaxis=log_axis(sample["NoOfExternalRef"]))
        show(fig, 446)


def data_preview(sel):
    info = load_cleaning_summary()
    before, after = info["before"], info["after"]
    raw, steps, clean, stats, outliers = st.tabs(["Raw data", "Cleaning steps", "Clean data", "Summary statistics",
                                                  "Outliers"])

    with raw:
        st.caption("The first rows of the original file, before any cleaning.")
        st.dataframe(pd.DataFrame(info["raw_sample"]), hide_index=True)
        st.caption("In the raw file `label = 1` means legitimate and `0` means phishing. That's easy to misread, "
                   "so the clean data uses `is_phishing` instead.")

    with steps:
        c = st.columns(4)
        for col, (key, label) in zip(c, [("rows", "Rows"), ("columns", "Columns"), ("repeated URLs", "Repeated URLs"),
                                         ("missing values", "Missing values")]):
            col.metric(label, f"{after[key]:,}", delta=f"{after[key] - before[key]:+,} (was {before[key]:,})",
                       delta_color="off", delta_arrow="off", border=True)
        a, b, d = st.columns(3, gap="large")
        a.markdown("**Removed**")
        a.markdown(f"- {before['repeated URLs']} repeated URLs\n- `FILENAME`, just a file id\n"
                   "- `Title`, free text with broken characters\n- `label`, replaced by `is_phishing`")
        b.markdown("**Renamed (typos)**")
        b.markdown("\n".join(f"- `{old}` to `{new}`" for old, new in info["renamed"].items()))
        b.markdown("**Fixed types**\n- 0/1 columns stored as small integers\n- TLD in lower case")
        d.markdown("**Added**")
        d.markdown("\n".join(f"- `{x}`" for x in info["added"]))

    with clean:
        st.caption(f"The first 500 of {len(sel):,} links that match the filters. The last 7 columns are the "
                   "trust signals that add up to trust_score.")
        st.dataframe(sel.head(500), hide_index=True)

    with stats:
        num = ["URLLength", "domain_length", "LineOfCode", "NoOfImage", "NoOfJS", "NoOfCSS", "NoOfSelfRef",
               "NoOfExternalRef", "trust_score"]
        st.caption("count = how many links, mean = average, std = how spread out, 25% / 50% / 75% = the values "
                   "a quarter, half and three quarters of the links are below (50% is the median).")
        a, b = st.columns([3, 2], gap="large")
        a.markdown("**All filtered links**")
        a.dataframe(sel[num].describe().T.round(2))
        b.markdown("**Median by class**")
        b.dataframe(sel.groupby("label_name")[num].median().T)

    with outliers:
        outlier_list(sel)


def short_number(x):
    return f"{x:,.0f}" if abs(x) >= 10 else f"{x:.1f}".rstrip("0").rstrip(".")


def outlier_table(sel):
    """The IQR rule, as in the notebook: more than 1.5 times the middle half's width below or above it."""
    rows = []
    for col, name in OUTLIER_COLUMNS.items():
        values = sel[col]
        q1, q3 = values.quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        out = values[(values < low) | (values > high)]
        rows.append({"Column": name, "Outliers": len(out), "Share": 100 * len(out) / len(values),
                     "Normal range": f"{short_number(max(low, 0))} to {short_number(high)}", "Largest": values.max(),
                     "Phishing": 100 * sel.loc[out.index, "is_phishing"].mean() if len(out) else None})
    return pd.DataFrame(rows).sort_values("Share", ascending=False)


def outlier_list(sel):
    st.caption("An outlier is a value far outside the usual range. The rule is the same as in the notebook (IQR): take "
               "the middle half of the values, from the 25% point to the 75% point. Anything more than 1.5 times that "
               "width below or above it is an outlier. Worked out on the links that match the filters.")
    table = outlier_table(sel)
    left, right = st.columns([1, 2], gap="large")
    with left:
        fig = px.bar(table.sort_values("Share"), x="Share", y="Column", orientation="h",
                     color_discrete_sequence=[VIOLET], custom_data=["Outliers"])
        fig.update_traces(hovertemplate="%{y}: %{x:.1f}% of links (%{customdata[0]:,})<extra></extra>", width=0.6)
        fig.update_layout(xaxis=dict(ticksuffix="%", title="share of links that are outliers"),
                          yaxis=dict(title="", showgrid=False))
        show(fig, 360)
    with right:
        st.dataframe(table, hide_index=True, height=388, column_config={
            "Outliers": st.column_config.NumberColumn(format="localized"),
            "Share": st.column_config.NumberColumn("Share of links", format="%.1f%%"),
            "Normal range": st.column_config.TextColumn(help="Values inside this range are not outliers."),
            "Largest": st.column_config.NumberColumn("Largest value", format="localized"),
            "Phishing": st.column_config.NumberColumn(
                "Phishing", format="%.0f%%", help="How many of these outliers are phishing. Long links are almost "
                                                  "all phishing, while big pages are almost all real sites.")})

    a, b = st.columns(2, gap="large")
    for box, col, title, label in [(a, "URLLength", "The longest links", "Characters"),
                                   (b, "LineOfCode", "The biggest pages", "Lines of code")]:
        box.markdown(f"**{title}**")
        top = sel.nlargest(5, col)[["URL", col, "label_name"]].rename(columns={col: label, "label_name": "Class"})
        box.dataframe(top, hide_index=True, column_config={"URL": st.column_config.TextColumn(width=250),
                                                           label: st.column_config.NumberColumn(format="localized")})
    st.caption("These are real links, not typing mistakes, so they were kept. The charts above use a log scale or cut "
               "the top 1%, so a few giant values don't squash everything else.")


def model_results():
    try:
        checker = get_checker()
    except FileNotFoundError:
        st.info("The link model is missing. Run the model section of eda+model.ipynb to see the results here.")
        return
    report = checker.get("report")
    if report is None:
        st.info("Run eda+model.ipynb once more. It now saves the results this section shows.")
        return
    full, link = st.tabs(["All three models", "The link checker"])
    with full:
        full_model_results(report, checker["test_scores"])
    with link:
        link_checker_results(report, checker)


def full_model_results(report, link_scores):
    reads = {"Model A (with leaks)": f"{report['columns_used']['Model A (with leaks)']} columns, leaky ones included",
             "Model B (no leaks)": f"{report['columns_used']['Model B (no leaks)']} columns about the page and the name",
             "Link checker": "only the website name and its ending"}
    scores = {**report["full_models"], "Link checker": link_scores}
    table = pd.DataFrame([{"Model": name, "What it reads": reads[name], "Recall": 100 * s["recall (phishing)"],
                           "Precision": 100 * s["precision (phishing)"], "Accuracy": 100 * s["accuracy"],
                           "Missed phishing": int(s["missed phishing"]), "False alarms": int(s["false alarms"])}
                          for name, s in scores.items()])
    st.caption(f"All three were tested on the same {report['n_test']:,} links. Recall = share of phishing caught, "
               "precision = share of warnings that were right.")
    st.dataframe(table, hide_index=True, column_config={
        "What it reads": st.column_config.TextColumn(width="medium"),
        **{c: st.column_config.NumberColumn(format="%.2f%%") for c in ["Recall", "Precision", "Accuracy"]},
        **{c: st.column_config.NumberColumn(format="localized") for c in ["Missed phishing", "False alarms"]}})
    st.markdown("""
- **Model A** uses every column, including the 21 that give the answer away. Its perfect score only proves the leak.
- **Model B** leaves those out and is still right almost every time, but it needs facts about the page itself, like
  its lines of code and images. Someone holding a suspicious SMS doesn't have those.
- **The link checker** only reads the website name, so it's much weaker. That's why the app adds red-flag rules and
  lists of trusted and popular sites on top of it.
""")


def link_checker_results(report, checker):
    s, cm = checker["test_scores"], report["confusion"]
    cols = st.columns(4)
    for col, (label, value, note) in zip(cols, [
            ("Catches phishing", f"{s['recall (phishing)']:.0%}", "recall"),
            ("Right when it warns", f"{s['precision (phishing)']:.0%}", "precision"),
            ("Right overall", f"{s['accuracy']:.1%}", "accuracy"),
            ("Ranks phishing higher", f"{s['ROC-AUC']:.2f}", "ROC-AUC, 1 is perfect")]):
        col.metric(label, value, delta=note, delta_color="off", delta_arrow="off", border=True)

    left, right = st.columns(2, gap="large")
    with left:
        chart_title("What it got right and wrong", f"The {report['n_test']:,} test links, with a warning from a "
                    f"score of {checker['threshold']:.0%}.")
        cell = lambda kind, n, what: f'<td class="{kind}"><div class="n">{n:,}</div><div class="what">{what}</div></td>'
        st.html(f'<table class="cm"><tr><th></th><th>Warned</th><th>Not warned</th></tr>'
                f'<tr><th class="row">Really phishing</th>{cell("right", cm["caught"], "phishing caught")}'
                f'{cell("wrong", cm["missed"], "phishing missed")}</tr>'
                f'<tr><th class="row">Really legitimate</th>{cell("wrong", cm["false alarms"], "false alarms")}'
                f'{cell("right", cm["passed"], "real sites passed")}</tr></table>')
    with right:
        chart_title("Does the score mean what it says?", "Test links grouped by score. On the dashed line a score "
                    "of 40% would mean exactly 40% were phishing.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=AXIS, dash="dash"),
                                 name="score = real chance", hoverinfo="skip"))
        for key, name, color in [("before", "before calibration", "#9aa5b8"), ("after", "after calibration", VIOLET)]:
            points = pd.DataFrame(report["calibration_curve"][key])
            fig.add_trace(go.Scatter(x=points["average_score"], y=points["really_phishing"], mode="lines+markers",
                                     name=name, line=dict(color=color, width=2.5), marker=dict(size=7),
                                     customdata=points["links"],
                                     hovertemplate="score %{x:.0%}: %{y:.0%} really phishing<br>"
                                                   "%{customdata:,} links<extra></extra>"))
        fig.update_layout(xaxis=dict(range=[-0.02, 1.02], tickformat=".0%", title="score the model gave"),
                          yaxis=dict(range=[-0.02, 1.04], tickformat=".0%", title="really phishing"))
        show(fig, 330)

    chart_title("How the settings were picked", "On a validation part of the training websites. The test websites "
                "weren't used for any of this.")
    a, b = st.columns(2, gap="large")
    config = {"version": "Version", "C": st.column_config.NumberColumn("C", help="How much the model is held back "
                                                                          "from trusting single letter patterns"),
              "longest chunk": st.column_config.NumberColumn("Longest letter chunk"),
              "AUC": st.column_config.NumberColumn(format="%.4f"),
              "recall": st.column_config.NumberColumn("Recall at 10% false alarms", format="%.1f%%")}
    for box, title, rows in [(a, "Do the new name features help?", report["feature_check"]),
                             (b, "Which settings work best?", report["tuning"])]:
        box.markdown(f"**{title}**")
        table = pd.DataFrame(rows)
        table["recall"] = 100 * table.pop("recall at 10% false alarms")
        box.dataframe(table, hide_index=True, column_config=config)
    st.caption("The new name features and the settings search changed the results by less than one point. "
               "Calibration was the change that mattered: it made the score work like a real chance.")


# ---------------------------------------------------------------- page 2: signs of phishing
@st.cache_data
def dataset_facts():
    df = load_data()
    tld = df.groupby("TLD")["is_phishing"].agg(["mean", "size"]).query("size >= 1000").nlargest(5, "mean")
    lines = df.groupby("label_name")["LineOfCode"].median()
    trust = df.groupby("label_name")["trust_score"].median()
    return {"endings": list(tld.index), "low": tld["mean"].min(), "high": tld["mean"].max(),
            "lines_phish": lines["Phishing"], "lines_legit": lines["Legitimate"],
            "trust_phish": trust["Phishing"], "trust_legit": trust["Legitimate"]}


def signs_english(f):
    endings = ", ".join(f"`.{t}`" for t in f["endings"])
    return {
        "intro": "What to look for before you tap a link. The numbers in grey come from the links in the dataset.",
        "example": "Example:",
        "groups": {
            "In the link": [
                ("A strange or cheap ending.", f"Endings like {endings} cost little or nothing, so scammers love them.",
                 "`spl-sa-delivery.top/pay`",
                 f"In the dataset, {f['low']:.0%} to {f['high']:.0%} of links with these endings are phishing."),
                ("A brand name in the wrong place.", "Only the name right before the ending tells you the real site. "
                 "Everything in front of it can be made up.",
                 "`paypal.com.verify-account.xyz` is really verify-account.xyz, not PayPal.", None),
                ("Extra words around a brand.", "Real companies don't add words like secure, login, update or verify "
                 "to their name.", "`alrajhi-secure-login.com`, `amazon-support-help.net`", None),
                ("Numbers in place of letters.", "A zero for an `o` or a one for an `l` is easy to miss.",
                 "`amaz0n.com`, `paypa1.com`, `g00gle.com`", None),
                ("A free website builder.", "Anyone can make a site on these in a minute, with any name they like.",
                 "`absher-update.web.app`, `stc-gift.weebly.com`", None),
                ("A short link or a bare IP address.", "A short link hides where it really goes, and an IP address "
                 "has no name at all.", "`bit.ly/3xYz9`, `http://185.12.4.9/login`", None),
            ],
            "On the page": [
                ("A very small or empty page.", "Often a login box and nothing else.",
                 "a page with only a logo, two fields and a Sign in button.",
                 f"In the dataset, the median phishing page has {f['lines_phish']:.0f} lines of code. "
                 f"A real site has {f['lines_legit']:,.0f}."),
                ("No trust signals.", "No icon in the browser tab, no description, no copyright line and no social "
                 "media links.", "a bank page with no footer and no links to other pages.",
                 f"In the dataset, the median trust score is {f['trust_phish']:.0f} out of 7 for phishing pages "
                 f"and {f['trust_legit']:.0f} for real ones."),
                ("It asks for passwords, card numbers or codes.", "Banks and government services never ask for your PIN "
                 "or one-time code through a link.",
                 "\"Enter your card number and the code we sent you to confirm your identity.\"", None),
                ("The padlock is not proof.", "HTTPS only means the connection is encrypted. Anyone can get a free "
                 "certificate.", "`https://alrajhi-verify.top` has a padlock and is still fake.", None),
            ],
            "In the message": [
                ("Pressure to act now.", "Scammers rush you so you don't stop and think.",
                 "\"Your account will be suspended in 24 hours. Update your details now.\"", None),
                ("Too good to be true.", "Prizes, refunds and gifts you never applied for.",
                 "\"Congratulations, you won 5,000 SAR! Claim your prize here.\"", None),
                ("Delivery or bank messages you didn't expect.", "A parcel you didn't order, or a payment you didn't make.",
                 "\"Your parcel is held at customs. Pay 12 SAR to release it.\"", None),
            ],
        },
        "closing": "**Not sure? Don't tap the link.** Open the official app or type the address yourself. "
                   "In Saudi Arabia you can forward scam text messages to **330330**.",
    }


def signs_arabic(f):
    endings = "، ".join(f"`.{t}`" for t in f["endings"])
    return {
        "title": "علامات الرابط الاحتيالي",
        "intro": "ما الذي تنتبه له قبل أن تضغط على أي رابط. الأرقام باللون الرمادي مأخوذة من الروابط في مجموعة البيانات.",
        "example": "مثال:",
        "groups": {
            "في الرابط نفسه": [
                ("نهاية غريبة أو رخيصة.", f"نهايات مثل {endings} رخيصة أو مجانية، ولذلك يحبها المحتالون.",
                 "`spl-sa-delivery.top/pay`",
                 f"في البيانات، من {f['low']:.0%} إلى {f['high']:.0%} من الروابط بهذه النهايات احتيالية."),
                ("اسم شركة معروفة في غير مكانه.", "الاسم الذي يأتي قبل النهاية مباشرة هو الموقع الحقيقي، "
                 "وكل ما قبله يمكن أن يكتبه أي شخص.",
                 "`paypal.com.verify-account.xyz` موقعه الحقيقي هو `verify-account.xyz` وليس PayPal.", None),
                ("كلمات إضافية حول اسم الشركة.", "الشركات الحقيقية لا تضيف إلى اسمها كلمات مثل `secure` أو `login` "
                 "أو `update` أو `verify`.", "`alrajhi-secure-login.com` و `amazon-support-help.net`", None),
                ("أرقام مكان الحروف.", "صفر مكان حرف `o` أو رقم `1` مكان حرف `l`، ومن السهل ألا تلاحظها.",
                 "`amaz0n.com` و `paypa1.com` و `g00gle.com`", None),
                ("موقع على منصة مجانية لإنشاء المواقع.", "يستطيع أي شخص أن ينشئ موقعًا عليها خلال دقيقة وبأي اسم يريده.",
                 "`absher-update.web.app` و `stc-gift.weebly.com`", None),
                ("رابط مختصر أو عنوان IP.", "الرابط المختصر يخفي وجهته الحقيقية، وعنوان IP ليس له اسم أصلًا.",
                 "`bit.ly/3xYz9` و `http://185.12.4.9/login`", None),
            ],
            "في الصفحة": [
                ("صفحة صغيرة جدًا أو شبه فارغة.", "غالبًا مربع لتسجيل الدخول ولا شيء غيره.",
                 "صفحة فيها شعار وخانتان وزر «تسجيل الدخول» فقط.",
                 f"في البيانات، الوسيط لصفحة الاحتيال {f['lines_phish']:.0f} سطرًا برمجيًا، "
                 f"وللموقع الحقيقي {f['lines_legit']:,.0f} سطر."),
                ("لا توجد علامات ثقة.", "لا أيقونة في تبويب المتصفح، ولا وصف، ولا حقوق نشر، ولا روابط لحسابات "
                 "التواصل الاجتماعي.", "صفحة «بنك» بلا تذييل ولا روابط لصفحات أخرى.",
                 f"في البيانات، الوسيط لدرجة الثقة {f['trust_phish']:.0f} من 7 لصفحات الاحتيال، "
                 f"و{f['trust_legit']:.0f} من 7 للمواقع الحقيقية."),
                ("تطلب كلمة المرور أو رقم البطاقة أو رمز التحقق.", "البنوك والجهات الحكومية لا تطلب منك الرقم السري "
                 "أو رمز التحقق عن طريق رابط.", "«أدخل رقم بطاقتك والرمز المرسل إليك لتأكيد هويتك».", None),
                ("القفل ليس دليلًا على الأمان.", "HTTPS يعني فقط أن الاتصال مشفّر، وأي شخص يستطيع الحصول على شهادة مجانًا.",
                 "`https://alrajhi-verify.top` عليه قفل ومع ذلك هو موقع مزيّف.", None),
            ],
            "في الرسالة": [
                ("الضغط عليك لتتصرف فورًا.", "المحتال يستعجلك حتى لا تتوقف وتفكر.",
                 "«سيتم إيقاف حسابك خلال 24 ساعة. حدّث بياناتك الآن».", None),
                ("عرض أجمل من أن يكون حقيقيًا.", "جوائز أو مبالغ مستردة أو هدايا لم تطلبها.",
                 "«مبروك! ربحت 5,000 ريال. اضغط هنا لاستلام جائزتك».", None),
                ("رسائل توصيل أو بنك لم تكن تتوقعها.", "شحنة لم تطلبها، أو عملية دفع لم تقم بها.",
                 "«شحنتك محجوزة في الجمارك. ادفع 12 ريالًا للإفراج عنها».", None),
            ],
        },
        "closing": "**غير متأكد؟ لا تضغط على الرابط.** افتح التطبيق الرسمي أو اكتب العنوان بنفسك. "
                   "وفي السعودية يمكنك إعادة إرسال الرسائل الاحتيالية إلى الرقم **330330**.",
    }


def show_signs(text, lang):
    st.markdown(text["intro"])
    for (group, items), (key, color) in zip(text["groups"].items(),
                                            [("link", COLORS["Phishing"]), ("page", VIOLET), ("msg", TEAL)]):
        st.html(f'<div class="group"><div class="group-mark" style="background:{color}"></div>'
                f'<div class="group-title">{group}</div></div>')
        lines = []
        for title, explanation, example, stat in items:
            item = f"- **{title}** {explanation}  \n  :gray[{text['example']}] {example}"
            if stat:
                item += f"  \n  :gray[{stat}]"
            lines.append(item)
        with st.container(key=f"g-{key}-{lang}"):
            st.markdown("\n".join(lines))
    with st.container(key=f"closing-{lang}"):
        st.markdown(text["closing"])


def page_signs():
    facts = dataset_facts()
    with st.container(key="reading"):
        page_header("Signs of a phishing link", "What to look for before you tap a link, in English and Arabic.",
                    COLORS["Phishing"])
        english, arabic = st.tabs(["English", "العربية"])
        with english:
            show_signs(signs_english(facts), "en")
        with arabic, st.container(key="arabic"):
            text = signs_arabic(facts)
            st.subheader(text["title"])
            show_signs(text, "ar")


# ---------------------------------------------------------------- page 3: check a link
# colour, background tint, dark text colour, message
VERDICTS = {
    "Likely phishing": ("#d92d20", "#fdecea", "#912018", "Don't open this link. It has strong signs of phishing."),
    "Suspicious": ("#e8a127", "#fff5e0", "#8a5300", "Be careful. Some things about this link don't look right."),
    "Looks OK": ("#12a150", "#e8f7ee", "#0b6b35", "Nothing obvious is wrong, but no checker is perfect."),
    "Known popular website": ("#12a150", "#e8f7ee", "#0b6b35", "This is one of the 10,000 most visited websites."),
    "Trusted Saudi website": ("#12a150", "#e8f7ee", "#0b6b35", "This is an official Saudi government or university "
                              "site, or one of the main Saudi banks and services."),
}
SEVERITY = {"high": ":red-badge[High]", "medium": ":orange-badge[Medium]", "low": ":gray-badge[Low]"}


def verdict_card(verdict, extra=""):
    color, bg, ink, message = VERDICTS[verdict]
    st.html(f'<div class="verdict" style="--c:{color};--bg:{bg};--ink:{ink}"><div class="verdict-title">{verdict}</div>'
            f'<div class="verdict-text">{message}{extra}</div></div>')


def link_anatomy(url, registered):
    """The pasted link with the real website highlighted and everything else greyed out."""
    if registered in FREE_HOSTS | OPEN_PLATFORMS:
        who = f"On {registered} anyone can make their own address, so the part in front says nothing about who made the page."
    else:
        who = f"Only the owner of {registered} can choose what comes in front of it."
    netloc = urlsplit(url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url).netloc
    i = url.find(netloc)
    j = netloc.lower().rfind(registered)
    if i < 0 or j < 0:      # odd input: just show it as it is
        before, real, after = "", url, ""
    else:
        before = url[:i] + netloc[:j]
        real = netloc[j:j + len(registered)]
        after = netloc[j + len(registered):] + url[i + len(netloc):]
    st.html(f'<div class="anatomy"><span class="rest">{html.escape(before)}</span>'
            f'<span class="real">{html.escape(real)}</span><span class="rest">{html.escape(after)}</span></div>'
            f'<div class="anatomy-note">The highlighted part is the website this link really opens. {html.escape(who)}</div>')


def score_meter(score, threshold):
    """The model score on a green / amber / red track. Amber starts 0.15 below the threshold, like the verdicts."""
    sus, thr = (threshold - 0.15) * 100, threshold * 100
    track = (f"linear-gradient(90deg, #4cc38a 0%, #4cc38a {sus:.0f}%, #f2b84b {sus:.0f}%, #f2b84b {thr:.0f}%, "
             f"#ef6b5b {thr:.0f}%, #d92d20 100%)")
    st.html(f'<div class="meter-head"><span>Model phishing score</span><strong>{score:.0%}</strong></div>'
            f'<div class="meter-track" style="background:{track}"><div class="meter-dot" style="left:{score:.1%}"></div></div>'
            f'<div class="meter-scale"><span style="width:{sus:.0f}%">looks OK</span>'
            f'<span style="width:{thr - sus:.0f}%">suspicious</span><span style="width:{100 - thr:.0f}%">'
            f'phishing (from {threshold:.0%})</span></div>')


def model_view(url, checker):
    """The text the model reads and the letter patterns that moved its score."""
    st.markdown(f"The model reads only the website name and its ending: `{link_text(url)}`. It learned from "
                f"{checker.get('n_train', 188295):,} links which letter patterns show up in phishing names and which "
                "in real ones. Each pattern below pushed this link's score towards one side.")
    pushes = explain_score(url, checker)
    rows = pd.DataFrame([p for p in pushes if p[1] > 0][:4] + [p for p in pushes if p[1] < 0][-4:],
                        columns=["pattern", "push"]).sort_values("push")
    if not rows.empty:
        rows["side"] = np.where(rows["push"] > 0, "towards phishing", "towards real")
        rows["label"] = "'" + rows["pattern"] + "'"
        fig = px.bar(rows, x="push", y="label", orientation="h", color="side",
                     color_discrete_map={"towards phishing": COLORS["Phishing"], "towards real": COLORS["Legitimate"]},
                     category_orders={"side": ["towards phishing", "towards real"]},
                     labels={"push": "push on the score", "label": ""})
        fig.update_traces(hovertemplate="%{y}: %{x:+.2f}<extra></extra>", width=0.55)
        fig.update_layout(yaxis=dict(showgrid=False, categoryorder="array", categoryarray=list(rows["label"])),
                          xaxis_zeroline=True, xaxis_zerolinecolor=AXIS, legend_title_text="")
        show(fig, 60 + 34 * len(rows))
    st.caption("The greyed-out parts of the link never reach the model. In the training data every real link was a "
               "bare homepage, so a model reading those parts would just learn that rule and fail on real links. "
               "The red-flag rules check those parts instead.")


def checker_or_error():
    try:
        return get_checker()
    except FileNotFoundError:
        st.error("The link model is missing. Run the model section of eda+model.ipynb, it saves models/link_model.joblib.")
        return None


def page_checker():
    with st.container(key="reading"):
        page_header("Check a link", "Paste a link from a message or an email. The link is never opened, only its "
                    "text is read.", VIOLET)
        with st.form("check", border=False):
            a, b = st.columns([5, 1], vertical_alignment="bottom")
            url = a.text_input("Link", label_visibility="collapsed", placeholder="Paste a link").strip()
            b.form_submit_button("Check", type="primary", width="stretch")

        if not url:
            return
        if "." not in split_link(url)["host"]:
            st.error("That doesn't look like a link. Check that you pasted the whole address.")
            return
        checker = checker_or_error()
        if checker is None:
            return

        r = check_link(url, checker)
        verdict_card(r["verdict"])
        link_anatomy(url, r["registered"])
        score_meter(r["score"], r["threshold"])

        st.subheader("Why")
        reasons = []
        if r["trusted"]:
            reasons.append(f"{r['registered']} is on the checker's short list of official Saudi sites: government "
                           "(.gov.sa), universities (.edu.sa) and the main banks.")
        if r["popular"]:
            reasons.append(f"{r['registered']} is on the list of the 10,000 most visited websites.")
        if r["platform"]:
            reasons.append(f"The model's {r['score']:.0%} only describes the name {r['registered']}, not the page, "
                           "so it isn't used here.")
        else:
            reasons.append(f"The model gives this name a {r['score']:.0%} phishing score.")
        reasons += [f"{SEVERITY[sev]} {text}" for sev, text in r["flags"]]
        st.markdown("\n".join(f"- {x}" for x in reasons))
        if (r["popular"] or r["trusted"]) and r["score"] >= r["threshold"]:
            st.caption("The model alone would flag this name, because thousands of phishing links copy famous brand "
                       "names. That's why trusted and popular sites are checked against a list first.")

        with st.expander("What the model looked at"):
            model_view(url, checker)

        with st.expander("How good is this checker?"):
            s = checker["test_scores"]
            st.markdown(f"""
Tested on websites it never saw during training, it:
- catches **{s['recall (phishing)']:.0%}** of phishing links (recall)
- is right **{s['precision (phishing)']:.0%}** of the time when it warns (precision)

It only reads the link, so it can't spot a real website that was hacked, and no checker is right every time.
When in doubt, don't open the link.
""")
            if "calibration_check" in checker:
                st.markdown("The score is calibrated, so it works like a chance. This is how the scores compared "
                            "with reality on the test websites:")
                check = pd.DataFrame(checker["calibration_check"])
                check[["average score", "really phishing"]] *= 100
                st.dataframe(check, hide_index=True, column_config={
                    "score": "Score given", "links": st.column_config.NumberColumn("Links", format="localized"),
                    "average score": st.column_config.NumberColumn("Average score", format="%.0f%%"),
                    "really phishing": st.column_config.NumberColumn("Really phishing", format="%.0f%%")})
                st.caption("That holds for this dataset, where 43% of the links are phishing. In a normal inbox "
                           "phishing is much rarer, so the same score means a lower chance there.")


# ---------------------------------------------------------------- app
st.set_page_config(page_title="Phishing URL Dashboard", layout="wide")
st.html(CSS)
st.navigation([
    st.Page(page_dashboard, title="Dashboard", default=True),
    st.Page(page_signs, title="Signs of phishing", url_path="signs"),
    st.Page(page_checker, title="Check a link", url_path="check"),
], position="top").run()
