"""
Flight Delay Predictor - Streamlit app.

Predicts the probability that a flight departs more than 15 minutes late
(BTS target `DepDel15`), using only information available ~2 hours before
scheduled departure. No post-departure (leaky) features are used.

Run from the project root:
    python -m streamlit run app/app.py
"""

import json
import pickle
import pathlib
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
BASE = pathlib.Path(__file__).resolve().parent
ART = BASE / "artifacts"
REF = BASE / "reference"

BASE_RATE = 0.18  # ~18% of flights depart >15 min late (training data)

# BTS convention: 1 = Monday ... 7 = Sunday
DAYS = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}

# Palette (from the cream + olive reference)
CREAM = "#F5F0DF"
CREAM_2 = "#FBF8EE"
OLIVE = "#586A2C"
OLIVE_DK = "#3E4C1E"
INK = "#33401A"

# Earthy risk ramp - harmonises with cream/olive instead of clashing RGB
BANDS = [
    (0.15, "Low",      "#5B6E2E", "Below the typical flight."),
    (0.25, "Typical",  "#7C7A33", "About average for U.S. flights."),
    (0.40, "Elevated", "#B07A2B", "Higher than the average flight."),
    (1.01, "High",     "#9E4324", "Well above the average flight."),
]

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(ART / "xgb_final.json"))
    return model


@st.cache_resource
def load_refs():
    with open(ART / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(ART / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(REF / "route_distance.json") as f:
        routes = json.load(f)
    with open(REF / "airport_city.json") as f:
        cities = json.load(f)
    with open(REF / "origin_dests.json") as f:
        origin_dests = json.load(f)
    with open(REF / "airlines.json") as f:
        airlines = json.load(f)
    return encoders, feature_columns, routes, cities, origin_dests, airlines


def build_features(day_of_week, dep_hour, distance, airline, origin, dest,
                   encoders, feature_columns):
    gm = encoders["global_mean"]
    row = {
        "DayOfWeek": day_of_week,
        "dep_hour": dep_hour,
        "Distance": distance,
        "airline_delay_rate": encoders["airline_delay_rate"].get(airline, gm),
        "origin_delay_rate": encoders["origin_delay_rate"].get(origin, gm),
        "dest_delay_rate": encoders["dest_delay_rate"].get(dest, gm),
    }
    return pd.DataFrame([row])[feature_columns]


def risk_band(p):
    for hi, name, color, note in BANDS:
        if p < hi:
            return name, color, note
    return BANDS[-1][1], BANDS[-1][2], BANDS[-1][3]


def fmt_hour(h):
    suffix = "AM" if h < 12 else "PM"
    return f"{h:02d}:00  ({h % 12 or 12} {suffix})"


def hour_short(h):
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def meter_html(p):
    pct = max(0.0, min(p * 100, 100.0))
    base = BASE_RATE * 100
    return f"""
    <div style="margin:18px 0 2px;">
      <div style="position:relative; height:16px; border-radius:9px;
                  background:rgba(255,255,255,.22);">
        <div style="position:absolute; top:0; bottom:0; left:0; width:{pct:.1f}%;
                    background:{CREAM_2}; border-radius:9px;"></div>
        <div style="position:absolute; top:-5px; bottom:-5px; left:{base:.1f}%;
                    width:2px; background:rgba(255,255,255,.9);"></div>
      </div>
      <div style="position:relative; height:16px; font-size:11px;
                  color:rgba(255,255,255,.85); margin-top:3px;">
        <span style="position:absolute; left:0;">0%</span>
        <span style="position:absolute; left:{base:.1f}%; transform:translateX(-50%);">avg ~{base:.0f}%</span>
        <span style="position:absolute; right:0;">100%</span>
      </div>
    </div>"""


def explain_contributions(model, X):
    booster = model.get_booster()
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dm, pred_contribs=True)[0]
    return list(X.columns), contribs[:len(X.columns)]


def driver_bars(items):
    m = max((abs(v) for _, v in items), default=1.0) or 1.0
    POS, NEG = "#B07A2B", OLIVE
    html = "<div>"
    for label, v in items:
        w = abs(v) / m * 46.0
        cue = "raises risk" if v > 0 else "lowers risk"
        if v >= 0:
            bar = (f'<div style="position:absolute; left:50%; width:{w:.0f}%; '
                   f'top:0; bottom:0; background:{POS}; border-radius:0 5px 5px 0;"></div>')
        else:
            bar = (f'<div style="position:absolute; right:50%; width:{w:.0f}%; '
                   f'top:0; bottom:0; background:{NEG}; border-radius:5px 0 0 5px;"></div>')
        html += f"""
        <div style="margin:9px 0;">
          <div style="display:flex; justify-content:space-between; font-size:12.5px;
                      color:#5b5a42; margin-bottom:3px;">
            <span><b style="color:{INK};">{label}</b></span>
            <span>{cue}</span>
          </div>
          <div style="position:relative; height:10px; background:#E7E0C6; border-radius:5px;">
            <div style="position:absolute; left:50%; top:-2px; bottom:-2px; width:1px;
                        background:#B8B089;"></div>
            {bar}
          </div>
        </div>"""
    return html + "</div>"


st.set_page_config(page_title="Flight Delay Predictor", page_icon=":airplane:",
                   layout="centered")

model = load_model()
encoders, feature_columns, routes, cities, origin_dests, airlines = load_refs()

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"], .stApp, button, input, select {{
        font-family: 'Poppins', system-ui, sans-serif;
      }}
      .stApp {{ max-width: 880px; margin: 0 auto; }}

      section[data-testid="stSidebar"] {{ background-color: {OLIVE_DK}; }}
      section[data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
      section[data-testid="stSidebar"] small {{ color: #D7DDBE !important; }}

      .hero-eyebrow {{ letter-spacing:.16em; text-transform:uppercase;
        font-size:12px; font-weight:600; color:{OLIVE}; margin-bottom:2px; }}
      .hero-title {{ font-size:46px; font-weight:800; line-height:1.02;
        color:{INK}; margin:0 0 8px; }}
      .hero-sub {{ font-size:16px; color:#5d5c43; margin:0 0 6px; max-width:62ch; }}
      .rule {{ height:4px; width:64px; background:{OLIVE}; border-radius:3px;
        margin:14px 0 6px; }}

      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background:{CREAM_2}; border:1px solid #E2DBC0 !important;
        border-radius:16px; padding:6px 4px;
        box-shadow:0 1px 0 rgba(0,0,0,.02);
      }}
      .card-h {{ font-size:13px; font-weight:600; letter-spacing:.04em;
        text-transform:uppercase; color:{OLIVE}; margin:2px 2px 0; }}

      .stButton > button {{
        background:{OLIVE}; color:{CREAM}; border:none; border-radius:12px;
        padding:.7rem 1rem; font-weight:600; font-size:15px; width:100%;
        transition:background .15s ease;
      }}
      .stButton > button:hover {{ background:{OLIVE_DK}; color:#fff; }}
      .stButton > button:focus {{ box-shadow:0 0 0 3px rgba(88,106,44,.3); }}

      label, .stSelectbox label {{ color:{INK} !important; font-weight:500; }}
      .muted {{ color:#7a7960; font-size:12.5px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-eyebrow">Pre-departure &middot; 2-hour window</div>
    <div class="hero-title">Flight Delay Predictor</div>
    <div class="hero-sub">The probability a flight <b>departs more than 15 minutes
    late</b> - estimated from the published schedule alone, with no leaked
    post-departure data.</div>
    <div class="rule"></div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About this model")
    st.markdown(
        "**Predicts** &nbsp;P(departure delay > 15 min)\n\n"
        "**Inputs** &nbsp;schedule only - leakage-free\n\n"
        "**Test ROC-AUC** &nbsp;0.65\n\n"
        "**Test PR-AUC** &nbsp;0.33\n\n"
        "**Base delay rate** &nbsp;~18%"
    )
    st.caption(
        "A screening signal, not a guarantee. Read the output as how far above "
        "or below average a flight's risk is - not a definitive yes / no."
    )

airline_items = sorted(airlines.items(), key=lambda kv: kv[1])
airline_codes = [c for c, _ in airline_items]
airline_labels = [f"{n} ({c})" for c, n in airline_items]


def airport_label(code):
    city = cities.get(code, "")
    return f"{code} - {city}" if city else code


origins = sorted(origin_dests.keys())

with st.container(border=True):
    st.markdown('<div class="card-h">Flight details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        airline_idx = st.selectbox("Airline", range(len(airline_codes)),
                                   format_func=lambda i: airline_labels[i])
        airline = airline_codes[airline_idx]
        origin = st.selectbox("Origin airport", origins,
                              index=origins.index("ATL") if "ATL" in origins else 0,
                              format_func=airport_label)
        valid_dests = origin_dests.get(origin, [])
        dest = st.selectbox("Destination airport", valid_dests,
                            format_func=airport_label)
    with c2:
        day_name = st.selectbox("Day of week", list(DAYS.keys()))
        day_of_week = DAYS[day_name]
        dep_hour = st.selectbox("Scheduled departure time", list(range(24)),
                                index=8, format_func=fmt_hour)

    distance = routes.get(f"{origin}-{dest}")
    if distance is not None:
        st.markdown(
            f'<span class="muted">Route distance (from schedule): '
            f'<b style="color:{INK};">{distance:,} miles</b></span>',
            unsafe_allow_html=True,
        )

    predict = st.button("Predict delay risk", type="primary")

if predict:
    if distance is None:
        st.error("No distance on record for this route. Pick another destination.")
    else:
        X = build_features(day_of_week, dep_hour, distance, airline,
                           origin, dest, encoders, feature_columns)
        p = float(model.predict_proba(X)[:, 1][0])
        band, color, note = risk_band(p)
        rel = p / BASE_RATE

        st.markdown(
            f"""
            <div style="background:{color}; color:{CREAM_2}; border-radius:18px;
                        padding:26px 30px; margin-top:18px;">
              <div style="font-size:13px; letter-spacing:.08em; text-transform:uppercase;
                          opacity:.85;">Estimated delay risk</div>
              <div style="display:flex; align-items:baseline; gap:14px; margin-top:2px;">
                <div style="font-size:60px; font-weight:800; line-height:1;">{p*100:.0f}%</div>
                <div style="font-size:20px; font-weight:600;">{band} &middot; {rel:.1f}x average</div>
              </div>
              <div style="font-size:14px; opacity:.92; margin-top:8px;">{note}
                The average U.S. flight has a ~{BASE_RATE*100:.0f}% chance of a
                15-minute departure delay.</div>
              {meter_html(p)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown('<div class="card-h">What\'s driving this</div>',
                        unsafe_allow_html=True)
            feats, vals = explain_contributions(model, X)
            label_map = {
                "DayOfWeek": day_name,
                "dep_hour": f"{hour_short(dep_hour)} departure",
                "Distance": f"{distance:,} mi flight",
                "airline_delay_rate": f"{airlines[airline]} history",
                "origin_delay_rate": f"{origin} origin",
                "dest_delay_rate": f"{dest} destination",
            }
            items = sorted(
                [(label_map.get(f, f), float(v)) for f, v in zip(feats, vals)],
                key=lambda kv: abs(kv[1]), reverse=True,
            )
            st.markdown(driver_bars(items), unsafe_allow_html=True)
            st.markdown(
                '<div class="muted" style="margin-top:8px;">Each bar is this '
                "detail's pull on <b>this</b> prediction (model contributions): "
                '<span style="color:#B07A2B; font-weight:600;">ochre raises</span> '
                f'risk, <span style="color:{OLIVE}; font-weight:600;">olive lowers</span> it.</div>',
                unsafe_allow_html=True,
            )

st.markdown(
    "<p class='muted' style='margin-top:18px;'>Trained on U.S. BTS On-Time "
    "Performance data (Aug-Dec 2024). Predicts departure delay - not arrival "
    "or cancellation.</p>",
    unsafe_allow_html=True,
)
