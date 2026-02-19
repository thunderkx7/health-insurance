"""
app.py  —  Streamlit prediction app.
Loads model.pkl + meta.pkl. No training at runtime.

    streamlit run app.py
"""

import pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Insurance Predictor", page_icon="🏥", layout="centered")

st.markdown("""
<style>
/* ── global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f6;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"]  { display: none; }
[data-testid="stSidebar"] { display: none; }
#MainMenu, footer         { display: none; }

/* ── single centered card ── */
.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 44px 48px 40px;
    box-shadow: 0 4px 24px rgba(0,0,0,.08);
    max-width: 740px;
    margin: 56px auto 0;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 4px;
}
.page-sub {
    color: #6b7280;
    font-size: .9rem;
    margin-bottom: 32px;
}

/* ── field labels ── */
.field-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: #374151;
    margin-bottom: 4px;
}
.row-gap { margin-bottom: 18px; }

/* ── widget overrides ── */
div[data-testid="stNumberInput"] > div,
div[data-testid="stSelectbox"]   > div > div {
    background: #f9fafb !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-size: .93rem !important;
    color: #111827 !important;
}
div[data-testid="stNumberInput"] input {
    background: transparent !important;
    color: #111827 !important;
}
div[data-testid="stNumberInput"] > div:focus-within,
div[data-testid="stSelectbox"]   > div > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
}

/* ── button ── */
div[data-testid="stButton"] > button {
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 13px 0;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    margin-top: 12px;
    transition: background .18s;
}
div[data-testid="stButton"] > button:hover { background: #4f46e5; }

/* ── result — only the charge ── */
.result {
    margin-top: 32px;
    padding-top: 28px;
    border-top: 1.5px solid #f0f0f0;
    text-align: center;
}
.result-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 10px;
}
.result-amount {
    font-size: 3.6rem;
    font-weight: 800;
    color: #6366f1;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────
@st.cache_resource
def load():
    with open("model.pkl", "rb") as f: model = pickle.load(f)
    with open("meta.pkl",  "rb") as f: meta  = pickle.load(f)
    return model, meta

try:
    model, meta = load()
except FileNotFoundError:
    st.error("❌ `model.pkl` / `meta.pkl` not found. Run `python train.py` first.")
    st.stop()


# ── Predict ───────────────────────────────────────────────────────
def predict(user_input: dict) -> float:
    row = pd.DataFrame([user_input])

    if "age" in meta["num_cols"] and "bmi" in meta["num_cols"]:
        row["_age_bmi"] = row["age"] * row["bmi"]
        row["_age_sq"]  = row["age"] ** 2
    if "bmi" in meta["num_cols"]:
        row["_obese"] = (row["bmi"] >= 30).astype(int)

    for col, lvls in meta["cat_opts"].items():
        val = user_input[col]
        if len(lvls) == 2:
            row[f"_enc_{col}"] = int(val != lvls[0])
        else:
            for lv in lvls[1:]:
                row[f"_ohe_{col}_{lv}"] = int(val == lv)

    if "_enc_smoker" in row.columns and "bmi" in row.columns:
        row["_smoker_bmi"] = row["_enc_smoker"] * row["bmi"]

    row.drop(columns=meta["cat_cols"], inplace=True, errors="ignore")
    for c in meta["feat_cols"]:
        if c not in row.columns:
            row[c] = 0
    row = row[meta["feat_cols"]]
    return float(np.expm1(model.predict(row)[0]))


# ── UI ────────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<div class='page-title'>🏥 Insurance Cost Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Enter your details to estimate your annual insurance charge.</div>",
            unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
user_input  = {}

with left:
    for col in meta["num_cols"]:
        s = meta["df_stats"][col]
        st.markdown(f"<div class='field-label'>{col.replace('_',' ').title()}</div>",
                    unsafe_allow_html=True)
        is_int = col in ("age", "children") or (s["max"] - s["min"]) < 20
        if is_int:
            user_input[col] = st.number_input(
                "", min_value=int(s["min"]), max_value=int(s["max"]),
                value=int(s["med"]), step=1,
                key=f"n_{col}", label_visibility="collapsed")
        else:
            user_input[col] = st.number_input(
                "", min_value=round(s["min"],1), max_value=round(s["max"],1),
                value=round(s["med"],1), step=0.1,
                key=f"n_{col}", label_visibility="collapsed")
        st.markdown("<div class='row-gap'></div>", unsafe_allow_html=True)

with right:
    for col in meta["cat_cols"]:
        opts    = meta["cat_opts"][col]
        display = [o.title() for o in opts]
        st.markdown(f"<div class='field-label'>{col.replace('_',' ').title()}</div>",
                    unsafe_allow_html=True)
        chosen = st.selectbox("", display, key=f"c_{col}", label_visibility="collapsed")
        user_input[col] = opts[display.index(chosen)]
        st.markdown("<div class='row-gap'></div>", unsafe_allow_html=True)

if st.button("Predict Charge"):
    charge = predict(user_input)
    st.markdown(f"""
    <div class='result'>
        <div class='result-label'>Estimated Annual Charge</div>
        <div class='result-amount'>${charge:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
