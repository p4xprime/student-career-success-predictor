"""
app.py  —  Student Career Success Prediction
Streamlit web application with dark theme.
Loads the trained Random Forest pipeline from models/model.pkl
and predicts Placement_Status from user-supplied feature values.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Student Career Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark-theme custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global background & text ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d;
}
/* headings */
h1, h2, h3, h4, h5, h6 { color: #58a6ff !important; }
/* labels */
label, .stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label { color: #c9d1d9 !important; }
/* inputs */
input, textarea, .stNumberInput input {
    background-color: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
/* selectbox */
[data-baseweb="select"] > div {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}
/* radio */
.stRadio > div { color: #c9d1d9 !important; }
/* slider track */
[data-testid="stSlider"] .st-ae { background: #58a6ff !important; }
/* metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
/* buttons */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 1rem;
    padding: 0.55rem 2rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043 !important; }
/* divider */
hr { border-color: #30363d !important; }
/* expander */
details { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px; }
/* dataframe */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
/* success / error / info */
.stAlert { border-radius: 8px !important; }
/* sidebar text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #c9d1d9 !important; }
/* progress bar */
[data-testid="stProgress"] > div > div { background: #58a6ff !important; }
/* section header band */
.section-band {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 6px 14px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.json")


@st.cache_resource(show_spinner="Loading model …")
def load_model():
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pipe, meta


# ─── Load ─────────────────────────────────────────────────────────────────────
model, meta = None, {}
try:
    model, meta = load_model()
except FileNotFoundError:
    st.error(
        "**Model not found.** "
        "Please run `python train_model.py` first to train and save the model."
    )
    st.stop()

CLASSES             = meta["classes"]
NOMINAL_FEATURES    = meta["nominal_features"]
ORDINAL_FEATURES    = meta["ordinal_features"]
ORDINAL_CATS        = {k: v for k, v in zip(ORDINAL_FEATURES, meta["ordinal_categories"])}
BINARY_FEATURES     = meta["binary_features"]
NUMERIC_FEATURES    = meta["numeric_features"]
NOMINAL_OHE_COLS    = meta["nominal_ohe_cols"]
ALL_FEATURE_COLS    = meta["feature_cols"]
DECISION_THRESHOLD  = meta.get("decision_threshold", 0.5)


def build_input_row(raw_inputs: dict) -> pd.DataFrame:
    """Convert widget values into a one-row DataFrame matching training columns."""
    # Compute the engineered composite feature before building the row
    raw_inputs = dict(raw_inputs)  # shallow copy — don't mutate caller's dict
    raw_inputs["Readiness_Score"] = (
        raw_inputs["Resume_Score"]
        + raw_inputs["Interview_Score"]
        + raw_inputs["Programming_Skill"]
        + raw_inputs["Problem_Solving"]
        + raw_inputs["Projects_Completed"]
        + raw_inputs["Internships"]
    ) / 6.0

    row = {}

    # Numeric: direct passthrough
    for col in NUMERIC_FEATURES:
        row[col] = raw_inputs[col]

    # Binary Yes/No -> 1/0
    for col in BINARY_FEATURES:
        row[col] = 1 if raw_inputs[col] == "Yes" else 0

    # Ordinal: pass raw string label (OrdinalEncoder handles it)
    for col in ORDINAL_FEATURES:
        row[col] = raw_inputs[col]

    # Nominal OHE columns
    for col in NOMINAL_OHE_COLS:
        row[col] = 0  # default off

    # Set the selected nominal value to 1
    for nom in NOMINAL_FEATURES:
        ohe_col = f"{nom}_{raw_inputs[nom]}"
        if ohe_col in row:
            row[ohe_col] = 1

    df = pd.DataFrame([row])
    return df.loc[:, ALL_FEATURE_COLS]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 About")
    st.markdown("""
**Dataset:** Student Career Success  
**Target:** Placement Status  
**Problem:** Binary Classification  
""")
    st.divider()
    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{meta['test_accuracy']*100:.1f}%")
    col2.metric("ROC-AUC",  f"{meta['test_auc']:.4f}" if meta["test_auc"] else "N/A")
    st.metric("F1 Score (weighted)", f"{meta['test_f1']*100:.1f}%")
    st.divider()
    st.markdown("### Model Comparison")
    cv_data = []
    for name, scores in meta["cv_results"].items():
        cv_data.append({
            "Model": name,
            "CV F1": f"{scores['cv_f1_mean']*100:.2f}%",
            "Std":   f"±{scores['cv_f1_std']*100:.2f}%",
        })
    cv_df = pd.DataFrame(cv_data)
    st.dataframe(cv_df, hide_index=True, use_container_width=True)
    st.caption(f"Best model: **{meta['model_name']}**")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("Student Career Success Predictor")
st.markdown(
    "Fill in your academic and skill profile below. "
    "The model will predict whether you are likely to be **Placed** or **Not Placed**."
)
st.divider()

# ── Section 1: Academic Profile ──────────────────────────────────────────────
st.markdown('<div class="section-band"><b>📚 Academic Profile</b></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
cgpa             = c1.number_input("CGPA",              min_value=2.0, max_value=10.0, value=3.1, step=0.01, format="%.2f")
attendance       = c2.number_input("Attendance (%)",    min_value=50, max_value=100, value=82, step=1)
study_hours      = c3.number_input("Study Hours / Week", min_value=5, max_value=120, value=21, step=1)

c4, c5, c6 = st.columns(3)
academic_perf    = c4.selectbox("Academic Performance", ORDINAL_CATS["Academic_Performance"])
major            = c5.selectbox("Major",              [
    "Computer Science", "Software Engineering", "Artificial Intelligence",
    "Data Science", "Cybersecurity", "Information Technology",
    "Business Analytics", "Electrical Engineering",
])

st.divider()

# ── Section 2: Technical Skills ───────────────────────────────────────────────
st.markdown('<div class="section-band"><b>💻 Technical Skills & Activities</b></div>', unsafe_allow_html=True)
c8, c9, c10, c11 = st.columns(4)
programming_skill  = c8.slider("Programming Skill (1-10)",  1, 10, 7)
projects_completed = c9.slider("Projects Completed",         0, 15, 8)
certifications     = c10.slider("Certifications",            0, 8,  2)
hackathons         = c11.slider("Hackathons",               0, 10, 3)

c12, c13, c14 = st.columns(3)
internships  = c12.slider("Internships",     0, 5, 4)
resume_score = c13.slider("Resume Score",   44, 100, 96)
interview_sc = c14.slider("Interview Score", 17, 100, 78)

st.divider()

# ── Section 3: Soft Skills ────────────────────────────────────────────────────
st.markdown('<div class="section-band"><b>🤝 Soft Skills</b></div>', unsafe_allow_html=True)
c15, c16, c17 = st.columns(3)
communication  = c15.slider("Communication Skills (1-10)", 3, 10, 8)
teamwork       = c16.slider("Teamwork (1-10)",              2, 10, 7)
problem_solving = c17.slider("Problem Solving (1-10)",     1, 10, 7)

c18, c19 = st.columns(2)
english_prof = c18.selectbox("English Proficiency", ORDINAL_CATS["English_Proficiency"])
gender       = c19.selectbox("Gender",              ["Male", "Female", "Other"])

st.divider()

# ── Section 4: Profiles ───────────────────────────────────────────────────────
st.markdown('<div class="section-band"><b>🔗 Online Presence & Leadership</b></div>', unsafe_allow_html=True)
c20, c21 = st.columns(2)
github_profile       = c20.radio("GitHub Profile",       ["Yes", "No"], horizontal=True)
leadership_exp       = c21.radio("Leadership Experience", ["Yes", "No"], horizontal=True)

st.divider()

# ─── Predict ──────────────────────────────────────────────────────────────────
predict_btn = st.button("Predict Placement", use_container_width=False)

if predict_btn:
    raw_inputs = {
        "Attendance_Percentage": attendance,
        "Study_Hours_Per_Week": study_hours,
        "CGPA":                 cgpa,
        "Programming_Skill":    programming_skill,
        "Projects_Completed":   projects_completed,
        "Certifications":       certifications,
        "Hackathons":           hackathons,
        "Internships":          internships,
        "Resume_Score":         resume_score,
        "Communication_Skills": communication,
        "Teamwork":             teamwork,
        "Problem_Solving":      problem_solving,
        "Interview_Score":      interview_sc,
        "GitHub_Profile":       github_profile,
        "Leadership_Experience": leadership_exp,
        "Academic_Performance": academic_perf,
        "English_Proficiency":  english_prof,
        "Gender":               gender,
        "Major":                major,
    }

    input_df   = build_input_row(raw_inputs)
    pred_proba = model.predict_proba(input_df)[0]

    placed_prob    = pred_proba[CLASSES.index("Placed")]
    notplaced_prob = pred_proba[CLASSES.index("Not Placed")]

    # Use the tuned threshold instead of the default 0.5
    pred_label = "Placed" if placed_prob >= DECISION_THRESHOLD else "Not Placed"

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred_label == "Placed":
        st.success(f"### ✅ Prediction: **{pred_label}**")
    else:
        st.error(f"### ❌ Prediction: **{pred_label}**")

    col_a, col_b = st.columns(2)
    col_a.metric("Placed Probability",     f"{placed_prob*100:.1f}%")
    col_b.metric("Not Placed Probability", f"{notplaced_prob*100:.1f}%")

    # Confidence bar
    st.markdown("**Confidence**")
    st.progress(float(max(placed_prob, notplaced_prob)))
    st.caption(
        f"Model confidence: **{max(placed_prob, notplaced_prob)*100:.1f}%**  |  "
        f"Model: {meta['model_name']}  |  "
        f"Accuracy: {meta['test_accuracy']*100:.1f}%"
    )

    # Feature summary
    with st.expander("View submitted feature values"):
        display_df = pd.DataFrame([{
            "Feature": k, "Value": v
        } for k, v in raw_inputs.items()])
        st.dataframe(display_df, hide_index=True, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Student Career Success Prediction | "
    "Dataset: mobeenfatimah/student-career-success-prediction-dataset (Kaggle) | "
    "Model: Random Forest Classifier"
)
