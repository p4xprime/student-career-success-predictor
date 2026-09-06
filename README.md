# Student Career Success Prediction

A Machine Learning web application that predicts whether a student will be **Placed** or **Not Placed** based on their academic profile, technical skills, and soft skills.

---

## Dataset

**Source:** [Kaggle – Student Career Success Prediction Dataset](https://www.kaggle.com/datasets/mobeenfatimah/student-career-success-prediction-dataset)  
50,000 student records with 29 features including CGPA, internships, programming skill, interview score, and more.

---

## ML Pipeline

| Step | Detail |
|------|--------|
| Target | `Placement_Status` (Placed / Not Placed) |
| Models compared | Logistic Regression, Random Forest, Extra Trees, Gradient Boosting |
| Best model | **Random Forest** (CV F1 ≈ 79.95%) |
| Test Accuracy | **79.4%** |
| Test ROC-AUC | **0.7964** |

---

## Local Setup

### 1. Clone / download the project

```bash
git clone https://github.com/<your-username>/student-career-predictor.git
cd student-career-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This saves `models/model.pkl` and `models/model_meta.json`.

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy on Streamlit Community Cloud

1. Push the project to a **public GitHub repository**.
2. Make sure `data/dataset.csv` and `models/model.pkl` are committed (or retrain on startup — see note below).
3. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
4. Select your repository, branch (`main`), and main file (`app.py`).
5. Click **Deploy**.

> **Note:** If `models/model.pkl` is too large for GitHub (>100 MB), add a startup `train_model.py` call at the top of `app.py`, or use Git LFS.

---

## Project Structure

```
project/
├── app.py                  # Streamlit web app
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md
├── agent_instructions.md
├── .env.example
├── data/
│   └── dataset.csv         # Student Career Success dataset
└── models/
    ├── model.pkl           # Saved Random Forest pipeline
    └── model_meta.json     # Feature names, class labels, metrics
```

---

## Environment Variables

No API keys are required. See `.env.example` for any optional configuration.
