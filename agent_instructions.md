# Agent Instructions

## Project Purpose
This project trains a machine learning classifier on the Student Career Success dataset
and serves predictions via a Streamlit web app.

## Key Files
- `train_model.py`  — loads data, preprocesses, trains 4 ML models, saves the best pipeline
- `app.py`          — Streamlit UI that loads the saved model and serves real-time predictions
- `data/dataset.csv` — raw dataset (50 000 rows, 29 columns)
- `models/model.pkl` — serialised sklearn Pipeline (Random Forest + ColumnTransformer)
- `models/model_meta.json` — feature lists, class labels, evaluation metrics

## How to Retrain
```bash
python train_model.py
```

## How to Run the App
```bash
streamlit run app.py
```

## Deployment Notes
- No API keys required.
- All paths are relative; works on any OS.
- Compatible with Streamlit Community Cloud and Render.
- If `model.pkl` exceeds GitHub's 100 MB file limit, use Git LFS or regenerate it on first run.

## Adding a New Model
1. Import the estimator in `train_model.py`.
2. Add it to the `CANDIDATES` dict.
3. Re-run `python train_model.py` — the best model is automatically selected and saved.
