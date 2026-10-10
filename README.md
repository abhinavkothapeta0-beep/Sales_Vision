# SalesIQ – AI-Powered Sales Forecasting

This is a Streamlit dashboard for the internship project **Sales Forecasting Using Machine Learning**.

## Important

The uploaded notebook contains the model training code, but it does not contain a separately saved `.joblib` model artifact. Therefore the dashboard will **not invent predictions**.

Run `train_model.py` once to reproduce the Random Forest training logic from the notebook and save:

- `models/rf_model.joblib`
- `models/feature_columns.json`

After that, the Streamlit app can make real predictions without retraining on every page load.

## Project setup

1. Put the Kaggle `train.csv` file here:

```text
SalesIQ_Streamlit/data/train.csv
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create the saved model artifacts:

```bash
python train_model.py
```

This is a one-time training/export step. It reproduces the notebook's feature engineering, `get_dummies(..., drop_first=True)`, chronological 80/20 split, and Random Forest parameters.

5. Start the website:

```bash
streamlit run app.py
```

## Expected project structure

```text
SalesIQ_Streamlit/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── train.csv
└── models/
    ├── rf_model.joblib
    └── feature_columns.json
```

## Dataset

The project uses the Kaggle Store Sales time-series dataset. The notebook loads `../data/train.csv` and works with about 3,000,888 rows and the columns:

`id, date, store_nbr, family, sales, onpromotion`

## Model

Random Forest:

- n_estimators = 50
- max_depth = 10
- random_state = 42
- n_jobs = -1

Reported evaluation:

- Linear Regression: RMSE 888.50, R² 0.5741
- Random Forest: RMSE 486.33, R² 0.8724

MAPE is not used as a primary dashboard metric because the dataset contains zero-sales observations.

## Notes

For a large dataset, the first load may take time and memory. The app caches the dataset and model. The Data Explorer only displays up to 500 rows at a time.
