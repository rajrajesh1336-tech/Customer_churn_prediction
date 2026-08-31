# Telco Customer Churn Prediction

An end-to-end customer churn classification project using the Telco Customer Churn dataset.

## Workflow

**Data → Cleaning → EDA → Preprocessing → Model Training → Hyperparameter Tuning → Model Comparison → Model Saving → Streamlit Deployment**

The project uses the existing `Model_Training.ipynb` workflow and keeps preprocessing inside the saved scikit-learn pipeline.

## Model Training

`Model_Training.ipynb`:

- loads `cleaned_data.csv`
- removes rows with a missing target
- separates `Churn` from the features
- maps `Churn`: `Yes → 1`, `No → 0`
- converts `SeniorCitizen`: `1 → Yes`, `0 → No`
- identifies numeric and categorical features
- uses median imputation + standard scaling for numeric features
- uses most-frequent imputation + one-hot encoding for categorical features
- tunes Logistic Regression, Random Forest, SVM, AdaBoost and XGBoost using `GridSearchCV`
- evaluates Accuracy, Precision, Recall, F1 Score and ROC-AUC
- sorts `result_df` by F1 Score descending
- selects the first row as the best model
- saves the complete preprocessing + model pipeline

### Important fixes made

The notebook contained two concrete XGBoost issues:

1. `eval_metrics` was corrected to `eval_metric`.
2. The XGBoost `max_depth` grid entry was missing the pipeline prefix and was corrected to `model__max_depth`.

The Windows dataset path was also made a raw string to avoid backslash path issues.

## Feature Schema

The cleaned dataset contains these model inputs:

| Feature | Training dtype |
|---|---|
| gender | object |
| SeniorCitizen | object (`Yes` / `No`) |
| Partner | object |
| Dependents | object |
| tenure | int64 |
| PhoneService | object |
| MultipleLines | object |
| InternetService | object |
| OnlineSecurity | object |
| OnlineBackup | object |
| DeviceProtection | object |
| TechSupport | object |
| StreamingTV | object |
| StreamingMovies | object |
| Contract | object |
| PaperlessBilling | object |
| PaymentMethod | object |
| MonthlyCharges | float64 |
| TotalCharges | float64 |

`Churn` is the target and is not entered in the Streamlit app.

`customerID` is not a model feature.

`Unnamed: 0`, if present in an exported CSV index, is not a model feature.

## Model Output

The notebook saves the best model to:

```text
C:\Victor\Customer_Churn_Prediction\best_model.pkl
```

The Streamlit app expects a copy of the trained model at:

```text
best_model.pkl
```

in the same directory as `app.py`.

The saved object is the complete pipeline, so the app does not manually reproduce scaling or one-hot encoding.

## Run Locally

### 1. Create a Python 3.11 environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Open:

```text
Model_Training_completed.ipynb
```

Run the notebook cells in order. The final cell sorts `result_df`, selects the best model, and saves the pipeline.

### 4. Place the model beside the app

Copy:

```text
best_model.pkl
```

into the project root beside `app.py`.

### 5. Run Streamlit

```bash
streamlit run app.py
```

## Project Structure

```text
Customer_Churn_Prediction/
│
├── Model_Training_completed.ipynb
├── app.py
├── best_model.pkl
├── requirements.txt
├── .gitignore
├── README.md
│
└── data/
    └── cleaned_data.csv
```

The dataset files are intentionally ignored by Git.

## GitHub

Before pushing:

```bash
git init
git add .
git status
```

Confirm that the CSV files and `*.pkl` files are not staged.

Then:

```bash
git commit -m "Add customer churn prediction project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Deployment

For Streamlit deployment, keep `app.py`, `requirements.txt`, and `best_model.pkl` in the repository if the deployment platform needs the model artifact.

If you choose not to commit the model artifact, the app will need an alternative model-storage mechanism. The current `.gitignore` intentionally prevents accidental commits of model/data artifacts.

## Notes

The Streamlit input form deliberately uses `SeniorCitizen` as `Yes/No`, because the training notebook converts the original CSV's `int64` 0/1 representation into categorical values before fitting the preprocessing pipeline.

The app also validates feature names and the expected semantic data types before prediction to reduce training/serving feature drift.
