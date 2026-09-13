# Customer Churn Predictor & Retention Dashboard

End-to-end churn prediction: a scikit-learn pipeline trained on the IBM Telco customer
dataset, served by a FastAPI backend, and surfaced in a Streamlit dashboard that ranks
at-risk accounts by **expected revenue loss** (churn probability × monthly bill) and
simulates retention offers.

```
[Telco customer CSV]
        ↓
[pandas cleaning + scikit-learn pipeline]  ──  train.py (3 candidates, CV model selection)
        ↓
[FastAPI]  POST /predict · POST /predict/batch · GET /model/info
        ↓
[Streamlit dashboard]  live predictions · retention levers · revenue-at-risk leaderboard
```

## Results

Trained on 5,634 customers, evaluated on a held-out 1,409 (stratified 80/20 split).
Threshold tuned on out-of-fold predictions only, so holdout numbers are honest.

| Model | CV ROC-AUC (5-fold) | Holdout ROC-AUC | Holdout accuracy @0.5 |
|---|---|---|---|
| **Logistic regression (selected)** | **0.846 ± 0.013** | **0.842** | **80.3%** |
| Random forest | 0.843 ± 0.009 | 0.839 | 80.0% |
| Histogram gradient boosting | 0.843 ± 0.009 | 0.838 | 79.8% |

At the tuned operating threshold (0.327, chosen to maximise F1):

| Metric | Value |
|---|---|
| Recall (churners caught) | 72.2% |
| Precision | 53.8% |
| F1 | 0.616 |
| Accuracy | 76.2% (the dashboard reports this number; 80.3% above is at the untuned 0.5 cutoff) |
| Customers flagged for outreach | 35.6% |
| **Monthly revenue of actual churners that the model flags in time** | **76.1%** ($20.7k of $27.2k on the holdout) |

Top churn drivers by permutation importance: tenure, contract type, internet service
type, total/monthly charges, online security, tech support, payment method.

## Quick start

Requires **Python 3.12 or newer** (the pinned numpy needs 3.12; stock macOS `python3` is 3.9).
If `python3` on your PATH is older, run `make setup PYTHON=python3.12`.

```bash
git clone <your-repo-url> churn-predictor && cd churn-predictor
make setup        # checks the Python version, then python3 -m venv .venv && pip install -r requirements.txt
make data         # downloads data/telco_churn.csv (7,043 rows)
make train        # ~30 s: trains 3 models, writes models/churn_pipeline.joblib + metrics.json
make api          # http://127.0.0.1:8000/docs  (terminal 1)
make dashboard    # http://localhost:8501         (terminal 2)
make test         # unit + API + headless dashboard tests
```

If you use [uv](https://github.com/astral-sh/uv): `uv venv .venv && uv pip install -r requirements.txt`.

## The API

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5
  }'
```

```json
{
  "churn_probability": 0.78,
  "will_churn": true,
  "risk_tier": "High",
  "monthly_charges": 89.5,
  "monthly_revenue_at_risk": 69.81,
  "annual_revenue_at_risk": 837.72,
  "threshold": 0.3271,
  "model_name": "logistic_regression"
}
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one customer. Field names match the dataset columns. `TotalCharges` is optional. |
| `POST /predict/batch` | Score up to 20,000 customers in one call, results in request order. |
| `GET /model/info` | Training metrics, confusion matrix, feature importance, business impact. |
| `GET /health` | Liveness check plus the loaded model name and threshold. |

Validation is strict: unknown fields, out-of-range numbers, unknown categories and
inconsistent combinations (e.g. `TechSupport: "Yes"` with `InternetService: "No"`)
all return `422` with a message explaining why.

## The dashboard

1. **Predict a customer.** Sliders and dropdowns for all 19 attributes; one click queries
   the API and shows a gauge, risk tier and revenue at risk. Below it, **retention levers**
   re-score the same customer under each plausible offer (longer contract, add tech
   support, automatic payment, loyalty discount) and rank them by revenue protected.
2. **Retention risk board.** Scores the whole customer base through `/predict/batch`,
   then ranks accounts by `churn_probability × MonthlyCharges`. Filters by tier,
   contract and bill size; shows total monthly and annualised revenue at risk, how
   concentrated the risk is in the top N, and a downloadable leaderboard.
3. **Model performance.** Holdout metrics, candidate comparison, confusion matrix,
   permutation importance and the revenue-capture numbers above.

## Project layout

```
churn/features.py     column lists, allowed category values, cleaning  (single source of truth)
churn/scoring.py      risk tiers and revenue-at-risk formulas
train.py              model selection, threshold tuning, metrics.json export
api/schemas.py        Pydantic request/response models (strict validation)
api/main.py           FastAPI app; loads the pipeline once at startup
dashboard/app.py      Streamlit UI; talks to the API over HTTP only
tests/                schema/cleaning tests, API tests, headless dashboard integration test
models/               churn_pipeline.joblib + metrics.json (produced by train.py)
```

## Design decisions worth knowing

- **One pipeline object.** Preprocessing (scaling, one-hot encoding with a fixed category
  list) lives inside the exported sklearn `Pipeline`, so the API never re-implements
  feature engineering and cannot drift from training.
- **Model selection by cross-validation, threshold by out-of-fold predictions.** The
  holdout set is touched exactly once, for the reported numbers.
- **Business metric, not just AUC.** "Revenue capture rate" answers the question a
  retention team asks: of the money that walked out the door, how much would we have
  flagged in time?
- **Schema tests guard the contract.** A test asserts the API's `Literal` types equal the
  encoder's category lists, and another pushes every dataset row through the schema.

## Caveats (say these out loud in an interview)

- The retention board scores the same customers the model was trained on, so 80% of
  those scores are in-sample. In production you would score customers the model has
  never seen. The holdout metrics on the model tab are the honest numbers.
- Retention levers are *what-if* re-scores, not causal estimates. A model that has
  learned "two-year contracts churn less" cannot tell you that *moving* someone to a
  two-year contract causes the reduction.
- The dataset is a single snapshot with no timestamps, so there is no time-based
  validation and no drift monitoring.

## Resume bullets

> **Machine Learning & API Developer | Customer Retention Dashboard**
> - Built an end-to-end churn pipeline with Python, pandas and scikit-learn (3 candidate
>   models, 5-fold CV selection, out-of-fold threshold tuning) reaching **0.84 ROC-AUC**
>   on a held-out set of 1,409 telecom customers and catching **72% of churners** at the
>   recall-tuned operating point the service uses (80% accuracy at the default 0.5 cutoff).
> - Designed a FastAPI service with strict Pydantic validation serving single and batch
>   predictions (7,000 customers scored in one call) with business-ready outputs:
>   risk tier and revenue at risk.
> - Shipped a Streamlit executive dashboard that ranks accounts by expected revenue
>   loss and simulates retention offers; at the tuned threshold the model flags 36% of
>   customers while capturing **76% of churning monthly revenue**.
