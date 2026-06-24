# Flight Delay Predictor — Web App

A Streamlit web app that predicts the probability a flight **departs more than
15 minutes late** (`DepDel15`), using only information available ~2 hours before
scheduled departure. No leaky, post-departure features are used.

## Run locally

From the **project root** (not from inside `app/`):

```bash
# 1. (optional) create a clean environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r app/requirements.txt

# 3. launch
streamlit run app/app.py
```

Streamlit prints a local URL (default http://localhost:8501). Open it in a browser.

## How it works

1. You pick airline, origin, destination, day of week, and departure time.
2. The destination list is restricted to **real routes** in the data, so the
   route's `Distance` is looked up automatically (users don't know mileage).
3. Inputs are turned into the 6 model features — `DayOfWeek`, `dep_hour`,
   `Distance`, and smoothed historical delay rates for the airline / origin /
   destination — exactly as in `notebooks/features.py::transform_features`.
4. The XGBoost model returns a delay **probability**, shown as a percentage and
   a base-rate-anchored risk band (Low / Typical / Elevated / High).

## Files

```
app/
├── app.py                  # the Streamlit application
├── requirements.txt        # pinned dependencies
├── artifacts/
│   ├── xgb_final.json      # model, re-exported for version-stable loading
│   ├── xgb_final.pkl       # original pickle (kept as backup)
│   ├── encoders.pkl        # target-encoding lookups + global mean
│   └── feature_columns.pkl # model feature order
└── reference/
    ├── route_distance.json # (origin-dest) -> miles  (6,550 routes)
    ├── origin_dests.json   # origin -> valid destinations
    ├── airport_city.json   # IATA code -> city name  (346 airports)
    └── airlines.json       # carrier code -> airline name (15 carriers)
```

The `reference/*.json` files are derived from the training data so the app is
fully self-contained — no parquet files or notebooks are needed at runtime.

## Honest performance note

On the held-out test set: **ROC-AUC ≈ 0.65, PR-AUC ≈ 0.33** (base delay rate
≈ 0.18). This is a modest screening signal, not a precise forecast. At a naive
0.5 threshold the model almost never fires (delays are the minority class), so
the app reports a **probability and a risk band** rather than a yes/no label.
