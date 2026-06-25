# Flight Delay Prediction

A machine learning project that predicts whether a flight will experience a departure delay of **15 minutes or more** using historical U.S. airline operations data from the Bureau of Transportation Statistics (BTS).

The project focuses on building a realistic pre-departure prediction system using only information available before a flight takes off, while carefully avoiding data leakage.

---

## Problem Statement

Flight delays affect passengers, airlines, and airports through missed connections, increased operational costs, and schedule disruptions.

The goal of this project is to predict:

**Will a flight be delayed by at least 15 minutes at departure?**

This is formulated as a binary classification problem:

* **0** → Not delayed (< 15 minutes)
* **1** → Delayed (≥ 15 minutes)

Target variable:

```text
DepDel15
```

---

## Dataset

Source:

* U.S. Department of Transportation (DOT)
* Bureau of Transportation Statistics (BTS) On-Time Performance Dataset

The dataset contains historical flight records including:

* Airline
* Origin airport
* Destination airport
* Scheduled departure time
* Flight distance
* Date information
* Delay indicators

Only information available before departure is used for prediction.

---

## Project Workflow

### 1. Data Collection

Downloaded BTS flight performance data and converted it into a machine-learning-friendly format.

### 2. Data Cleaning

* Removed unusable records
* Standardized categorical variables
* Handled missing values
* Created train/validation/test splits

### 3. Exploratory Data Analysis (EDA)

Studied relationships between delay probability and:

* Departure hour
* Day of week
* Month
* Airline
* Airport
* Distance

Key observations:

* Delay probability generally increases throughout the day.
* Evening departures experience higher delay rates.
* Different airlines and airports exhibit distinct delay patterns.
* Distance alone has relatively weak predictive power.

### 4. Feature Engineering

Time Features:

* Departure hour
* Month
* Day of week
* Weekend indicator
* Part of day

Categorical Features:

* Airline
* Origin airport
* Destination airport

Numerical Features:

* Distance

Historical Delay Features:

* Smoothed airline delay rate
* Smoothed origin airport delay rate
* Smoothed destination airport delay rate

### 5. Leakage Prevention

The project explicitly avoids data leakage.

Excluded features include:

* Actual departure delay
* Arrival delay information
* Delay attribution variables such as WeatherDelay, NASDelay, CarrierDelay, etc.

All target-encoding mappings are computed using the training set only and then applied to validation and test sets.

Unseen categories are assigned the global training-set delay rate.

---

## Model Development

### Baseline Model

Initial baseline models were trained using simple temporal and operational features.

### Final Model

Model used:

```text
XGBoost Classifier
```

Reasons for choosing XGBoost:

* Handles mixed feature types well
* Captures non-linear relationships
* Robust on tabular data
* Strong performance with class imbalance

Techniques used:

* Early stopping
* Target encoding
* Hyperparameter tuning
* Class imbalance handling through scale_pos_weight

---

## Evaluation Metric

Because only around 18% of flights are delayed, accuracy alone is misleading.

Primary metric:

```text
PR-AUC (Precision-Recall AUC)
```

Why PR-AUC?

* Better suited for imbalanced datasets
* Measures performance on the positive (delayed) class
* More informative than accuracy

Baseline PR-AUC:

```text
0.1438
```

Final Validation PR-AUC:

```text
0.223
```

This represents a substantial improvement over the no-skill baseline.

---

## Model Performance

The final model successfully learns operational delay patterns from historical flight data.

Key strengths:

* Uses only pre-departure information
* Leakage-free feature engineering
* Interpretable historical delay features
* Generalizable deployment pipeline

Limitations:

* No real-time weather data
* No aircraft rotation information
* No air traffic control congestion information
* Limited to historical operational patterns

---

## Streamlit Web Application

A simple web interface was developed using Streamlit.

Users can enter:

* Airline
* Origin airport
* Destination airport
* Departure time
* Date information

The model returns:

* Delay prediction
* Delay probability

The application demonstrates how the trained model can be deployed for real-world usage.

---

## Tech Stack

Python

Libraries:

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## Project Structure

```text
flight-delay-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Modeling.ipynb
│
├── models/
│   └── xgb_model.joblib
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
│
└── README.md
```

---

## Future Improvements

* Integrate historical weather data
* Add flight route congestion features
* Incorporate aircraft-level information
* Perform automated hyperparameter optimization
* Deploy on cloud infrastructure

---

## Author

Hani Sohaan Shaik
BITS Pilani Hyderabad Campus
Machine Learning Project