# flight-delay-prediction

# Flight Delay Prediction

Predicting whether a flight will be delayed using historical airline operational data from the U.S. Bureau of Transportation Statistics (BTS).

## Project Overview

Flight delays impact millions of passengers each year and create significant operational costs for airlines and airports. This project aims to build a machine learning pipeline that predicts flight delays before departure using information that would realistically be available at prediction time.

A key focus of this project is avoiding **data leakage** by ensuring that only pre-departure features are used during model training and evaluation.

## Objectives

* Explore and clean historical flight performance data.
* Perform exploratory data analysis (EDA) to identify delay patterns.
* Engineer meaningful features from scheduling, airport, and temporal information.
* Train and compare multiple machine learning models.
* Evaluate models using appropriate classification metrics.
* Deploy a reproducible prediction pipeline.

## Dataset

Data source:

* U.S. Bureau of Transportation Statistics (BTS) On-Time Performance Dataset
* Alternative source: Kaggle Flight Delays Dataset

### Target Variable

Example binary classification target:
is_delayed = ARR_DELAY > 15
A flight is considered delayed if it arrives more than 15 minutes after its scheduled arrival time.