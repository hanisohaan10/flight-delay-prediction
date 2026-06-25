# Flight Delay Prediction — Project Notes

## 1. Project Goal

Predict whether a flight will be **delayed by more than 15 minutes at departure**.
This is a **binary classification** problem.

**Target variable:**
```
DepDel15 = 1   if DepDelay > 15 minutes
DepDel15 = 0   otherwise
```

`DepDel15` already exists in the BTS dataset, so no manual target construction is required.

**Prediction constraint:** The model makes its prediction **2 hours before the scheduled departure time.** Only information available at that point may be used as a feature. Any other information will be considered leaky.

### What Is Data Leakage?

Data leakage occurs when a model is trained on information that would not be available at prediction time. A leaky model achieves unrealistically high accuracy during development but fails in production, because the "future" information it relied on does not exist when a real prediction must be made.

For flight delay prediction this risk is severe: the BTS dataset is recorded *after* each flight completes, so the majority of its columns describe events that happen during or after departure. These must be identified and removed.

---
## 2. Prediction-Time Boundary

**The model predicts 2 hours before scheduled departure.**

| Category   | Available at Prediction Time                                                              | NOT Available                                                                              |
| ---------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Schedule   | Airline, Origin, Dest, CRSDepTime, CRSArrTime, CRSElapsedTime, Distance, DayOfWeek, Month | DepTime, ArrTime, ActualElapsedTime                                                        |
| Operations | (none)                                                                                    | TaxiOut, WheelsOff, WheelsOn, TaxiIn, AirTime, Cancelled, Diverted                         |
| Delays     | (none direct)                                                                             | DepDelay, ArrDelay, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay |
| Weather    | Historical delay patterns by (Airport, Month, Hour)                                       | Real-time weather, WeatherDelay                                                            |

---

## 3. Leaky Features — REMOVE

These columns encode information generated during or after the flight. Including any of them would leak the answer into the model.

| Feature                   | Columns                                                                          | Reason                                |
| ------------------------- | -------------------------------------------------------------------------------- | ------------------------------------- |
| Actual departure time     | `DepTime`                                                                        | Reveals whether the flight left late  |
| Departure delay           | `DepDelay`, `DepDelayMinutes`, `DepartureDelayGroups`                            | Directly defines the target           |
| Arrival delay             | `ArrDelay`, `ArrDelayMinutes`, `ArrDel15`, `ArrivalDelayGroups`                  | Occurs after departure                |
| Actual arrival time       | `ArrTime`                                                                        | Known only after the flight lands     |
| Actual elapsed / air time | `ActualElapsedTime`, `AirTime`                                                   | Known only after the flight completes |
| Taxi durations            | `TaxiOut`, `TaxiIn`                                                              | Measured during ground operations     |
| Wheels off / on           | `WheelsOff`, `WheelsOn`                                                          | Generated during flight operations    |
| Delay-cause breakdown     | `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay` | Attributed after the flight           |
| Gate-time additions       | `FirstDepTime`, `TotalAddGTime`, `LongestAddGTime`                               | Recorded during operations            |
| Cancellation / diversion  | `Cancelled`, `CancellationCode`, `Diverted`, all `Div*` columns                  | Known only at/after the event         |

---

## 4. Safe Features — KEEP

All sourced from the published flight schedule and therefore available 2 hours before departure.

| Feature                  | Column                            | Rationale                                      |
| ------------------------ | --------------------------------- | ---------------------------------------------- |
| Airline                  | `Reporting_Airline`               | Carriers have different historical delay rates |
| Origin airport           | `Origin`                          | Congestion varies by airport                   |
| Destination airport      | `Dest`                            | Destination traffic affects operations         |
| Scheduled departure time | `CRSDepTime`                      | Time-of-day is a strong delay signal           |
| Scheduled arrival time   | `CRSArrTime`                      | From the flight plan                           |
| Scheduled elapsed time   | `CRSElapsedTime`                  | From the flight plan                           |
| Distance                 | `Distance`                        | From the flight plan (miles)                   |
| Day of week              | `DayOfWeek`                       | Weekday vs weekend traffic                     |
| Month                    | `Month`                           | Seasonality                                    |
| Aircraft                 | `Tail_Number`                     | Safe if taken from the schedule                |
| Flight number            | `Flight_Number_Reporting_Airline` | Route identifier                               |

---

### 5. Weather Information and Leakage Considerations

The BTS on-time performance dataset does not contain pre-departure weather variables such as temperature, precipitation, wind speed, visibility, or cloud cover. The only weather-related field available is `WeatherDelay`, which is assigned after the flight and directly reflects the realized cause of delay.

**Decision:** `WeatherDelay` and other post-flight delay attribution fields were excluded from feature engineering.

**Reason:**

- These variables are unavailable at prediction time.
- Using them would leak information about the actual outcome into the model.
- The objective is to predict delays using only information known before departure.

**Limitation:**

- The model cannot directly account for real-time weather conditions.
- Some weather effects may be indirectly captured through variables such as airport, month, day of week, airline, and scheduled departure time, which reflect recurring seasonal and operational patterns.

---

## 6. Train / Validation / Test Split (Temporal Integrity)

Flight data is time-ordered, so the split must be **chronological**, not random.
```python
# CORRECT: chronological split
Train:      Aug 2024 – Oct 2024
Validation: Nov 2024
Test:       Dec 2024
```

```python
# WRONG: random shuffle
train_test_split(..., shuffle=True)
# Leaks future flights into training and past flights into testing.
```

**Baseline computation rule:** all historical delay baselines and aggregated statistics (e.g., airport- or airline-level delay rates) were computed using the **training period only** and then applied to validation and test sets. This prevents future information from leaking into feature engineering and ensures that every prediction uses only information that would have been available at the time of prediction.

---

## 7. Leakage Detection Experiment

A controlled comparison to confirm leakage handling is correct.

**Model A — Leaky baseline (sanity check, not for deployment)**

```
Features: DepDelay, ArrDelay, ActualElapsedTime, TaxiOut
Expected accuracy: > 95% (unrealistically high)
```

**Model B — Leak-free (deployable)**

```
Features: Reporting_Airline, Origin, Dest, CRSDepTime (dep_hour),
          DayOfWeek, Month, Distance, origin_delay_baseline, dest_delay_baseline
Expected accuracy: ~ 65–75% (realistic)
```

The large accuracy gap demonstrates how much "performance" came purely from leakage — and confirms Model B is the honest, deployable system.

---
## 9. Key Principle

We must ensure that every feature would genuinely be available 2 hours before departure.

A model with slightly lower accuracy and **zero leakage** is far more valuable than a high-accuracy model trained on information from the future. This project is an exercise in realistic deployment thinking and disciplined handling of temporal data.

---
