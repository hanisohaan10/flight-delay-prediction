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

**Prediction constraint:** The model makes its prediction **2 hours before the scheduled departure time.** Only information available at that point may be used as a feature.

---

## 2. What Is Data Leakage?

Data leakage occurs when a model is trained on information that would not be available at prediction time. A leaky model achieves unrealistically high accuracy during development but fails in production, because the "future" information it relied on does not exist when a real prediction must be made.

For flight delay prediction this risk is severe: the BTS dataset is recorded *after* each flight completes, so the majority of its columns describe events that happen during or after departure. These must be identified and removed.

---

## 3. Prediction-Time Boundary

**The model predicts 2 hours before scheduled departure.**

| Category | Available at Prediction Time | NOT Available |
|----------|------------------------------|---------------|
| Schedule | Airline, Origin, Dest, CRSDepTime, CRSArrTime, CRSElapsedTime, Distance, DayOfWeek, Month | DepTime, ArrTime, ActualElapsedTime |
| Operations | (none) | TaxiOut, WheelsOff, WheelsOn, TaxiIn, AirTime, Cancelled, Diverted |
| Delays | (none direct) | DepDelay, ArrDelay, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay |
| Weather | Historical delay patterns by (Airport, Month, Hour) | Real-time weather, WeatherDelay |

---

## 4. Leaky Features — REMOVE

These columns encode information generated during or after the flight. Including any of them would leak the answer into the model.

| Feature | Columns | Reason |
|---------|---------|--------|
| Actual departure time | `DepTime` | Reveals whether the flight left late |
| Departure delay | `DepDelay`, `DepDelayMinutes`, `DepartureDelayGroups` | Directly defines the target |
| Arrival delay | `ArrDelay`, `ArrDelayMinutes`, `ArrDel15`, `ArrivalDelayGroups` | Occurs after departure |
| Actual arrival time | `ArrTime` | Known only after the flight lands |
| Actual elapsed / air time | `ActualElapsedTime`, `AirTime` | Known only after the flight completes |
| Taxi durations | `TaxiOut`, `TaxiIn` | Measured during ground operations |
| Wheels off / on | `WheelsOff`, `WheelsOn` | Generated during flight operations |
| Delay-cause breakdown | `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay` | Attributed after the flight |
| Gate-time additions | `FirstDepTime`, `TotalAddGTime`, `LongestAddGTime` | Recorded during operations |
| Cancellation / diversion | `Cancelled`, `CancellationCode`, `Diverted`, all `Div*` columns | Known only at/after the event |

---

## 5. Safe Features — KEEP

All sourced from the published flight schedule and therefore available 2 hours before departure.

| Feature | Column | Rationale |
|---------|--------|-----------|
| Airline | `Reporting_Airline` | Carriers have different historical delay rates |
| Origin airport | `Origin` | Congestion varies by airport |
| Destination airport | `Dest` | Destination traffic affects operations |
| Scheduled departure time | `CRSDepTime` | Time-of-day is a strong delay signal |
| Scheduled arrival time | `CRSArrTime` | From the flight plan |
| Scheduled elapsed time | `CRSElapsedTime` | From the flight plan |
| Distance | `Distance` | From the flight plan (miles) |
| Day of week | `DayOfWeek` | Weekday vs weekend traffic |
| Month | `Month` | Seasonality |
| Aircraft | `Tail_Number` | Safe if taken from the schedule |
| Flight number | `Flight_Number_Reporting_Airline` | Route identifier |

**Derived features (planned):**

- `dep_hour` — extracted from `CRSDepTime` (morning vs evening delay patterns)
- Optional later iteration: `DayOfWeek × dep_hour` interaction

---

## 6. Weather Strategy — Historical Delay Patterns (Option C)

The BTS dataset contains **no raw weather data** (no temperature, wind, precipitation, visibility, or cloud ceiling). The only weather field, `WeatherDelay`, is leaky because it is attributed after the flight.

**Chosen approach:** use historical delay patterns as a weather proxy.

**Definition:**

```
For each (Airport, Month, departure hour):
    origin_delay_baseline = median historical DepDelay at this origin / month / hour
    dest_delay_baseline   = median historical DepDelay at this dest / month / hour
```

These baselines are merged back onto each flight as features.

**Why this is not leakage:**

- Baselines are computed **only from training-set flights**, then applied to validation and test sets.
- They summarise *past* behaviour, never the current flight.

**Why this is defensible:**

- Time-of-day and season correlate strongly with weather-driven delays (winter ice, summer thunderstorms, evening turbulence buildup).
- Captures recurring weather effects without requiring an external weather feed.

---

## 7. Train / Validation / Test Split (Temporal Integrity)

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

**Baseline computation rule:** all historical delay baselines (Section 6) are calculated using the **training period only**, then applied forward to validation and test. This prevents future leakage through the engineered weather proxy.

---

## 8. Leakage Detection Experiment

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

The hardest part of this project is **not** model selection. It is ensuring that every feature would genuinely be available 2 hours before departure.

A model with slightly lower accuracy and **zero leakage** is far more valuable than a high-accuracy model trained on information from the future. This project is an exercise in realistic deployment thinking and disciplined handling of temporal data.

---
