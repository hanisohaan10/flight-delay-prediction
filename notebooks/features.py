import pandas as pd

def encode_features(X_train, X_val):

    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)

    X_train, X_val = X_train.align(
        X_val,
        join="left",
        axis=1,
        fill_value=0
    )

    return X_train, X_val

def make_features(df):

    df = df.copy()

    df["dep_hour"] = df["CRSDepTime"] // 100

    df["is_weekend"] = (
        df["DayOfWeek"].isin([6,7])
    ).astype(int)

    def get_part_of_day(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    df["part_of_day"] = df["dep_hour"].apply(get_part_of_day)

    X = df[[
        "Reporting_Airline",
        "DayOfWeek",
        "dep_hour",
        "is_weekend",
        "part_of_day",
        "Distance",
        "OriginAirportID",
        "DestAirportID",
        
    ]]

    return X