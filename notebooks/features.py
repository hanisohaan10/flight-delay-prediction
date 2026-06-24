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

def add_target_encoding(train_df,val_df,category_col,target_col,feature_name,m=100):
    
    # overall mean in training data
    global_mean = train_df[target_col].mean()

    stats = (
        train_df
        .groupby(category_col)[target_col]
        .agg(["mean", "count"])
    )

    stats[feature_name] = (
        stats["count"] * stats["mean"]
        + m * global_mean
    ) / (
        stats["count"] + m
    )

    mapping = stats[feature_name]

    train_df[feature_name] = (
        train_df[category_col]
        .map(mapping)
        .fillna(global_mean)
    )

    val_df[feature_name] = (
        val_df[category_col]
        .map(mapping)
        .fillna(global_mean)
    )

def make_features(train_df, val_df):

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["dep_hour"] = train_df["CRSDepTime"] // 100
    val_df["dep_hour"] = val_df["CRSDepTime"] // 100

    encodings = [
        ("Reporting_Airline", "airline_delay_rate"),
        ("Origin", "origin_delay_rate"),
        ("Dest", "dest_delay_rate")
    ]

    for col, new_col in encodings:
        add_target_encoding(
            train_df,
            val_df,
            col,
            "DepDel15",
            new_col,
            m=100
        )

    X_train = train_df[[
        #"Reporting_Airline",
        "DayOfWeek",
        "dep_hour",
        #"is_weekend",
        #"part_of_day",
        "Distance",
        #"OriginAirportID",
        #"DestAirportID",
        #"arr_hour",
        "airline_delay_rate",
        "origin_delay_rate",
        "dest_delay_rate"
    ]]
    X_val = val_df[[
        #"Reporting_Airline",
        "DayOfWeek",
        "dep_hour",
        "Distance",
        #"OriginAirportID",
        #"DestAirportID",
        "airline_delay_rate",
        "origin_delay_rate",
        "dest_delay_rate"
    ]]

    return X_train,X_val

def fit_encoders(train_df, m=100):

    global_mean = train_df["DepDel15"].mean()

    encoders = {
        "global_mean": global_mean
    }

    for category_col, feature_name in [
        ("Reporting_Airline", "airline_delay_rate"),
        ("Origin", "origin_delay_rate"),
        ("Dest", "dest_delay_rate")
    ]:

        stats = (
            train_df
            .groupby(category_col)["DepDel15"]
            .agg(["mean", "count"])
        )

        stats[feature_name] = (
            stats["count"] * stats["mean"]
            + m * global_mean
        ) / (
            stats["count"] + m
        )

        encoders[feature_name] = (
            stats[feature_name]
            .to_dict()
        )

    return encoders

def transform_features(df, encoders):

    df = df.copy()

    global_mean = encoders["global_mean"]

    df["dep_hour"] = (
        df["CRSDepTime"] // 100
    )

    df["airline_delay_rate"] = (
        df["Reporting_Airline"]
        .map(encoders["airline_delay_rate"])
        .fillna(global_mean)
    )

    df["origin_delay_rate"] = (
        df["Origin"]
        .map(encoders["origin_delay_rate"])
        .fillna(global_mean)
    )

    df["dest_delay_rate"] = (
        df["Dest"]
        .map(encoders["dest_delay_rate"])
        .fillna(global_mean)
    )

    return df[
        [
            "DayOfWeek",
            "dep_hour",
            "Distance",
            "airline_delay_rate",
            "origin_delay_rate",
            "dest_delay_rate"
        ]
    ]