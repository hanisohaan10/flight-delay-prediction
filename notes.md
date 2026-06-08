# Leakage Decision: DEP_DELAY

**Decision:** `DEP_DELAY` is treated as leakage and excluded from the feature set.

**Reasoning:** The project's goal is to predict whether a flight will be delayed by more than 15 minutes **before departure**. `DEP_DELAY` represents the actual departure delay, which is only known at or after departure time and is directly used to define the target variable. Including it would give the model access to information unavailable at prediction time, leading to unrealistically high performance that would not generalize to real-world deployment.

**Rule Used:** A feature is allowed only if it would be known before the prediction is made. Since `DEP_DELAY` is not known before departure, it violates this rule and is therefore considered data leakage.
