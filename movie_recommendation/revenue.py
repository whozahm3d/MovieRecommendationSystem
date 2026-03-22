from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RevenueModelArtifacts:
    model: RandomForestRegressor
    scaler: StandardScaler
    features: list[str]
    r2: float
    mae: float
    training_frame: pd.DataFrame


def train_revenue_model(df: pd.DataFrame) -> RevenueModelArtifacts:
    revenue_df = df.dropna(subset=["revenue"]).copy()
    features = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
    revenue_df = revenue_df.dropna(subset=features)

    X = revenue_df[features]
    y = revenue_df["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return RevenueModelArtifacts(
        model=model,
        scaler=scaler,
        features=features,
        r2=r2_score(y_test, predictions),
        mae=mean_absolute_error(y_test, predictions),
        training_frame=revenue_df[features + ["revenue"]],
    )
