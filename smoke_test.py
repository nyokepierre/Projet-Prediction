
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
bundle = joblib.load(APP_DIR / "modeles" / "modele_acces_soins_bundle.joblib")
config = __import__("json").loads(
    (APP_DIR / "app_config.json").read_text(encoding="utf-8")
)

scenario = config["scenarios"]["Ménage intermédiaire"]["inputs"]
row = pd.DataFrame(
    [[
        np.nan if scenario.get(feature) is None else scenario.get(feature)
        for feature in bundle["feature_order"]
    ]],
    columns=bundle["feature_order"],
)

transformed = bundle["preprocessor"].transform(row)
p_xgb = float(bundle["xgb_model"].predict_proba(transformed)[0, 1])
p_rf = float(bundle["rf_model"].predict_proba(transformed)[0, 1])

assert transformed.shape[1] == 319, transformed.shape
assert 0.0 <= p_xgb <= 1.0
assert 0.0 <= p_rf <= 1.0
assert len(bundle["feature_order"]) == 54

print("Test réussi")
print(f"Dimensions transformées : {transformed.shape}")
print(f"Probabilité XGBoost : {p_xgb:.4f}")
print(f"Probabilité Random Forest : {p_rf:.4f}")
