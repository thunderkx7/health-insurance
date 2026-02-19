"""
train.py  —  Run ONCE offline.
Trains models, picks best, saves model.pkl + meta.pkl.

    python train.py
"""

import pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── SET YOUR PATH ─────────────────────────────────────────────────
CSV_PATH = r"C:\Users\GANESH\OneDrive - National Institute of Technology\Desktop\manoj project\insurance.csv"
# ─────────────────────────────────────────────────────────────────

# Columns that are meaningless for prediction — always dropped
ID_COLS = {"id", "index", "unnamed: 0", "unnamed:0",
           "patient_id", "customerid", "customer_id", "no"}

# ── Load & basic clean ────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()

dropped = [c for c in df.columns if c in ID_COLS]
df.drop(columns=dropped, inplace=True)
if dropped:
    print(f"Dropped ID cols: {dropped}")

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip().str.lower()
    df[c].fillna(df[c].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

# Outlier removal (3×IQR)
for c in df.select_dtypes(include=np.number).columns:
    Q1, Q3 = df[c].quantile(.25), df[c].quantile(.75)
    df = df[df[c].between(Q1 - 3*(Q3-Q1), Q3 + 3*(Q3-Q1))]

df = df.reset_index(drop=True)
print(f"Clean shape: {df.shape}")

# ── Column roles ──────────────────────────────────────────────────
TARGET   = "charges" if "charges" in df.columns else df.select_dtypes(include=np.number).columns[-1]
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]
cat_opts = {c: sorted(df[c].unique().tolist()) for c in cat_cols}

print(f"Target     : {TARGET}")
print(f"Numeric    : {num_cols}")
print(f"Categorical: {cat_cols}")

# ── Feature engineering ───────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns and "bmi" in df.columns:
        df["_age_bmi"] = df["age"] * df["bmi"]
        df["_age_sq"]  = df["age"] ** 2
    if "bmi" in df.columns:
        df["_obese"] = (df["bmi"] >= 30).astype(int)
    for col, lvls in cat_opts.items():
        if len(lvls) == 2:
            df[f"_enc_{col}"] = (df[col] != lvls[0]).astype(int)
        else:
            for lv in lvls[1:]:
                df[f"_ohe_{col}_{lv}"] = (df[col] == lv).astype(int)
    if "_enc_smoker" in df.columns and "bmi" in df.columns:
        df["_smoker_bmi"] = df["_enc_smoker"] * df["bmi"]
    df.drop(columns=cat_cols, inplace=True, errors="ignore")
    return df

df_f      = make_features(df)
feat_cols = [c for c in df_f.columns if c != TARGET]

X, y_raw = df_f[feat_cols], df_f[TARGET]
y        = np.log1p(y_raw)

X_tr, X_te, y_tr, y_te = train_test_split(X, y,     test_size=.2, random_state=42)
_,    _,    _,  y_te_r  = train_test_split(X, y_raw, test_size=.2, random_state=42)

# ── Models (all constrained to prevent overfitting) ───────────────
models = {
    "Linear Regression (Ridge)": Pipeline([
        ("sc", StandardScaler()), ("m", Ridge(alpha=10))
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=6,          # shallow → generalises better
        min_samples_leaf=10,  # each leaf needs ≥10 samples
        max_features=0.6,     # random feature subsampling
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=.05,
        max_depth=4, subsample=.8,
        min_samples_leaf=10,
        loss="huber", random_state=42
    ),
}
try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=.05,
        max_depth=4, subsample=.8, colsample_bytree=.8,
        min_child_weight=10, reg_alpha=0.1, reg_lambda=5,
        objective="reg:squarederror",
        random_state=42, verbosity=0, n_jobs=-1
    )
except ImportError:
    print("XGBoost not installed — skipping.")

# ── Train & compare ───────────────────────────────────────────────
print("\n{'Model':<35} {'Train R²':>10} {'Test R²':>9} {'Test RMSE':>11}")
print("-" * 70)
best_rmse, best_model, best_name = np.inf, None, ""

for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    p_tr = np.expm1(mdl.predict(X_tr))
    p_te = np.expm1(mdl.predict(X_te))
    r2_tr = r2_score(np.expm1(y_tr), p_tr)
    r2_te = r2_score(y_te_r, p_te)
    rmse  = np.sqrt(mean_squared_error(y_te_r, p_te))
    print(f"{name:<35} {r2_tr:>10.3f} {r2_te:>9.3f} ${rmse:>10,.0f}")
    if rmse < best_rmse:
        best_rmse, best_model, best_name = rmse, mdl, name

best_r2 = r2_score(y_te_r, np.expm1(best_model.predict(X_te)))
print(f"\n✅ Best: {best_name}  (Test R²={best_r2:.3f}, RMSE=${best_rmse:,.0f})")

# ── Save artifacts ────────────────────────────────────────────────
meta = {
    "target":    TARGET,
    "feat_cols": feat_cols,
    "num_cols":  num_cols,
    "cat_cols":  cat_cols,
    "cat_opts":  cat_opts,
    "q33":       float(df[TARGET].quantile(.33)),
    "q67":       float(df[TARGET].quantile(.67)),
    "df_stats":  {c: {"min": float(df[c].min()),
                       "max": float(df[c].max()),
                       "med": float(df[c].median())}
                  for c in num_cols},
}

with open("model.pkl", "wb") as f: pickle.dump(best_model, f)
with open("meta.pkl",  "wb") as f: pickle.dump(meta, f)
print("✅ Saved model.pkl + meta.pkl")
print("Now run:  streamlit run app.py")
