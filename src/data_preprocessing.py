import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================

FILE_PATH = "data/raw_data/energydata_complete.csv"
TARGET_COL = "Appliances"

PAST_WINDOW = 144      # previous 24 hrs (10 min interval)
FUTURE_HORIZON = 18    # next 3 hrs

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

XGB_DIR = Path("data/xgb_data")
LSTM_DIR = Path("data/lstm_data")

XGB_DIR.mkdir(parents=True, exist_ok=True)
LSTM_DIR.mkdir(parents=True, exist_ok=True)

# Laod Data
df = pd.read_csv(FILE_PATH)

# Parse date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# drop useless random columns
drop_cols = [c for c in ["rv1", "rv2"] if c in df.columns]
df = df.drop(columns=drop_cols)

print("Loaded shape:", df.shape)

# Time Features
df["hour"] = df["date"].dt.hour
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# Cyclical Encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)


# LAG FEATURES
lags = [1, 2, 3, 6, 12, 18, 36, 72, 132, 138, 144]

for lag in lags:
    df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

# ======================================================
# ROLLING FEATURES
# ======================================================

windows = [6, 18, 36, 144]

for w in windows:
    df[f"roll_mean_{w}"] = df[TARGET_COL].shift(1).rolling(w).mean()
    df[f"roll_std_{w}"] = df[TARGET_COL].shift(1).rolling(w).std()
    df[f"roll_max_{w}"] = df[TARGET_COL].shift(1).rolling(w).max()

# ======================================================
# FUTURE TARGETS (18 exact steps)
# ======================================================

for h in range(1, FUTURE_HORIZON + 1):
    df[f"target_{h}"] = df[TARGET_COL].shift(-h)


# DROP NaNs
df = df.dropna().reset_index(drop=True)

print("After feature engineering:", df.shape)


# TRAIN / VAL / TEST SPLIT
n = len(df)

train_end = int(TRAIN_RATIO * n)
val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\nTrain:", train_df.shape)
print("Val  :", val_df.shape)
print("Test :", test_df.shape)

# ======================================================
# TABULAR DATA FOR XGBOOST / LIGHTGBM
# ======================================================

target_cols = [f"target_{i}" for i in range(1, FUTURE_HORIZON + 1)]

drop_features = ["date", TARGET_COL] + target_cols

feature_cols = [c for c in df.columns if c not in drop_features]

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[target_cols]
y_val = val_df[target_cols]
y_test = test_df[target_cols]

# save raw tabular csv
X_train.to_csv(f"{XGB_DIR}/X_train.csv", index=False)
X_val.to_csv(f"{XGB_DIR}/X_val.csv", index=False)
X_test.to_csv(f"{XGB_DIR}/X_test.csv", index=False)

y_train.to_csv(f"{XGB_DIR}/y_train.csv", index=False)
y_val.to_csv(f"{XGB_DIR}/y_val.csv", index=False)
y_test.to_csv(f"{XGB_DIR}/y_test.csv", index=False)

print("\nSaved tabular datasets.")

# ======================================================
# SCALE FEATURES FOR LSTM
# ======================================================

# use original real columns (excluding targets/date)
base_feature_cols = [c for c in df.columns if c not in ["date"] + target_cols]

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_df[base_feature_cols])
val_scaled = scaler.transform(val_df[base_feature_cols])
test_scaled = scaler.transform(test_df[base_feature_cols])

joblib.dump(scaler, f"{LSTM_DIR}/scaler.pkl")

# ======================================================
# LSTM SEQUENCE CREATION
# ======================================================

def create_sequences(data_array, original_df):
    X_seq = []
    y_seq = []

    target_array = original_df[target_cols].values

    for i in range(PAST_WINDOW, len(data_array)):
        X_seq.append(data_array[i-PAST_WINDOW:i])
        y_seq.append(target_array[i])

    return np.array(X_seq), np.array(y_seq)


X_train_seq, y_train_seq = create_sequences(train_scaled, train_df)
X_val_seq, y_val_seq = create_sequences(val_scaled, val_df)
X_test_seq, y_test_seq = create_sequences(test_scaled, test_df)

print("\nLSTM Shapes:")
print("Train:", X_train_seq.shape, y_train_seq.shape)
print("Val  :", X_val_seq.shape, y_val_seq.shape)
print("Test :", X_test_seq.shape, y_test_seq.shape)

# SAVE LSTM ARRAYS

np.save(f"{LSTM_DIR}/X_train_seq.npy", X_train_seq)
np.save(f"{LSTM_DIR}/X_val_seq.npy", X_val_seq)
np.save(f"{LSTM_DIR}/X_test_seq.npy", X_test_seq)

np.save(f"{LSTM_DIR}/y_train_seq.npy", y_train_seq)
np.save(f"{LSTM_DIR}/y_val_seq.npy", y_val_seq)
np.save(f"{LSTM_DIR}/y_test_seq.npy", y_test_seq)

