# data_processing.py
# ==========================================================
# Energy Forecasting - Data Processing Pipeline
#
# Reads raw household power consumption data from:
#   data/raw_data/
#
# Produces:
#   data/processed/train.csv
#   data/processed/test.csv
#
# Design Goals:
# - Preserve same schema in train/test for live row transfer
# - Hourly aggregated data
# - Chronological split
# - Test set simulates incoming live stream
# - Adds shifted target for evaluation:
#       target_next_hour = next hour Global_active_power
#
# So at time t:
#   features row uses current known values
#   target_next_hour = actual at t+1
#
# Later when row is consumed from test -> append to train.
# ==========================================================

import os
import pandas as pd
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================

RAW_DIR = "data/raw_data"
OUT_DIR = "data/processed"

TRAIN_PATH = os.path.join(OUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUT_DIR, "test.csv")

TRAIN_END_YEAR = 2009   # train until end 2009
TEST_START_YEAR = 2010  # test from 2010 onward

os.makedirs(OUT_DIR, exist_ok=True)


def load_raw_data(path):
    print(f"Loading raw file: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values=["nan", "?"]
    )

    # combine date + time
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    df.drop(columns=["Date", "Time"], inplace=True)

    df.set_index("timestamp", inplace=True)

    # numeric conversion
    df = df.apply(pd.to_numeric, errors="coerce")

    # fill missing values by time interpolation
    df = df.interpolate(method="time")

    return df


def resample_hourly(df):
    # convert minute data to hourly averages
    hourly = df.resample("h").mean()
    hourly = hourly.interpolate(method="time")
    return hourly


def add_target(df):
    # next hour actual consumption
    df["target_next_hour"] = df["Global_active_power"].shift(-1)

    # final row has no next hour target
    df = df.iloc[:-1].copy()

    return df


def split_train_test(df):
    train = df.loc[:str(TRAIN_END_YEAR)].copy()
    test = df.loc[str(TEST_START_YEAR):].copy()

    return train, test


def save_files(train, test):
    train.to_csv(TRAIN_PATH, index=True)
    test.to_csv(TEST_PATH, index=True)

    print("\nSaved files:")
    print(TRAIN_PATH, train.shape)
    print(TEST_PATH, test.shape)


# ==========================================================
# MAIN
# ==========================================================

def main():
    raw_file ='data/raw_data/household_power_consumption.txt'

    df = load_raw_data(raw_file)

    print("Raw shape:", df.shape)

    df = resample_hourly(df)

    print("Hourly shape:", df.shape)

    df = add_target(df)

    print("After target shift:", df.shape)

    train, test = split_train_test(df)

    print("\nTrain Range:")
    print(train.index.min(), "->", train.index.max())

    print("\nTest Range:")
    print(test.index.min(), "->", test.index.max())

    print("\nTrain shape:", train.shape)
    print("Test shape :", test.shape)

    save_files(train, test)

    print("\nColumns:")
    print(list(train.columns))


if __name__ == "__main__":
    main()