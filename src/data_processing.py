import os
import pandas as pd
import numpy as np

# Config

RAW_DIR = "data/raw_data"
OUT_DIR = "data/processed"

TRAIN_PATH = os.path.join(OUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUT_DIR, "test.csv")
LIVE_PATH = os.path.join(OUT_DIR, "live_data.csv")

TRAIN_END_YEAR = 2009   # train until end 2009
TEST_START_YEAR = 2010  # test from 2010 onward

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
    #  Save the full train and test datasets
    train.to_csv(TRAIN_PATH, index=True)
    test.to_csv(TEST_PATH, index=True)

    #  Slice 0 rows from train to get just the headers (and the index)
    empty_live = train.iloc[0:0]
    empty_live.to_csv(LIVE_PATH, index=True)

    print("\nSaved files:")
    print(f"{TRAIN_PATH} : {train.shape}")
    print(f"{TEST_PATH}  : {test.shape}")
    print(f"{LIVE_PATH}  : {empty_live.shape} (Headers only)")



def main():
    raw_file ='data/raw_data/household_power_consumption.txt'

    df = load_raw_data(raw_file)

    print("Raw shape:", df.shape)

    os.makedirs(OUT_DIR, exist_ok=True)

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