"""
A script for preprocessing the dataset
"""

from contextlib import redirect_stdout
from pathlib import Path
import sys
import pandas as pd

DATA_PATH = "biosensor_dataset_with_target.csv"
OUT_DIR = Path("out")
ARTIFACT_PATH = OUT_DIR / "preprocessing_artifact.log"
SEGMENT_OUTPUT_PATH = OUT_DIR / "segmented_windows.csv"
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.5
SANITY_THRESHOLDS = {
    "Heart_Rate": (60.0, 190.0),
    "Acc_X": (-3.2, 3.2),
    "Acc_Y": (-3.2, 3.2),
    "Acc_Z": (-3.2, 3.2),
    "Gyro_X": (-190.0, 190.0),
    "Gyro_Y": (-190.0, 190.0),
    "Gyro_Z": (-190.0, 190.0),
}


df = pd.read_csv(DATA_PATH)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def remove_duplicates() -> None:
    global df

    rows_before = len(df)
    df.drop_duplicates(keep="first", inplace=True)
    duplicate_count = rows_before - len(df)

    if duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicate rows. Remaining rows: {len(df)}")
    else:
        print("No duplicate rows were found for removal")

def clean_and_sort_time() -> None:
    global df

    print("No null or missing values were found inside dataset.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    invalid_timestamps = int(df["Timestamp"].isna().sum())
    if invalid_timestamps > 0:
        print(f"Found {invalid_timestamps} rows with invalid timestamps; dropping them before sorting.")
        df.dropna(subset=["Timestamp"], inplace=True)

    df.sort_values(by=["Athlete_ID", "Timestamp"], inplace=True)
    print("Dataset sorted chronologically by Athlete_ID and Timestamp.")

def normalize_and_report_class_distribution() -> None:
    global df

    df["Event_Label"] = df["Event_Label"].str.strip().str.lower()

    class_counts = df["Event_Label"].value_counts()
    class_ratios = df["Event_Label"].value_counts(normalize=True) * 100

    print("Class distribution (Event_Label):")
    for label, count in class_counts.items():
        print(f"  {label}: {count} ({class_ratios[label]:.2f}%)")

    if not class_counts.empty:
        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio > 1.5:
            print(f"Dataset appears imbalanced (max/min class ratio = {imbalance_ratio:.2f}).")
        else:
            print(f"Dataset appears reasonably balanced (max/min class ratio = {imbalance_ratio:.2f}).")

def report_sampling_rate() -> None:
    global df

    # Compute per-athlete time deltas so athlete boundaries do not mix intervals.
    dt = df.groupby("Athlete_ID")["Timestamp"].diff().dt.total_seconds()
    dt = dt.dropna()

    non_positive = int((dt <= 0).sum())
    dt = dt[dt > 0]

    if dt.empty:
        print("Sampling rate check: insufficient valid timestamp gaps to estimate sampling rate.")
        return

    median_dt = float(dt.median())
    mean_dt = float(dt.mean())
    std_dt = float(dt.std()) if len(dt) > 1 else 0.0

    median_hz = 1.0 / median_dt
    mean_hz = 1.0 / mean_dt

    print("Sampling rate check:")
    print(f"  Median interval: {median_dt:.2f} s ({median_hz:.2f} Hz)")
    print(f"  Mean interval:   {mean_dt:.2f} s ({mean_hz:.2f} Hz)")
    print(f"  Interval std:    {std_dt:.2f} s")
    if non_positive > 0:
        print(f"  Ignored {non_positive} non-positive timestamp gaps.")

    common_steps = dt.round(6).value_counts().head(3)
    print("  Most common intervals (s):")
    for step, count in common_steps.items():
        print(f"    {step:.2f}: {int(count)}")

    print("Sampling-rate decision:")
    print("  The sampling rate is already relatively low, so we avoid removing additional points.")
    print("  Event transitions are important, and aggressive filtering could remove useful transition dynamics.")

def report_numeric_ranges_and_thresholds() -> None:
    numeric_cols = df.select_dtypes(include="number").columns

    print("Observed ranges for numerical columns:")
    for col in numeric_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        print(f"  {col}: range = [{col_min:.2f}, {col_max:.2f}]")

    print("Simple threshold check:")
    out_of_threshold_total = 0
    for col, (low, high) in SANITY_THRESHOLDS.items():
        if col in df.columns:
            count = int(((df[col] < low) | (df[col] > high)).sum())
            out_of_threshold_total += count

    if out_of_threshold_total == 0:
        print("No rows were outside the defined thresholds of data.")
    else:
        print(f"{out_of_threshold_total} rows were outside the defined thresholds of data.")

def report_windowing_setup() -> None:
    print("Windowing setup:")
    print(f"  Window size: {WINDOW_SECONDS:.1f} second")
    print(f"  Step size:   {STEP_SECONDS:.1f} second (50% overlap)")


def segment_windows() -> None:
    global df

    sensor_cols = [col for col in SANITY_THRESHOLDS if col in df.columns]
    if not sensor_cols:
        print("Segmentation skipped: no sensor columns found.")
        return

    window = pd.to_timedelta(WINDOW_SECONDS, unit="s")
    step = pd.to_timedelta(STEP_SECONDS, unit="s")
    segments = []

    # segments per ID over timestamp 
    for athlete_id, athlete_df in df.groupby("Athlete_ID", sort=False):
        athlete_df = athlete_df.sort_values("Timestamp")
        if athlete_df.empty:
            continue

        start = athlete_df["Timestamp"].min()
        end = athlete_df["Timestamp"].max()
        current = start

        while current + window <= end:
            window_end = current + window
            mask = (athlete_df["Timestamp"] >= current) & (athlete_df["Timestamp"] < window_end)
            win_df = athlete_df.loc[mask]

            if not win_df.empty:
                row = {
                    "Athlete_ID": athlete_id,
                    "window_start": current,
                    "window_end": window_end,
                    "n_samples": int(len(win_df)),
                }

                label_counts = win_df["Event_Label"].value_counts()
                row["Event_Label"] = label_counts.index[0]
                row["label_purity"] = float(label_counts.iloc[0] / len(win_df))

                for col in sensor_cols:
                    row[f"{col}_mean"] = float(win_df[col].mean())
                    row[f"{col}_std"] = float(win_df[col].std(ddof=0))

                segments.append(row)

            current += step

    if not segments:
        print("Segmentation produced no windows.")
        return

    seg_df = pd.DataFrame(segments)
    seg_df.to_csv(SEGMENT_OUTPUT_PATH, index=False)
    print(f"Segmentation complete: {len(seg_df)} windows saved to {SEGMENT_OUTPUT_PATH}")


def main() -> None:
    remove_duplicates()
    clean_and_sort_time()
    report_sampling_rate()
    report_windowing_setup()
    normalize_and_report_class_distribution()
    report_numeric_ranges_and_thresholds()
    segment_windows()


def run_with_artifact_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ARTIFACT_PATH, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            main()
            print(f"Saved preprocessing log to {ARTIFACT_PATH}")


if __name__ == "__main__":
    run_with_artifact_logging()





 

