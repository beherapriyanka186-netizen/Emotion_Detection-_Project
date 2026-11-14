
"""
File: src/data/data_validator.py

Purpose:
1. Check dataset file existence.
2. Validate required columns.
3. Check missing values.
4. Identify duplicate records.
5. Identify empty text.
6. Validate emotion labels.
7. Check data types.
8. Generate a dataset quality report.
"""

from pathlib import Path

import pandas as pd
import yaml


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ==========================================
# 2. LOAD CONFIGURATION
# ==========================================

def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ==========================================
# 3. LOAD DATASET FILES
# ==========================================

def load_data(config):

    data_config = config["data"]

    train_path = PROJECT_ROOT / data_config["train_file"]
    validation_path = PROJECT_ROOT / data_config["validation_file"]
    test_path = PROJECT_ROOT / data_config["test_file"]

    datasets = {}

    file_paths = {
        "Training": train_path,
        "Validation": validation_path,
        "Testing": test_path
    }

    for name, path in file_paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}"
            )

        datasets[name] = pd.read_csv(path)

    return datasets


# ==========================================
# 4. VALIDATE REQUIRED COLUMNS
# ==========================================

def validate_columns(datasets, config):

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]

    required_columns = {text_column, label_column}

    print("\n========== COLUMN VALIDATION ==========")

    for name, df in datasets.items():

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{name} dataset missing columns: {missing_columns}"
            )

        print(f"{name}: Required columns available")


# ==========================================
# 5. CHECK MISSING VALUES
# ==========================================

def check_missing_values(datasets):

    print("\n========== MISSING VALUE CHECK ==========")

    for name, df in datasets.items():

        missing = df.isnull().sum().sum()

        print(f"{name} missing values: {missing}")

        if missing > 0:
            print(df.isnull().sum())


# ==========================================
# 6. CHECK DUPLICATE RECORDS
# ==========================================

def check_duplicates(datasets, text_column):

    print("\n========== DUPLICATE CHECK ==========")

    for name, df in datasets.items():

        duplicates = df.duplicated(
            subset=[text_column]
        ).sum()

        print(f"{name} duplicate texts: {duplicates}")


# ==========================================
# 7. CHECK EMPTY TEXT
# ==========================================

def check_empty_text(datasets, text_column):

    print("\n========== EMPTY TEXT CHECK ==========")

    for name, df in datasets.items():

        empty_text = (
            df[text_column]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
            .sum()
        )

        print(f"{name} empty text records: {empty_text}")


# ==========================================
# 8. VALIDATE EMOTION LABELS
# ==========================================

def validate_labels(datasets, config, label_column):

    print("\n========== LABEL VALIDATION ==========")

    expected_labels = set(
        range(config["labels"]["num_classes"])
    )

    for name, df in datasets.items():

        actual_labels = set(
            pd.to_numeric(
                df[label_column],
                errors="coerce"
            ).dropna().unique()
        )

        invalid_labels = actual_labels - expected_labels

        missing_labels = expected_labels - actual_labels

        print(f"\n{name} dataset:")

        print("Available labels:", sorted(actual_labels))

        if invalid_labels:
            print("Invalid labels:", sorted(invalid_labels))

        else:
            print("Invalid labels: None")

        if missing_labels:
            print("Labels not represented:", sorted(missing_labels))

        else:
            print("All expected classes represented")

        if not pd.api.types.is_integer_dtype(df[label_column]):
            raise ValueError(
                f"{name}: Label column must contain integer labels"
            )

        if invalid_labels:
            raise ValueError(
                f"{name}: Invalid emotion labels found"
            )


# ==========================================
# 9. CHECK DATA TYPES
# ==========================================

def check_data_types(datasets, text_column, label_column):

    print("\n========== DATA TYPE CHECK ==========")

    for name, df in datasets.items():

        print(f"\n{name} dataset:")

        print(df[[text_column, label_column]].dtypes)

        if not (
            pd.api.types.is_object_dtype(df[text_column])
            or pd.api.types.is_string_dtype(df[text_column])
        ):
            raise TypeError(
                f"{name}: Text column must contain strings"
            )


# ==========================================
# 10. DISPLAY DATASET SUMMARY
# ==========================================

def display_summary(datasets):

    print("\n========== DATASET SUMMARY ==========")

    total_records = 0

    for name, df in datasets.items():

        records = len(df)

        total_records += records

        print(f"{name} records: {records}")

    print("--------------------------------------")

    print("Total records:", total_records)


# ==========================================
# 11. MAIN EXECUTION
# ==========================================

def main():

    print("======================================")
    print("      DATASET VALIDATION MODULE")
    print("======================================")

    config = load_config()

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]

    # Load datasets
    datasets = load_data(config)

    # Validate columns
    validate_columns(datasets, config)

    # Check missing values
    check_missing_values(datasets)

    # Check duplicate records
    check_duplicates(datasets, text_column)

    # Check empty text
    check_empty_text(datasets, text_column)

    # Validate emotion labels
    validate_labels(datasets, config, label_column)

    # Check data types
    check_data_types(
        datasets,
        text_column,
        label_column
    )

    # Display summary
    display_summary(datasets)

    print("\nDataset validation completed successfully!")


# ==========================================
# 12. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
