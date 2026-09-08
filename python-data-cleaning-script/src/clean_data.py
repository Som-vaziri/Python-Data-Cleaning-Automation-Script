import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "adult.csv"
CLEANED_FILE = OUTPUT_DIR / "cleaned_data_adult.csv"
REPORT_FILE = OUTPUT_DIR / "report_adult.txt"


# ---------------- Validation ----------------
def validate_columns(df: pd.DataFrame):
    required_cols = {"age", "income"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------- Cleaning functions ----------------
def clean_age(series: pd.Series) -> pd.Series:
    age_map = {
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
    }

    cleaned = (
        series.astype(str)
        .str.lower()
        .replace(age_map)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )

    return cleaned.where((cleaned >= 0) & (cleaned <= 120))


def clean_salary(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )

    return cleaned.where(cleaned > 0)


# ---------------- Main cleaning pipeline ----------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    df = df.drop_duplicates().copy()

    # Keep raw data (debugging & traceability)
    df["age_raw"] = df["age"]
    df["income_raw"] = df["income"]

    # ---- AGE ----
    cleaned_age = clean_age(df["age"])

    if cleaned_age.notna().sum() == 0:
        raise ValueError("Age cleaning failed: all values are NaN")

    df.loc[:, "age"] = cleaned_age.fillna(cleaned_age.median())

    # ---- SALARY ----
    cleaned_salary = clean_salary(df["income"])

    if cleaned_salary.notna().sum() == 0:
        raise ValueError("Salary cleaning failed: all values are NaN")

    df.loc[:, "income"] = cleaned_salary.fillna(cleaned_salary.median())

    return df


# ---------------- Report ----------------
def generate_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    report = [
        "=== DATA CLEANING REPORT ===",
        f"Original rows: {len(original_df)}",
        f"Cleaned rows: {len(cleaned_df)}",
        "\nMissing values after cleaning:",
        str(cleaned_df.isnull().sum()),
        "\nSummary statistics:",
        str(cleaned_df.describe()),
    ]

    REPORT_FILE.write_text("\n".join(report))


# ---------------- Visualization ----------------
def plot_distributions(df: pd.DataFrame):
    # Raw age
    raw_age = (
        df["income_raw"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

    plt.figure()
    raw_age.hist()
    plt.title("Raw income Distribution")

    # Cleaned age
    plt.figure()
    df["income"].hist()
    plt.title("Cleaned income Distribution")

    plt.show()


# ---------------- Main ----------------
def main():
    print("📥 Loading data...")
    df = pd.read_csv(INPUT_FILE)

    print("🧹 Cleaning data...")
    cleaned_df = clean_data(df)

    print("💾 Saving outputs...")
    cleaned_df.to_csv(CLEANED_FILE, index=False)
    generate_report(df, cleaned_df)

    print("📊 Plotting...")
    plot_distributions(cleaned_df)

    print("✅ Done successfully!")


if __name__ == "__main__":
    main()
