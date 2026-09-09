
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(path):
    """Load dataset from CSV file."""

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


def inspect_data(df):
    """Display basic information about the dataset."""

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

def clean_data(df):
    """Clean dataset by handling spaces, missing values, and redundant columns."""

    # Remove leading/trailing whitespace from string values
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Replace '?' with NaN
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with missing values
    df = df.dropna()

    # Remove redundant column if present
    if 'education.num' in df.columns:
        df = df.drop(columns=['education.num'])

    return df



def encode_target(df):
    """Convert income column to binary labels."""

    df['income'] = df['income'].map({
        '<=50K': 0,
        '>50K': 1
    })

    return df


def identify_columns(df, target_col):
    """Automatically identify numerical and categorical columns."""

    # Remove target column from feature columns
    X = df.drop(columns=[target_col])

    # Find numerical columns automatically
    numerical_cols = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    # Find categorical columns automatically
    categorical_cols = X.select_dtypes(
        include=['object', 'string', 'category']
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_cols)

    print("\nCategorical columns:")
    print(categorical_cols)

    return numerical_cols, categorical_cols


def build_pipeline(numerical_cols, categorical_cols):
    """Create preprocessing and model pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return model


def train_model(df, model):
    """Split data, train the model, and return test set."""

    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate(model, X_test, y_test):
    """Evaluate model performance."""

    y_pred = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    """Main execution pipeline."""

    DATA_PATH = Path("../data/adult.csv")

    # 1. Load dataset
    df = load_data(DATA_PATH)

    # 2. Inspect dataset
    inspect_data(df)

    # 3. Clean dataset
    df = clean_data(df)

    # 4. Encode target variable
    df = encode_target(df)

    # 5. Automatically identify numerical and categorical columns
    numerical_cols, categorical_cols = identify_columns(
        df,
        target_col='income'
    )

    # 6. Build machine learning pipeline
    model = build_pipeline(
        numerical_cols,
        categorical_cols
    )

    # 7. Train model
    model, X_test, y_test = train_model(df, model)

    # 8. Evaluate model
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
