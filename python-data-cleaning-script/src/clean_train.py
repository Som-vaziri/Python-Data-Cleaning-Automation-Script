import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Define feature groups
categorical_cols = [
    'workclass', 'education', 'marital.status',
    'occupation', 'relationship', 'race',
    'sex', 'native.country'
]

numerical_cols = [
    'age', 'fnlwgt', 'capital.gain',
    'capital.loss', 'hours.per.week'
]


def load_data(path):
    """Load dataset from CSV file."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df


def clean_data(df):
    """Clean dataset by handling spaces, missing values, and redundant columns."""
    
    # Remove leading/trailing whitespace from string values
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Replace '?' with NaN (common in Adult dataset)
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


def build_pipeline():
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
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate(model, X_test, y_test):
    """Evaluate model performance."""
    
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


def main():
    """Main execution pipeline."""
    
    DATA_PATH = Path("../data/adult.csv")  # Update this path if needed

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_target(df)

    model = build_pipeline()
    model, X_test, y_test = train_model(df, model)

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
