import os
import joblib
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

load_dotenv()

MODEL_PATH   = "ml/models/severity_rf_v1.pkl"
ENCODER_PATH = "ml/models/label_encoders.pkl"
TFIDF_PATH   = "ml/models/tfidf_vectorizer.pkl"

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}


def load_data():
    """Load incidents from PostgreSQL for training."""
    print("Loading incidents from database...")

    query = """
        SELECT 
            issue_type                                  AS category,
            product_area                                AS subcategory,
            priority,
            channel                                     AS contact_type,
            CASE WHEN reopened THEN 1 ELSE 0 END        AS reopen_count,
            COALESCE(ROUND(resolution_time_hours)::int,
                     0)                                 AS sys_mod_count,
            CASE WHEN sla_plan IS NOT NULL THEN 1
                 ELSE 0 END                             AS made_sla,
            CASE WHEN csat_score >= 4 THEN 1
                 ELSE 0 END                             AS knowledge,
            created_at                                  AS opened_at
        FROM incidents
        WHERE priority IS NOT NULL
        AND TRIM(priority) != ''
        AND TRIM(priority) != '?'
        AND priority IN ('low', 'medium', 'high', 'urgent', 'critical')
    """

    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(query, conn)

    print(f"Loaded {len(df)} incidents for training")
    print(f"Priority distribution:\n{df['priority'].value_counts()}")
    return df


def engineer_features(df):
    """Convert raw incident data into ML features."""
    print("\nEngineering features...")

    df = df.copy()

    # Time features
    df['opened_at']         = pd.to_datetime(df['opened_at'], errors='coerce')
    df['hour_of_day']       = df['opened_at'].dt.hour.fillna(12).astype(int)
    df['day_of_week']       = df['opened_at'].dt.dayofweek.fillna(0).astype(int)
    df['is_weekend']        = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)

    # Numeric features
    df['reassignment_count'] = 0  # not in new dataset, default to 0
    df['reopen_count']       = pd.to_numeric(df['reopen_count'], errors='coerce').fillna(0).astype(int)
    df['sys_mod_count']      = pd.to_numeric(df['sys_mod_count'], errors='coerce').fillna(0).astype(int)
    df['made_sla']           = df['made_sla'].fillna(0).astype(int)
    df['knowledge']          = df['knowledge'].fillna(0).astype(int)

    # Categorical features
    for col in ['category', 'subcategory', 'contact_type']:
        df[col] = df[col].fillna('unknown').astype(str).str.strip().str.lower()

    # Remove rare priority classes (less than 10 samples)
    class_counts = df['priority'].value_counts()
    rare_classes = class_counts[class_counts < 10].index.tolist()
    if rare_classes:
        print(f"Removing rare priority classes: {rare_classes}")
        df = df[~df['priority'].isin(rare_classes)]

    print(f"Final priority distribution:\n{df['priority'].value_counts()}")
    return df


def encode_features(df_train, df_test):
    """Encode categorical features using LabelEncoder."""
    print("\nEncoding categorical features...")

    encoders = {}
    categorical_cols = ['category', 'subcategory', 'contact_type']

    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col].astype(str))

        class_to_idx = {label: idx for idx, label in enumerate(le.classes_)}
        df_test[col + '_encoded'] = (
            df_test[col]
            .astype(str)
            .map(lambda v: class_to_idx.get(v, 0))
            .astype(int)
        )
        encoders[col] = le

    # Encode target
    priority_encoder = LabelEncoder()
    df_train['priority_encoded'] = priority_encoder.fit_transform(df_train['priority'].astype(str))

    priority_map = {label: idx for idx, label in enumerate(priority_encoder.classes_)}
    df_test['priority_encoded'] = (
        df_test['priority']
        .astype(str)
        .map(lambda v: priority_map.get(v, -1))
        .astype(int)
    )

    # Drop unseen test labels
    df_test = df_test[df_test['priority_encoded'] != -1].copy()

    encoders['priority'] = priority_encoder
    print(f"Priority classes: {priority_encoder.classes_}")

    return df_train, df_test, encoders


def prepare_xy(df):
    """Select final 12 features and target."""

    feature_cols = [
        'category_encoded',
        'subcategory_encoded',
        'contact_type_encoded',
        'reassignment_count',
        'reopen_count',
        'sys_mod_count',
        'made_sla',
        'knowledge',
        'hour_of_day',
        'day_of_week',
        'is_weekend',
        'is_business_hours',
    ]

    X = df[feature_cols]
    y = df['priority_encoded']

    print(f"Features: {len(feature_cols)}")
    print(f"Samples:  {len(X)}")

    return X, y, feature_cols


def train_model(X_train, y_train):
    """Train RandomForest classifier."""
    print("\nTraining Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("Training complete!")
    return model


def evaluate_model(model, X_test, y_test, encoders, feature_cols):
    """Evaluate model performance."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    labels    = list(range(len(encoders['priority'].classes_)))
    priority_classes = encoders['priority'].classes_

    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=priority_classes,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    os.makedirs("ml/models", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=priority_classes,
        yticklabels=priority_classes
    )
    plt.title('DecisionLens — Confusion Matrix')
    plt.ylabel('Actual Priority')
    plt.xlabel('Predicted Priority')
    plt.tight_layout()
    plt.savefig('ml/models/confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved to ml/models/confusion_matrix.png")

    # Feature importance
    importances = pd.DataFrame({
        'feature':    feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importances:")
    print(importances.to_string(index=False))

    return accuracy


def save_model(model, encoders):
    """Save trained model and encoders to disk."""
    os.makedirs("ml/models", exist_ok=True)

    joblib.dump(model,    MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    print(f"\nModel saved to:    {MODEL_PATH}")
    print(f"Encoders saved to: {ENCODER_PATH}")


def main():
    # Load data
    df = load_data()

    # Engineer features
    df = engineer_features(df)

    # Train/test split
    try:
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['priority']
        )
    except ValueError as e:
        print(f"Warning: stratified split failed ({e}), using random split")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Encode features
    df_train, df_test, encoders = encode_features(df_train, df_test)

    # Prepare X and y
    X_train, y_train, feature_cols = prepare_xy(df_train)
    X_test,  y_test,  _            = prepare_xy(df_test)

    print(f"\nTraining set: {len(X_train)} incidents")
    print(f"Test set:     {len(X_test)} incidents")

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test, encoders, feature_cols)

    # Save
    save_model(model, encoders)

    print("\n" + "="*50)
    print(f"DONE! Final accuracy: {accuracy:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()