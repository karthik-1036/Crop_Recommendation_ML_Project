# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_model(data_path: str, model_path: str = "model.pkl"):
    """
    Trains a RandomForestClassifier on the crop dataset and saves the model.
    """

    # 1. Load the dataset
    df = pd.read_csv(data_path)
    
    # 2. Separate features and labels
    X = df.drop('label', axis=1)  # all columns except target
    y = df['label']               # target column

    # 3. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Make predictions and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the model
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to: {model_path}")
