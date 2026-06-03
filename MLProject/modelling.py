import pandas as pd
import logging
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi logging

logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s'
)

# Gunakan database MLflow lokal

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_base_model():

    # Load dataset
    df = pd.read_csv("MLProject/dataset_preprocessing.csv")

    # Pisahkan fitur dan target
    X = df.drop(columns=['Response'])
    y = df['Response']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Set experiment
    mlflow.set_experiment("Base_Model_Experiment")

    # Mulai MLflow run
    with mlflow.start_run(run_name="RandomForest_Base"):

        logging.info("Memulai training model RandomForest...")

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Hitung accuracy
        acc = accuracy_score(y_test, y_pred)

        logging.info(f"Akurasi model: {acc:.4f}")

        # Logging parameter
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Logging metric
        mlflow.log_metric("accuracy", acc)

        # Logging model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model"
        )

        logging.info("Model berhasil dicatat ke MLflow!")


if __name__ == "__main__":
    train_base_model()
