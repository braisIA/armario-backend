import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance


# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "full_training_dataset.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "recommendation_model.pkl")
PREPROCESSOR_FILE = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# --- CARGAR MODELO Y DATOS ---
print("Cargando modelo y datos...")
model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)
df = pd.read_csv(DATA_FILE)


# --- APLICAR PREPROCESADOR ---
print("Transformando datos...")
from train_recommendation_model import create_features
df = create_features(df)
X = preprocessor.transform(df.drop(columns=["rating"], errors="ignore"))

# --- IMPORTANCIA DE VARIABLES ---
print("Mostrando importancia de características...")
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=20)
plt.title("Importancia de características (XGBoost)")
plt.show()

# --- DISTRIBUCIÓN DE PREDICCIONES ---
print("Generando predicciones...")
y_pred = model.predict(X)

plt.figure(figsize=(8, 5))
plt.hist(y_pred, bins=30, edgecolor='black')
plt.title("Distribución de ratings predichos")
plt.xlabel("Predicted rating")
plt.ylabel("Frecuencia")
plt.show()

# --- OPCIONAL: Comparar con valores reales ---
if "rating" in df.columns:
    plt.figure(figsize=(6, 6))
    plt.scatter(df["rating"], y_pred, alpha=0.3)
    plt.xlabel("Rating real")
    plt.ylabel("Rating predicho")
    plt.title("Predicho vs Real")
    plt.grid(True)
    plt.show()

print("Visualización completada.")


