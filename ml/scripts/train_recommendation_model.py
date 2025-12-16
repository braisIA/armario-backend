import pandas as pd
import numpy as np
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Librerías de Machine Learning ---
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Librerías de optimización y modelos avanzados
import optuna
import lightgbm as lgb
import xgboost as xgb

# Ignorar advertencias para una salida limpia
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'full_training_dataset.csv')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'recommendation_model.pkl')
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.pkl')
BEST_PARAMS_FILE = os.path.join(MODEL_DIR, 'best_params.json')


# =============================
# 1. INGENIERÍA DE CARACTERÍSTICAS
# =============================

def _compute_style_match(row):
    """
    Devuelve un valor entre 0 y 1 que indica cuánto encaja el outfit
    con el estilo declarado por el usuario (`style_preference`),
    usando tipos y colores de las prendas.
    """
    style = str(row['style_preference'])

    top_type = str(row['top_type'])
    bottom_type = str(row['bottom_type'])
    shoes_type = str(row['shoes_type'])

    top_color = str(row['top_primary_color'])
    bottom_color = str(row['bottom_primary_color'])
    shoes_color = str(row['shoes_primary_color'])

    STYLE_TYPE_PREFS = {
        'clasico': {
            'types': [
                'camisa', 'blusa', 'blazer', 'blazer_estructurado',
                'top_liso', 'top_estructurado', 'pantalon_sastre', 'vestido'
            ],
            'colors': ['negro', 'blanco', 'gris', 'beige', 'azul_marino']
        },
        'urbano': {
            'types': [
                'hoodie', 'bomber', 'camiseta_grafica', 'camiseta_oversized',
                'croptop', 'vaqueros', 'jeans', 'leggings', 'zapatillas'
            ],
            'colors': ['negro', 'blanco', 'rojo', 'azul', 'verde']
        },
        'bohemio': {
            'types': [
                'top_etnico', 'top_bordado', 'vestido_floral',
                'blusa_volantes', 'top_suave'
            ],
            'colors': ['beige', 'marron', 'mostaza', 'verde_oliva', 'burdeos']
        },
        'minimalista': {
            'types': [
                'top_liso', 'camisa', 'blazer_negro', 'vestido'
            ],
            'colors': ['negro', 'blanco', 'gris', 'beige']
        },
        'romantico': {
            'types': [
                'top_encaje', 'vestido', 'vestido_floral',
                'blusa_volantes', 'top_suave'
            ],
            'colors': ['rosa', 'rojo', 'blanco', 'lavanda']
        },
    }

    prefs = STYLE_TYPE_PREFS.get(style, None)
    if prefs is None:
        return 0.5  # neutro si el estilo no está mapeado

    def garment_match(g_type, g_color):
        m_type = g_type in prefs['types']
        m_color = g_color in prefs['colors']
        return 1 if (m_type or m_color) else 0

    matches = [
        garment_match(top_type, top_color),
        garment_match(bottom_type, bottom_color),
        garment_match(shoes_type, shoes_color),
    ]

    return sum(matches) / len(matches)


def _check_seasonal_appropriateness(row):
    """Verifica si los materiales son apropiados para la temporada."""
    season = row['season']
    materials = [
        str(row['top_material']).lower(),
        str(row['bottom_material']).lower(),
        str(row['shoes_material']).lower()
    ]

    summer_materials = ['lino', 'algodon', 'sintetico', 'poliester', 'seda']
    winter_materials = ['lana', 'terciopelo', 'tweed']

    if season == 'verano':
        return sum(1 for m in materials if m in summer_materials) / len(materials)
    elif season == 'invierno':
        return sum(1 for m in materials if m in winter_materials) / len(materials)
    else:  # primavera, otoño
        return 0.5


def _check_temp_material_match(row):
    """Verifica si los materiales son apropiados para la temperatura."""
    temp = row['temperature']
    materials = [
        str(row['top_material']).lower(),
        str(row['bottom_material']).lower(),
        str(row['shoes_material']).lower()
    ]

    hot_materials = ['lino', 'algodon', 'sintetico', 'poliester', 'seda']
    cold_materials = ['lana', 'terciopelo', 'tweed']

    if temp > 25:  # Caliente
        return sum(1 for m in materials if m in hot_materials) / len(materials)
    elif temp < 10:  # Frío
        return sum(1 for m in materials if m in cold_materials) / len(materials)
    else:  # Templado
        return 0.5


def create_features(df):
    """
    Crea características avanzadas a partir del DataFrame original.
    - Interacción entre prendas.
    - Compatibilidad (temporada, material, estilo).
    """
    print("1.1. Creando características avanzadas...")

    text_cols = ['top_description', 'bottom_description', 'shoes_description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Interacción
    df['color_match'] = (df['top_primary_color'] == df['bottom_primary_color']).astype(int)
    df['material_match'] = (df['top_material'] == df['bottom_material']).astype(int)
    df['pattern_match'] = (df['top_pattern'] == df['bottom_pattern']).astype(int)

    # Diferencias
    df['formality_diff_top_bottom'] = abs(df['top_formality_level'] - df['bottom_formality_level'])
    df['formality_diff_top_shoes'] = abs(df['top_formality_level'] - df['shoes_formality_level'])
    df['formality_diff_bottom_shoes'] = abs(df['bottom_formality_level'] - df['shoes_formality_level'])

    # Estilo
    df['style_match'] = df.apply(_compute_style_match, axis=1)

    # Compatibilidad clima/temporada
    df['seasonal_appropriateness'] = df.apply(_check_seasonal_appropriateness, axis=1)
    df['temp_material_match'] = df.apply(_check_temp_material_match, axis=1)

    return df


# =============================
# 2. PREPROCESADOR
# =============================

def create_preprocessor():
    """
    Crea el preprocesador para diferentes tipos de características.
    """
    print("2. Creando el preprocesador de datos...")

    numeric_features = [
        'age', 'temperature',
        'top_formality_level', 'bottom_formality_level', 'shoes_formality_level',
        'color_match', 'material_match', 'pattern_match',
        'formality_diff_top_bottom', 'formality_diff_top_shoes', 'formality_diff_bottom_shoes',
        'style_match', 'seasonal_appropriateness', 'temp_material_match'
    ]

    categorical_features = [
        'body_shape', 'style_preference', 'gender', 'skin_tone',
        'season', 'weather_condition', 'activity', 'mood',
        'top_type', 'top_primary_color', 'top_pattern', 'top_material',
        'bottom_type', 'bottom_primary_color', 'bottom_pattern', 'bottom_material',
        'shoes_type', 'shoes_primary_color', 'shoes_pattern', 'shoes_material'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


# =============================
# 3. OBJETIVO OPTUNA
# =============================

def objective(trial, X_train, y_train, preprocessor):
    model_name = trial.suggest_categorical('model', ['lightgbm', 'xgboost', 'randomforest'])

    if model_name == 'lightgbm':
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        model = lgb.LGBMRegressor(**params)

    elif model_name == 'xgboost':
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)

    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)

    preproc_clone = clone(preprocessor)

    pipeline = Pipeline(steps=[
        ('preprocess', preproc_clone),
        ('model', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1
    )

    return -scores.mean()


# =============================
# 4. FUNCIÓN PRINCIPAL
# =============================

def train_model():
    print("--- Iniciando el Entrenamiento del Modelo de Recomendación ---")
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"❌ ERROR: El archivo de datos '{DATA_FILE}' no se encuentra.")
        return

    print("1. Cargando el dataset completo...")
    df = pd.read_csv(DATA_FILE)

    # Ingeniería de características
    df = create_features(df)

    # --- 1.b Quitar columnas de IDs / nombres que NO queremos usar como features ---
    cols_to_drop = [
        'id',
        'user_id',
        'username',
        'context_id',
        'top_garment_id',
        'bottom_garment_id',
        'shoes_garment_id'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Separar X / y
    X = df.drop('rating', axis=1)
    y = df['rating']
    print(f"   Dataset final con {X.shape[0]} muestras y {X.shape[1]} características.")

    # 2. Preprocesador base
    preprocessor_base = create_preprocessor()

    # 3. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

    # Baselines
    y_test_array = y_test.values
    mean_rating = y_train.mean()
    median_rating = y_train.median()

    y_pred_mean = np.full_like(y_test_array, fill_value=mean_rating, dtype=float)
    y_pred_median = np.full_like(y_test_array, fill_value=median_rating, dtype=float)

    mae_mean = mean_absolute_error(y_test_array, y_pred_mean)
    mae_median = mean_absolute_error(y_test_array, y_pred_median)
    mape_mean = mean_absolute_percentage_error(y_test_array, y_pred_mean)
    mape_median = mean_absolute_percentage_error(y_test_array, y_pred_median)

    print("\n   >>> Baselines (modelos muy simples, en TEST) <<<")
    print(f"   Baseline media  -> MAE = {mae_mean:.4f}, MAPE = {mape_mean:.4f}")
    print(f"   Baseline mediana-> MAE = {mae_median:.4f}, MAPE = {mape_median:.4f}")

    # 4. Optuna
    print("\n3. Optimizando hiperparámetros con Optuna (usando MAPE en CV)...")
    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, preprocessor_base),
        n_trials=50,
        timeout=600,
        n_jobs=1
    )

    best_model_name = study.best_trial.params['model']
    best_params = study.best_trial.params
    best_mape_cv = study.best_value
    print(f"\n   ¡Optimización completada!")
    print(f"   Mejor modelo encontrado: {best_model_name}")
    print(f"   Mejor MAPE (validación cruzada sobre TRAIN): {best_mape_cv:.4f}")
    print(f"   Mejores hiperparámetros: {best_params}")

    # 5. Evaluación en TEST
    print("\n4. Entrenando y evaluando el mejor modelo en el conjunto de prueba...")

    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if best_model_name == 'lightgbm':
        best_model = lgb.LGBMRegressor(**{k: v for k, v in best_params.items() if k != 'model'})
    elif best_model_name == 'xgboost':
        best_model = xgb.XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model'})
    else:
        best_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != 'model'})

    best_model.fit(X_train_proc, y_train)
    y_pred_test = best_model.predict(X_test_proc)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print("\n5. Métricas en TEST con el mejor modelo:")
    print(f"   MAE  (test): {mae_test:.4f}")
    print(f"   MAPE (test): {mape_test:.4f}")
    print(f"   RMSE (test): {rmse_test:.4f}")
    print(f"   R²   (test): {r2_test:.4f}")

    # 6. CV final sobre todo el dataset
    print("\n6. Validación cruzada FINAL sobre TODO el dataset con el mejor modelo...")

    preprocessor_cv = create_preprocessor()
    if best_model_name == 'lightgbm':
        model_for_cv = lgb.LGBMRegressor(**{k: v for k, v in best_params.items() if k != 'model'})
    elif best_model_name == 'xgboost':
        model_for_cv = xgb.XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model'})
    else:
        model_for_cv = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != 'model'})

    pipeline_final_cv = Pipeline(steps=[
        ('preprocess', preprocessor_cv),
        ('model', model_for_cv)
    ])

    cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_mape = cross_val_score(
        pipeline_final_cv, X, y,
        cv=cv_final,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1
    )
    scores_mae = cross_val_score(
        pipeline_final_cv, X, y,
        cv=cv_final,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    print(f"   MAPE medio (CV final sobre TODO el dataset): {-scores_mape.mean():.4f}")
    print(f"   MAE  medio (CV final sobre TODO el dataset): {-scores_mae.mean():.4f}")

    # 7. Entrenamiento final sobre TODO X,y y guardado
    print("\n7. Entrenando el modelo FINAL con TODO el dataset y guardando artefactos...")

    preprocessor_full = create_preprocessor()
    preprocessor_full.fit(X)
    X_full_proc = preprocessor_full.transform(X)

    if best_model_name == 'lightgbm':
        final_model = lgb.LGBMRegressor(**{k: v for k, v in best_params.items() if k != 'model'})
    elif best_model_name == 'xgboost':
        final_model = xgb.XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model'})
    else:
        final_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != 'model'})

    final_model.fit(X_full_proc, y)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(final_model, f)

    with open(PREPROCESSOR_FILE, 'wb') as f:
        pickle.dump(preprocessor_full, f)

    import json
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"\n--- Entrenamiento Completado ---")
    print(f"   Modelo ganador: {best_model_name}")
    print(f"   Mejor MAPE CV (train, Optuna): {best_mape_cv:.4f}")
    print(f"   MAPE test: {mape_test:.4f}")
    print(f"   MAPE CV final (TODO el dataset): {-scores_mape.mean():.4f}")
    print(f"   Modelo final guardado en: '{MODEL_FILE}'")
    print(f"   Preprocesador final guardado en: '{PREPROCESSOR_FILE}'")
    print(f"   Mejores hiperparámetros guardados en: '{BEST_PARAMS_FILE}'")
    print("\nListo para usar en tu servidor Flask con el modelo final entrenado en todo el dataset.")


if __name__ == '__main__':
    train_model()
