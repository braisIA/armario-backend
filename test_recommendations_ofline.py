import os
import pickle
import itertools

from recommendation_service import (
    create_features_for_prediction,
    compute_rule_bonus,
)

# Rutas de modelo y preprocesador
BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, "ml", "models", "recommendation_model.pkl")
PREPROCESSOR_FILE = os.path.join(BASE_DIR, "ml", "models", "preprocessor.pkl")


# --- Clase sencilla para simular Garment (sin base de datos) ---
class DummyGarment:
    def __init__(self, name, type_, primary_color, pattern, material, formality_level, type_group):
        self.name = name
        self.type = type_
        self.primary_color = primary_color
        self.pattern = pattern
        self.material = material
        self.formality_level = formality_level
        self.type_group = type_group


def load_model_and_preprocessor():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_FILE, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor


def run_scenario(scenario_name, user_data, context_data, tops, bottoms, shoes, model, preprocessor):
    print("\n" + "=" * 70)
    print(f"ESCENARIO: {scenario_name}")
    print(f"Usuario: {user_data}")
    print(f"Contexto: {context_data}")
    print("-" * 70)

    results = []

    for top, bottom, shoe in itertools.product(tops, bottoms, shoes):
        df = create_features_for_prediction(user_data, context_data, top, bottom, shoe)
        X_proc = preprocessor.transform(df)
        ml_score = float(model.predict(X_proc)[0])
        rule_bonus = compute_rule_bonus(context_data, top, bottom, shoe)
        final_score = ml_score + rule_bonus

        results.append((final_score, ml_score, rule_bonus, top, bottom, shoe))

    # Ordenar por score final
    results.sort(key=lambda x: x[0], reverse=True)

    print("TOP 3 OUTFITS\n")
    for i, (final_s, ml_s, rb_s, top, bottom, shoe) in enumerate(results[:3], start=1):
        print(f" #{i}")
        print(f"   Top:    {top.name}  ({top.type}, {top.primary_color}, {top.material})")
        print(f"   Bottom: {bottom.name}  ({bottom.type}, {bottom.primary_color}, {bottom.material})")
        print(f"   Shoes:  {shoe.name}  ({shoe.type}, {shoe.primary_color}, {shoe.material})")
        print(f"   ML score   = {ml_s:.3f}")
        print(f"   Rule bonus = {rb_s:.3f}")
        print(f"   FINAL      = {final_s:.3f}")
        print("")


def main():
    # 1. Cargar modelo + preprocesador
    model, preprocessor = load_model_and_preprocessor()

    # 2. Definir algunos usuarios de prueba
    user_romantica = {
        "age": 25,
        "body_shape": "reloj_de_arena",
        "style_preference": "romantico",
        "gender": "mujer",
        "skin_tone": "cálida",
    }

    user_urbano = {
        "age": 30,
        "body_shape": "rectangulo",
        "style_preference": "urbano",
        "gender": "hombre",
        "skin_tone": "fría",
    }

    user_minimal = {
        "age": 40,
        "body_shape": "triangulo_invertido",
        "style_preference": "minimalista",
        "gender": "mujer",
        "skin_tone": "neutra",
    }

    # 3. Contextos de prueba
    contexto_cita_invierno = {
        "season": "invierno",
        "weather_condition": "frio",
        "temperature": 7.0,
        "activity": "cita_romantica",
        "mood": "romantico",
        "formality_level": 4,
    }

    contexto_gym_verano = {
        "season": "verano",
        "weather_condition": "caluroso",
        "temperature": 32.0,
        "activity": "deporte",
        "mood": "energico",
        "formality_level": 1,
    }

    contexto_oficina_primavera = {
        "season": "primavera",
        "weather_condition": "templado",
        "temperature": 20.0,
        "activity": "trabajo_oficina",
        "mood": "formal",
        "formality_level": 3,
    }

    # 4. Armarios MUY pequeños para entender el comportamiento

    # --- Armario para la usuaria romántica (cita) ---
    tops_romantica = [
        DummyGarment("blusa encaje marfil", "top_encaje", "blanco", "liso", "seda", 4, "top"),
        DummyGarment("jersey punto rosa", "jersey", "rosa", "punto", "lana", 3, "top"),
    ]
    bottoms_romantica = [
        DummyGarment("falda midi vaporosa", "falda_midi", "rosa", "liso", "lino", 3, "bottom"),
        DummyGarment("pantalon sastre negro", "pantalones_sastre", "negro", "liso", "poliester", 4, "bottom"),
    ]
    shoes_romantica = [
        DummyGarment("pumps negros", "pumps", "negro", "liso", "cuero", 4, "shoes"),
        DummyGarment("zapatillas blancas", "sneakers", "blanco", "liso", "sintetico", 1, "shoes"),
    ]

    # --- Armario para el chico urbano (gym/verano) ---
    tops_urbano = [
        DummyGarment("camiseta tecnica gris", "camiseta", "gris", "liso", "algodon", 1, "top"),
        DummyGarment("hoodie negro oversize", "hoodie", "negro", "liso", "algodon", 2, "top"),
    ]
    bottoms_urbano = [
        DummyGarment("short deportivo negro", "pantalon_corto", "negro", "liso", "poliester", 1, "bottom"),
        DummyGarment("jeans rectos azul", "jeans_rectos", "azul", "liso", "jean", 2, "bottom"),
    ]
    shoes_urbano = [
        DummyGarment("zapatillas running", "sneakers", "blanco", "liso", "sintetico", 1, "shoes"),
        DummyGarment("mocasines marrones", "mocasines", "marron", "liso", "cuero", 3, "shoes"),
    ]

    # --- Armario para la minimalista (oficina) ---
    tops_minimal = [
        DummyGarment("camisa blanca", "camisa", "blanco", "liso", "algodon", 4, "top"),
        DummyGarment("top liso beige", "top_liso", "beige", "liso", "algodon", 3, "top"),
    ]
    bottoms_minimal = [
        DummyGarment("pantalon sastre negro", "pantalones_sastre", "negro", "liso", "poliester", 4, "bottom"),
        DummyGarment("jeans azul oscuro", "jeans", "azul_marino", "liso", "jean", 2, "bottom"),
    ]
    shoes_minimal = [
        DummyGarment("zapatos de vestir negros", "zapatos", "negro", "liso", "cuero", 4, "shoes"),
        DummyGarment("sneakers blancas", "sneakers", "blanco", "liso", "sintetico", 1, "shoes"),
    ]

    # 5. Ejecutar escenarios
    run_scenario(
        "Cita romántica en invierno (usuaria romántica)",
        user_romantica,
        contexto_cita_invierno,
        tops_romantica,
        bottoms_romantica,
        shoes_romantica,
        model,
        preprocessor,
    )

    run_scenario(
        "Verano y gimnasio (usuario urbano)",
        user_urbano,
        contexto_gym_verano,
        tops_urbano,
        bottoms_urbano,
        shoes_urbano,
        model,
        preprocessor,
    )

    run_scenario(
        "Oficina en primavera (usuaria minimalista)",
        user_minimal,
        contexto_oficina_primavera,
        tops_minimal,
        bottoms_minimal,
        shoes_minimal,
        model,
        preprocessor,
    )


if __name__ == "__main__":
    main()
