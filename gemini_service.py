# backend/gemini_service.py
import os
import json
import google.generativeai as genai

# ==========================
#  Configuración de Gemini
# ==========================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Si quieres que no pete al arrancar, puedes cambiar esto por un simple print
    raise RuntimeError(
        "La variable de entorno GEMINI_API_KEY no está definida. "
        "Ponla en el .env de backend."
    )

genai.configure(api_key=GEMINI_API_KEY)

# 👈 OJO: nombre EXACTO según test_gemini_models.py
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

# Contexto por defecto si Gemini falla
DEFAULT_CONTEXT = {
    "season": "primavera",
    "weather_condition": "templado",
    "temperature": 20.0,
    "activity": "casual_paseo",
    "mood": "relajado",
    "formality_level": 3,
    "avoid_items": [],  # 👈 importante para no romper .copy()
}

# ==========================
#  Helpers internos
# ==========================

def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _normalize_context(data: dict) -> dict:
    """
    Se asegura de que el contexto tenga TODAS las claves
    con tipos correctos y valores razonables.
    """
    season = (data.get("season") or DEFAULT_CONTEXT["season"]).lower()
    weather = (data.get("weather_condition") or DEFAULT_CONTEXT["weather_condition"]).lower()
    activity = (data.get("activity") or DEFAULT_CONTEXT["activity"]).lower()
    mood = (data.get("mood") or DEFAULT_CONTEXT["mood"]).lower()

    temperature = _safe_float(data.get("temperature"), DEFAULT_CONTEXT["temperature"])
    formality_level = _safe_int(
        data.get("formality_level"), DEFAULT_CONTEXT["formality_level"]
    )

    # Acotar formality_level entre 1 y 5
    formality_level = max(1, min(5, formality_level))

    # Normalizar avoid_items
    avoid_items = data.get("avoid_items") or []
    if not isinstance(avoid_items, list):
        avoid_items = [str(avoid_items)]
    avoid_items = [str(x).lower() for x in avoid_items]

    return {
        "season": season,
        "weather_condition": weather,
        "temperature": temperature,
        "activity": activity,
        "mood": mood,
        "formality_level": formality_level,
        "avoid_items": avoid_items,
    }


def _extract_json_from_text(text: str) -> dict:
    """
    Gemini a veces rodea el JSON con texto. Aquí buscamos la primera
    '{' y la última '}' y hacemos json.loads sobre ese fragmento.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No se encontró un JSON válido en la respuesta de Gemini.")

    json_str = text[start:end]
    return json.loads(json_str)


def _is_greeting_only(message: str) -> bool:
    """
    Devuelve True si el mensaje parece solo un saludo / charla ligera
    y NO contiene nada de contexto de ropa, clima o actividad.
    Ejemplos: 'hola', 'buenas', 'qué tal', 'hola buenas', etc.
    """
    if not message:
        return False

    m = message.strip().lower()

    # Si el mensaje es muy largo, probablemente ya no sea solo un saludo
    if len(m) > 60:
        return False

    greeting_keywords = [
        "hola",
        "buenas",
        "buenos dias",
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "hey",
        "qué tal",
        "que tal",
        "hola buenas",
    ]

    # Palabras que indican que ya hay intención de outfit / clima / actividad
    fashion_keywords = [
        "ropa", "vestir", "vestirme", "vestido", "outfit", "look",
        "pondria", "ponerme", "que me pongo", "qué me pongo",
        "cita", "boda", "fiesta", "trabajo", "oficina",
        "gym", "gimnasio", "deporte", "salir", "quedar",
        "frio", "frío", "calor", "lluvia", "lloviendo", "nieve",
    ]

    if any(fk in m for fk in fashion_keywords):
        return False

    # Si contiene alguna palabra de saludo y nada de lo anterior → saludo simple
    return any(gk in m for gk in greeting_keywords)


# ==========================
# 1) Extraer contexto
# ==========================

def extract_context_from_message(message: str) -> dict:
    """
    Usa Gemini para extraer el contexto (season, weather, temperatura, etc.)
    de CUALQUIER frase que escriba el usuario.

    Devuelve SIEMPRE un diccionario con las claves:
      season, weather_condition, temperature, activity, mood,
      formality_level, avoid_items

    Si el mensaje es solo un saludo (sin info de outfit) → NO se llama a Gemini
    y devolvemos un contexto neutro con activity="saludo".
    Si ocurre cualquier error → devuelve DEFAULT_CONTEXT.
    """
    # 1) Caso especial: saludo sin contexto
    if _is_greeting_only(message):
        print("[GEMINI] Mensaje detectado como SALUDO simple. No se extrae contexto de outfit.")
        context = DEFAULT_CONTEXT.copy()
        # Marcamos explícitamente que esto viene de un saludo
        context["activity"] = "saludo"
        context["mood"] = "relajado"
        context["avoid_items"] = []
        return context

    # 2) Resto de mensajes → usamos Gemini normalmente
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY no configurada. Usando contexto por defecto.")
        return DEFAULT_CONTEXT.copy()

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    prompt = f"""
Eres un extractor de contexto para una app de recomendación de ropa.

Lee el MENSAJE DEL USUARIO y devuelve SOLO un JSON válido (sin texto extra)
con estas claves EXACTAS:

- season: una de ["primavera","verano","otoño","invierno"]
- weather_condition: por ejemplo "soleado", "lluvioso", "nublado", "nevando", "templado", etc.
- temperature: número (en grados Celsius, puede ser estimado)
- activity: por ejemplo "trabajo_oficina", "casual_paseo", "fiesta_nocturna", "boda_formal",
            "deporte", "gimnasio", "cita_romantica", etc.
- mood: por ejemplo "relajado","energico","formal","creativo","elegante","informal", etc.
- formality_level: número entero 1–5 (1 muy informal, 5 muy formal).
- avoid_items: lista de palabras o frases cortas que indiquen prendas, colores o materiales
               que el usuario NO quiere usar en este outfit.
               Ejemplos: ["vaqueros","chaqueta de cuero","rojo"].

Importante:
- SIEMPRE responde SOLO con un JSON. Nada de explicaciones.
- Si el usuario dice cosas como "no quiero vaqueros", "quita la chaqueta de cuero",
  añádelas a avoid_items.
- Si no hay nada que evitar, usa: "avoid_items": [].

MENSAJE DEL USUARIO:
\"\"\"{message}\"\"\""""

    try:
        response = model.generate_content(prompt)
        response_text = response.text or ""

        data = _extract_json_from_text(response_text)
        context = _normalize_context(data)

        print(f"[GEMINI] Contexto extraído: {context}")
        return context

    except Exception as e:
        print(f"Error al extraer contexto con Gemini: {e}")
        # Fallback para que la app nunca reviente
        return DEFAULT_CONTEXT.copy()


# ==========================
# 2) Mensaje “inteligente” del asistente
# ==========================

def generate_assistant_message(
    user_message: str,
    context_data: dict,
    outfit: dict | None,
    predicted_rating: float | None,
) -> str:
    """
    Genera el texto de respuesta del chatbot.
    - Si es solo un saludo -> respondemos nosotros (sin Gemini y sin outfit).
    - Si hay Gemini -> SOLO habla Gemini (no añadimos texto prefijado).
    - Si NO hay Gemini o falla -> mensaje neutro, sin construir nosotros una "solución".
    """

    # 0) Si es solo un saludo → respondemos nosotros, sin Gemini y sin outfit
    if _is_greeting_only(user_message):
        return (
            "¡Hola! 😊 Soy tu asistente de armario.\n"
            "Cuéntame: ¿para qué ocasión necesitas un look hoy? "
            "(trabajo, cita, gimnasio, fiesta, paseo, etc.)"
        )

    # 1) Si NO hay API de Gemini → respuesta neutra, sin proponer outfit nosotros
    if not GEMINI_API_KEY:
        return (
            "Ahora mismo no puedo usar el asistente de texto avanzado, "
            "pero el sistema sigue pudiendo proponerte combinaciones de ropa. "
            "Dime para qué ocasión y qué clima hace, y busco algo que encaje."
        )

    # 2) Gemini disponible → él se encarga de TODO el texto al usuario
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    ctx_json = json.dumps(context_data, ensure_ascii=False)

    if outfit:
        top = outfit["top"]
        bottom = outfit["bottom"]
        shoes = outfit["shoes"]

        outfit_description = f"""
OUTFIT PROPUESTO:
- Top: {top.name} (tipo: {top.type}, color: {top.primary_color})
- Bottom: {bottom.name} (tipo: {bottom.type}, color: {bottom.primary_color})
- Shoes: {shoes.name} (tipo: {shoes.type}, color: {shoes.primary_color})
"""
    else:
        outfit_description = "OUTFIT PROPUESTO: ninguno (no se pudo generar)."

    rating_text = (
        f"Puntuación estimada del outfit (1–5): {predicted_rating:.2f}"
        if predicted_rating is not None
        else "Sin puntuación estimada."
    )

    prompt = f"""
Eres un asistente de moda amigable y directo para una app de armario inteligente.

El usuario ha escrito:
\"\"\"{user_message}\"\"\"


El sistema ha inferido este CONTEXTO (JSON):
{ctx_json}

{outfit_description}

{rating_text}

Tu tarea:
- Responde al usuario en ESPAÑOL.
- Usa un tono cercano y útil.
- Explica brevemente por qué el outfit encaja (o no) con el contexto: clima, actividad, formalidad, estilo, etc.
- Si el outfit es poco apropiado (ej. muy elegante para el gym o muy informal para una boda), coméntalo y da algún consejo.
- Máximo 6–7 líneas de texto.

Responde SOLO con el mensaje al usuario (sin JSON).
"""

    try:
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        print(f"[GEMINI] Mensaje asistente: {text}")
        return text
    except Exception as e:
        print(f"Error en generate_assistant_message: {e}")
        # 3) Si Gemini falla → mensaje neutro, sin inventar outfit ni explicación
        return (
            "Ha habido un problema al generar la explicación del outfit. "
            "Aun así, la combinación que ves está calculada por el modelo. "
            "Si quieres, vuelve a describir la ocasión y lo intentamos otra vez."
        )
