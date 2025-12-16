# ======================================================
#  RECOMMENDATION SERVICE - VERSION TFG 2025
# ======================================================

import pandas as pd
from models import Garment


# ======================================================
# 1) FEATURE ENGINEERING CONSISTENTE CON ENTRENAMIENTO
# ======================================================

def _compute_style_match(row):
    style = str(row['style_preference']).lower()

    STYLE_TYPE_PREFS = {
        'clasico': {
            'types': ['camisa', 'blusa', 'blazer', 'top_liso', 'pantalon_sastre', 'vestido'],
            'colors': ['negro', 'blanco', 'gris', 'beige', 'azul_marino'],
        },
        'urbano': {
            'types': ['hoodie', 'bomber', 'camiseta_grafica', 'croptop', 'jeans', 'leggings'],
            'colors': ['negro', 'blanco', 'rojo', 'azul', 'verde'],
        },
        'bohemio': {
            'types': ['top_etnico', 'top_bordado', 'vestido_floral', 'blusa_volantes'],
            'colors': ['beige', 'marron', 'mostaza', 'verde_oliva'],
        },
        'minimalista': {
            'types': ['top_liso', 'camisa', 'blazer_negro'],
            'colors': ['negro', 'blanco', 'gris', 'beige'],
        },
        'romantico': {
            'types': ['top_encaje', 'vestido_floral', 'blusa_volantes'],
            'colors': ['rosa', 'rojo', 'lavanda'],
        },
    }

    prefs = STYLE_TYPE_PREFS.get(style, None)
    if prefs is None:
        return 0.5

    def garment_ok(t, col):
        return 1 if (t in prefs['types'] or col in prefs['colors']) else 0

    m_top = garment_ok(str(row['top_type']), str(row['top_primary_color']))
    m_bottom = garment_ok(str(row['bottom_type']), str(row['bottom_primary_color']))
    m_shoes = garment_ok(str(row['shoes_type']), str(row['shoes_primary_color']))

    return (m_top + m_bottom + m_shoes) / 3.0


def check_seasonal_appropriateness(row):
    season = (row.get("season") or "").lower()
    mats = [str(row.get(k, "")).lower() for k in
            ["top_material", "bottom_material", "shoes_material"]]

    summer = ['lino', 'algodon', 'sintetico', 'poliester', 'seda']
    winter = ['lana', 'terciopelo', 'tweed']

    if season == "verano":
        return sum(m in summer for m in mats) / len(mats)
    if season == "invierno":
        return sum(m in winter for m in mats) / len(mats)
    return 0.5


def check_temp_material_match(row):
    temp = row.get("temperature")
    mats = [str(row.get(k, "")).lower() for k in
            ["top_material", "bottom_material", "shoes_material"]]

    hot = ['lino', 'algodon', 'sintetico', 'poliester', 'seda']
    cold = ['lana', 'terciopelo', 'tweed']

    if temp is None:
        return 0.5
    if temp > 25:
        return sum(m in hot for m in mats) / len(mats)
    if temp < 10:
        return sum(m in cold for m in mats) / len(mats)
    return 0.5


def create_features_for_prediction(user, context, top, bottom, shoes):
    df = pd.DataFrame([{
        # usuario
        'age': user.get('age'),
        'body_shape': user.get('body_shape'),
        'style_preference': user.get('style_preference'),
        'gender': user.get('gender'),
        'skin_tone': user.get('skin_tone'),

        # contexto
        'season': context.get('season'),
        'weather_condition': context.get('weather_condition'),
        'temperature': context.get('temperature'),
        'activity': context.get('activity'),
        'mood': context.get('mood'),
        'formality_level': context.get('formality_level'),

        # prendas
        'top_type': top.type,
        'top_primary_color': top.primary_color,
        'top_pattern': top.pattern,
        'top_material': top.material,
        'top_formality_level': top.formality_level,

        'bottom_type': bottom.type,
        'bottom_primary_color': bottom.primary_color,
        'bottom_pattern': bottom.pattern,
        'bottom_material': bottom.material,
        'bottom_formality_level': bottom.formality_level,

        'shoes_type': shoes.type,
        'shoes_primary_color': shoes.primary_color,
        'shoes_pattern': shoes.pattern,
        'shoes_material': shoes.material,
        'shoes_formality_level': shoes.formality_level,
    }])

    # mismos features del entrenamiento
    df["color_match"] = (df["top_primary_color"] == df["bottom_primary_color"]).astype(int)
    df["material_match"] = (df["top_material"] == df["bottom_material"]).astype(int)
    df["pattern_match"] = (df["top_pattern"] == df["bottom_pattern"]).astype(int)

    df["formality_diff_top_bottom"] = abs(df["top_formality_level"] - df["bottom_formality_level"])
    df["formality_diff_top_shoes"] = abs(df["top_formality_level"] - df["shoes_formality_level"])
    df["formality_diff_bottom_shoes"] = abs(df["bottom_formality_level"] - df["shoes_formality_level"])

    df["style_match"] = df.apply(_compute_style_match, axis=1)
    df["seasonal_appropriateness"] = df.apply(check_seasonal_appropriateness, axis=1)
    df["temp_material_match"] = df.apply(check_temp_material_match, axis=1)

    return df


# ======================================================
# 2) DETECTORES DE TIPO DE PRENDA
# ======================================================

def _f(g):   # simple getter formalidad
    try:
        return int(g.formality_level or 3)
    except Exception:
        return 3


def _sport_shoe(g):
    return any(k in (g.type or "").lower() for k in ["sneaker", "zapatilla", "running", "trail", "gym"])


def _elegant_shoe(g):
    return any(k in (g.type or "").lower() for k in [
        "pumps", "tacon", "tacón", "mocasin", "mocasines", "loafers",
        "bailarina", "bailarinas", "botin", "botines"
    ])


def _boot(g):
    t = (g.type or "").lower()
    return "bota" in t or "boot" in t


def _sandal(g):
    return "sandalia" in (g.type or "").lower()


def _very_warm(m):
    return any(k in (m or "").lower() for k in ["lana", "tweed", "terciopelo"])


def _very_cool(m):
    return any(k in (m or "").lower() for k in ["lino", "algodon", "seda"])


def _sport_bottom(g):
    return any(k in (g.type or "").lower() for k in ["jogger", "chandal", "leggings", "malla", "short"])


def _jeans(g):
    return any(k in (g.type or "").lower() for k in ["jean", "vaquero", "denim"])


# ======================================================
# 2.5) FILTRO DE PRENDAS A EVITAR (avoid_items)
# ======================================================

def _violates_avoid(garment: Garment, avoid_items: list[str]) -> bool:
    """
    Devuelve True si la prenda contiene alguna palabra/frase de avoid_items
    en su nombre, tipo, color, patrón o material.
    """
    if not avoid_items:
        return False

    text = " ".join([
        str(garment.name or "").lower(),
        str(garment.type or "").lower(),
        str(garment.primary_color or "").lower(),
        str(garment.pattern or "").lower(),
        str(garment.material or "").lower(),
    ])

    return any(term in text for term in avoid_items)


# ======================================================
# 3) BONUS / PENALIZACIONES PROFESIONALES 2025
# ======================================================

def compute_rule_bonus(context, top, bottom, shoes):
    activity = (context.get("activity") or "").lower()
    season = (context.get("season") or "").lower()
    temp = float(context.get("temperature") or 20)

    ftop, fbot, fsho = _f(top), _f(bottom), _f(shoes)
    avg = (ftop + fbot + fsho) / 3

    # categorías normalizadas
    is_sport = activity in ["deporte", "gimnasio", "gym", "running"]
    is_date = activity in ["cita_romantica", "cita"]
    is_office = activity == "trabajo_oficina"
    is_formal = activity == "evento_formal"
    is_party = activity == "fiesta_nocturna"
    is_home = activity == "relax_en_casa"
    is_casual = not (is_sport or is_date or is_office or is_formal or is_party or is_home)

    bonus = 0.0

    # ------------------------------------
    # FORMALIDAD GLOBAL
    # ------------------------------------
    MINMAX = {
        'sport': (1, 2.2),
        'home': (1, 2.2),
        'date': (3.2, 4.8),
        'office': (2.8, 4.2),
        'formal': (3.8, 5),
        'party': (2.5, 4.5),
        'casual': (1.8, 3.2),
    }

    if is_sport:
        low, high = MINMAX['sport']
    elif is_date:
        low, high = MINMAX['date']
    elif is_office:
        low, high = MINMAX['office']
    elif is_formal:
        low, high = MINMAX['formal']
    elif is_party:
        low, high = MINMAX['party']
    elif is_home:
        low, high = MINMAX['home']
    else:
        low, high = MINMAX['casual']

    if avg < low:
        bonus -= (low - avg) * 2
    elif avg > high:
        bonus -= (avg - high) * 2
    else:
        bonus += 1

    # ------------------------------------
    # CALZADO
    # ------------------------------------
    if is_sport:
        if _sport_shoe(shoes):
            bonus += 4
        else:
            bonus -= 5

    elif is_date:
        if _elegant_shoe(shoes):
            bonus += 3
        if _sport_shoe(shoes):
            bonus -= 4
        if _sandal(shoes) and (season == 'invierno' or temp < 8):
            bonus -= 3

    elif is_office or is_formal:
        if fsho >= 4:
            bonus += 2
        if _sport_shoe(shoes):
            bonus -= 3

    elif is_party:
        if _sport_shoe(shoes):
            bonus -= 1
        if _elegant_shoe(shoes):
            bonus += 1

    # ------------------------------------
    # PANTALÓN
    # ------------------------------------
    if is_sport:
        if _sport_bottom(bottom):
            bonus += 3
        if _jeans(bottom):
            bonus -= 4

    if is_formal and _sport_bottom(bottom):
        bonus -= 3

    # ------------------------------------
    # CLIMA
    # ------------------------------------
    # calor fuerte
    if temp >= 28:
        if _very_warm(top.material):
            bonus -= 2
        if _very_warm(bottom.material):
            bonus -= 2
        if _boot(shoes):
            bonus -= 2

    # frío fuerte
    if temp <= 8:
        if _sandal(shoes):
            bonus -= 4
        if bottom.type and bottom.type.lower() in ["short", "falda"]:
            bonus -= 3

    # ------------------------------------
    # ARMONÍA COLOR/MATERIAL
    # ------------------------------------
    if top.primary_color == bottom.primary_color:
        bonus += 0.4
    if bottom.primary_color == shoes.primary_color:
        bonus += 0.4
    if top.material == bottom.material:
        bonus += 0.3

    # ------------------------------------
    # CLAMP FINAL
    # ------------------------------------
    return max(-6.0, min(4.0, bonus))


# ======================================================
# 4) MOTOR DE RECOMENDACIÓN
# ======================================================

def get_recommendation(user_id, context_data, user_data, model, preprocessor, avoid_items=None):
    """
    Genera la mejor combinación top+bottom+shoes para un usuario y contexto,
    respetando la lista de elementos a evitar (avoid_items).
    """
    # seguridad: si no hay modelo o preprocesador, no hacemos nada
    if not model or not preprocessor:
        return None

    # recoger avoid_items: prioridad parámetro, luego contexto
    if avoid_items is None:
        avoid_items = context_data.get("avoid_items", [])
    if not isinstance(avoid_items, list):
        avoid_items = [str(avoid_items)]
    avoid_items = [str(x).lower() for x in avoid_items]

    garments = Garment.query.filter_by(user_id=user_id).all()
    tops = [g for g in garments if g.type_group == 'top']
    bottoms = [g for g in garments if g.type_group == 'bottom']
    shoes = [g for g in garments if g.type_group == 'shoes']

    if not tops or not bottoms or not shoes:
        return None

    best = None
    best_score = float("-inf")
    combos = []
    skipped_avoid = 0

    for t in tops:
        for b in bottoms:
            for s in shoes:
                # -----------------------
                # FILTRO: avoid_items
                # -----------------------
                if avoid_items:
                    if (_violates_avoid(t, avoid_items) or
                        _violates_avoid(b, avoid_items) or
                        _violates_avoid(s, avoid_items)):
                        skipped_avoid += 1
                        continue

                df = create_features_for_prediction(user_data, context_data, t, b, s)
                pred = float(model.predict(preprocessor.transform(df))[0])
                rules = compute_rule_bonus(context_data, t, b, s)

                final = pred + rules
                combos.append((t, b, s, pred, rules, final))

                if final > best_score:
                    best_score = final
                    best = {"top": t, "bottom": b, "shoes": s}

    combos.sort(key=lambda x: x[5], reverse=True)

    print("\n[RECO] Top 3 combinaciones (Final = ML + Rules):")
    for i, (t, b, s, p, r, f) in enumerate(combos[:3], 1):
        print(f" #{i} → {t.name} + {b.name} + {s.name} | ML={p:.2f} | R={r:.2f} | Final={f:.2f}")

    if avoid_items:
        print(f"[RECO] Combinaciones descartadas por avoid_items={avoid_items}: {skipped_avoid}")

    return best, best_score
