import pandas as pd
import random
from faker import Faker

# --- Semilla para reproducibilidad ---
fake = Faker('es_ES')
Faker.seed(42)
random.seed(42)

# --- 1. Universo de posibilidades ---
BODY_SHAPES = ['pera', 'manzana', 'reloj_de_arena', 'rectangulo', 'triangulo_invertido']
STYLE_PREFERENCES = ['minimalista', 'clasico', 'romantico', 'urbano', 'bohemio']

GARMENT_TYPES = [
    'camiseta', 'pantalones', 'vestido', 'blazer', 'falda', 'jersey', 'cazadora',
    'polo', 'camisa', 'monos', 'pantalon_corto', 'zapatos', 'botas', 'sneakers',
    # variantes usadas en wardrobes
    'falda_lapiz', 'blusa_volantes', 'top_encaje', 'vestido_floral', 'croptop',
    'hoodie', 'joggers', 'cargo', 'top_bordado', 'pantalones_flare', 'falda_larga',
    'top_liso', 'pantalones_sastre', 'blazer_negro', 'sandalias_cuadradas', 'mules',
    'blusa_ligera', 'shorts', 'pantalon_corto', 'top_lino', 'falda_midi', 'pumps',
    'loafers', 'bailarinas', 'sandalias_finas', 'sneakers_chunky', 'botas_planas',
    'blusa_impermeable', 'botas_impermeables', 'pantalones_cargo', 'jeans_rectos',
    'jeans', 'jeans_grueso', 'pantalones_sastre', 'pantalon_corto'
]

COLORS = [
    'negro', 'blanco', 'azul', 'gris', 'beige', 'marron', 'verde', 'rojo', 'rosa',
    'mostaza', 'azul_marino', 'granate', 'ocre', 'turquesa', 'violeta'
]

PATTERNS = ['liso', 'rayas', 'lunares', 'floral', 'geometrico', 'cuadros', 'punto', 'animal_print']

MATERIALS = ['algodon', 'lana', 'jean', 'lino', 'poliester', 'cuero', 'sintetico', 'seda', 'tweed', 'terciopelo']

SEASONS = ['primavera', 'verano', 'otonio', 'invierno']

# 🚨 ACTIVIDADES ALINEADAS CON EL BACKEND / GEMINI
ACTIVITIES = [
    'trabajo_oficina',  # oficina, trabajo
    'deporte',          # deporte genérico
    'gimnasio',         # gym explícito
    'senderismo',       # hiking/outdoor
    'casual_paseo',     # paseo con amigos
    'fiesta_nocturna',  # salir de noche
    'evento_formal',    # boda, gala...
    'relax_en_casa',    # casa
    'comida_cena',      # comida o cena algo arreglada
    'viaje',            # turismo
    'cita_romantica',   # date 💘
]

MOODS = ['energico', 'relajado', 'formal', 'creativo', 'cansado', 'romantico']

# --- Género ---
GENDER = ['hombre', 'mujer', 'no_binario', 'otro']

# --- Colorimetría y contraste ---
SKIN_TONES = ['cálida', 'fría', 'neutra']
COLORIMETRY = {
    'cálida': {
        'recommended': ['marron', 'mostaza', 'verde', 'coral', 'dorado', 'beige', 'ocre'],
        'avoid': ['gris', 'azul_marino_claro']
    },
    'fría': {
        'recommended': ['azul', 'violeta', 'rosa', 'plata', 'malva', 'azul_marino'],
        'avoid': ['amarillo_fuerte', 'naranja', 'mostaza']
    },
    'neutra': {
        'recommended': ['beige', 'malva', 'azul_marino', 'gris', 'blanco', 'negro', 'marron'],
        'avoid': []
    }
}

# --- Complejidad de estampados ---
PATTERN_COMPLEXITY = {
    'liso': 0,
    'punto': 1,
    'cuadros': 1,
    'texturizado': 1,
    'rayas': 2,
    'lunares': 2,
    'geometrico': 2,
    'floral': 3,
    'animal_print': 3
}

# --- Reglas clima-prenda (por clima estacional) ---
SEASONAL_WEATHER = {
    'primavera': ['templado', 'lluvia ligera', 'soleado'],
    'verano': ['caluroso', 'alta_humedad', 'soleado'],
    'otonio': ['fresco', 'viento', 'lluvia'],
    'invierno': ['frio', 'nieve', 'lluvioso']
}

CLIMATE_GARMENT_RULES = {
    'Tropical_Caluroso': {
        'top': ['camiseta', 'top_lino'],
        'bottom': ['shorts', 'falda_midi'],
        'shoes': ['sandalias']
    },
    'Templado_Primaveral': {
        'top': ['camisa', 'blusa_ligera'],
        'bottom': ['jeans', 'falda'],
        'shoes': ['sneakers', 'mules']
    },
    'Frio_Invernal': {
        'top': ['sueter', 'jersey', 'blazer'],
        'bottom': ['pantalones_sastre', 'jeans_grueso'],
        'shoes': ['botines', 'botas']
    },
    'Seco_Otonal': {
        'top': ['camisa_capas', 'blazer'],
        'bottom': ['pantalones_cargo', 'falda_midi'],
        'shoes': ['botines', 'mocasines']
    },
    'Humedo_Lluvioso': {
        'top': ['blusa_impermeable'],
        'bottom': ['pantalones_cropped'],
        'shoes': ['botas_impermeables']
    }
}

# --- Wardrobes por estilo y género ---
STYLE_WARDROBES = {
    'clasico': {
        'hombre': {
            'types': ['camisa', 'blazer', 'pantalones_sastre', 'pumps', 'loafers'],
            'colors': ['beige', 'azul_marino', 'blanco', 'gris'],
            'patterns': ['liso', 'rayas', 'punto'],
        },
        'mujer': {
            'types': ['blusa', 'blazer', 'falda_lapiz', 'vestido', 'pumps', 'loafers'],
            'colors': ['beige', 'azul_marino', 'blanco', 'rosa', 'malva'],
            'patterns': ['liso', 'rayas', 'lunares', 'punto'],
        },
        'no_binario': {
            'types': ['camiseta_oversized', 'blazer_estructurado', 'pantalones_cargo', 'botas_chunky'],
            'colors': ['negro', 'gris', 'blanco', 'verde_militar'],
            'patterns': ['liso', 'geometrico'],
        },
    },
    'romantico': {
        'mujer': {
            'types': ['blusa_volantes', 'top_encaje', 'falda_midi', 'vestido_floral', 'bailarinas'],
            'colors': ['rosa', 'lavanda', 'crema', 'blanco'],
            'patterns': ['floral', 'lunares', 'encaje'],
        },
        'no_binario': {
            'types': ['top_suave', 'pantalones_fluidos', 'botas_planas'],
            'colors': ['blanco', 'malva', 'verde', 'beige'],
            'patterns': ['liso', 'punto'],
        },
    },
    'urbano': {
        'hombre': {
            'types': ['camiseta_grafica', 'jeans_rectos', 'hoodie', 'sneakers_chunky'],
            'colors': ['negro', 'gris', 'azul', 'blanco'],
            'patterns': ['liso', 'grafico'],
        },
        'mujer': {
            'types': ['croptop', 'jeans_rectos', 'bomber', 'sneakers_chunky'],
            'colors': ['negro', 'gris', 'beige', 'rojo'],
            'patterns': ['liso', 'letras', 'geometrico'],
        },
        'no_binario': {
            'types': ['camiseta_asimetrica', 'joggers', 'botas_militar'],
            'colors': ['negro', 'verde', 'gris'],
            'patterns': ['liso', 'camuflaje'],
        },
    },
    'bohemio': {
        'mujer': {
            'types': ['top_bordado', 'pantalones_flare', 'falda_larga', 'botas_planas'],
            'colors': ['marron', 'mostaza', 'verde', 'terracota'],
            'patterns': ['floral', 'lunares', 'etnico'],
        },
        'no_binario': {
            'types': ['top_etnico', 'pantalones_palazzo', 'mocasines'],
            'colors': ['marron', 'beige', 'terracota'],
            'patterns': ['etnico', 'geométrico'],
        },
    },
    'minimalista': {
        'hombre': {
            'types': ['camiseta_blanca', 'pantalones_negros', 'blazer_negro'],
            'colors': ['blanco', 'negro', 'gris', 'azul_marino'],
            'patterns': ['liso'],
        },
        'mujer': {
            'types': ['top_liso', 'pantalones_sastre', 'blazer_negro', 'sandalias_cuadradas'],
            'colors': ['blanco', 'negro', 'gris', 'beige'],
            'patterns': ['liso'],
        },
        'no_binario': {
            'types': ['top_estructurado', 'pantalones_asimetricos', 'mocasines'],
            'colors': ['blanco', 'negro', 'gris'],
            'patterns': ['liso'],
        },
    },
}

# --- Reglas tipo de cuerpo ---
BODY_SHAPE_RULES = {
    'reloj_de_arena': {
        'top_bonus': {'type': ['ajustado', 'cruzado']},
        'bottom_bonus': {'type': ['talle_alto', 'recto']},
        'shoes_bonus': {'type': ['taco_medio', 'taco_fino']},
        'penalties': {'type': ['cuadrado', 'sin_forma']},
    },
    'triangulo': {
        'top_bonus': {'type': ['blazer', 'escote_v'], 'color': ['claro']},
        'bottom_bonus': {'color': ['oscuro'], 'type': ['recto', 'evasé']},
        'shoes_bonus': {'type': ['taco_fino']},
        'penalties': {'type': ['ajustado_en_cadera']},
    },
    'triangulo_invertido': {
        'top_bonus': {'type': ['cuello_u', 'tela_suave']},
        'bottom_bonus': {'type': ['falda_amplia', 'wide_leg']},
        'shoes_bonus': {'type': ['plataforma', 'chunky']},
        'penalties': {'type': ['hombreras']},
    },
    'rectangulo': {
        'top_bonus': {'type': ['top_nudo', 'drapeado']},
        'bottom_bonus': {'type': ['cintura_paperbag', 'flare']},
        'shoes_bonus': {'type': ['botines', 'sandalias_altas']},
        'penalties': {'type': ['recto_sin_forma']},
    },
    'manzana': {
        'top_bonus': {'type': ['lineas_verticales', 'camisa'], 'material': ['fluido']},
        'bottom_bonus': {'type': ['recto', 'no_ajustado']},
        'shoes_bonus': {'type': ['punta_afinada', 'taco_medio']},
        'penalties': {'pattern': 'grande'},
    },
}

# --- Formalidad base y rol de cada tipo de prenda ---
GARMENT_META = {
    # TOPS
    'camiseta':              {'base_formality': 1, 'role': 'sport_casual'},
    'camiseta_blanca':       {'base_formality': 1, 'role': 'sport_casual'},
    'camiseta_grafica':      {'base_formality': 1, 'role': 'sport_casual'},
    'camiseta_tecnica':      {'base_formality': 1, 'role': 'sport'},
    'top_lino':              {'base_formality': 2, 'role': 'casual'},
    'blusa_ligera':          {'base_formality': 2, 'role': 'casual'},
    'blusa_volantes':        {'base_formality': 3, 'role': 'romantic'},
    'top_encaje':            {'base_formality': 3, 'role': 'romantic'},
    'croptop':               {'base_formality': 1, 'role': 'casual'},
    'hoodie':                {'base_formality': 1, 'role': 'sport_casual'},
    'jersey':                {'base_formality': 2, 'role': 'casual'},
    'sueter':                {'base_formality': 2, 'role': 'casual'},
    'camisa':                {'base_formality': 4, 'role': 'smart'},
    'camisa_elegante':       {'base_formality': 5, 'role': 'formal'},
    'blazer':                {'base_formality': 4, 'role': 'formal'},
    'blazer_negro':          {'base_formality': 5, 'role': 'formal'},
    'top_bordado':           {'base_formality': 2, 'role': 'boho'},
    'top_liso':              {'base_formality': 2, 'role': 'minimal'},

    # BOTTOMS
    'jeans':                 {'base_formality': 2, 'role': 'casual'},
    'jeans_rectos':          {'base_formality': 2, 'role': 'casual'},
    'jeans_grueso':          {'base_formality': 2, 'role': 'casual'},
    'pantalones':            {'base_formality': 2, 'role': 'casual'},
    'pantalones_sastre':     {'base_formality': 4, 'role': 'formal'},
    'pantalon_sastre':       {'base_formality': 4, 'role': 'formal'},
    'joggers':               {'base_formality': 1, 'role': 'sport'},
    'pantalon_chandal':      {'base_formality': 1, 'role': 'sport'},
    'mallas_deporte':        {'base_formality': 1, 'role': 'sport'},
    'mallas deporte':        {'base_formality': 1, 'role': 'sport'},
    'pantalon_corto':        {'base_formality': 1, 'role': 'sport_casual'},
    'shorts':                {'base_formality': 1, 'role': 'sport_casual'},
    'falda':                 {'base_formality': 2, 'role': 'casual'},
    'falda_midi':            {'base_formality': 3, 'role': 'smart'},
    'falda_larga':           {'base_formality': 2, 'role': 'boho'},
    'falda_lapiz':           {'base_formality': 4, 'role': 'formal'},

    # SHOES
    'sneakers':              {'base_formality': 1, 'role': 'sport'},
    'sneakers_chunky':       {'base_formality': 1, 'role': 'sport'},
    'zapatillas':            {'base_formality': 1, 'role': 'sport'},
    'zapatillas_trail':      {'base_formality': 1, 'role': 'sport'},
    'zapatos':               {'base_formality': 4, 'role': 'formal'},
    'zapato vestir':         {'base_formality': 5, 'role': 'formal'},
    'pumps':                 {'base_formality': 4, 'role': 'formal'},
    'loafers':               {'base_formality': 3, 'role': 'smart'},
    'mocasines':             {'base_formality': 3, 'role': 'smart'},
    'bailarinas':            {'base_formality': 2, 'role': 'smart_casual'},
    'sandalias':             {'base_formality': 2, 'role': 'casual'},
    'tacones_altos':         {'base_formality': 5, 'role': 'formal'},
    'botines':               {'base_formality': 3, 'role': 'smart'},
    'sandalias_cuadradas':   {'base_formality': 3, 'role': 'smart'},
    'sandalias_finas':       {'base_formality': 3, 'role': 'formal'},
    'botas':                 {'base_formality': 3, 'role': 'casual'},
    'botas_planas':          {'base_formality': 2, 'role': 'casual'},
    'botas_impermeables':    {'base_formality': 2, 'role': 'outdoor'},
}

# --- Helpers ---
def generate_description(garment_type, material, color, pattern):
    material_adjectives = {
        'lana': ['cálido', 'acogedor'],
        'lino': ['fresco', 'ligero'],
        'algodon': ['suave', 'versátil'],
        'jean': ['resistente', 'casual'],
        'cuero': ['elegante', 'resistente'],
        'seda': ['fina', 'brillante'],
        'tweed': ['texturizado', 'británico'],
        'terciopelo': ['suave al tacto', 'elegante'],
        'poliester': ['práctico', 'moderno'],
        'sintetico': ['económico', 'versátil'],
    }
    base_templates = [
        f"Una {garment_type} de {material} color {color}.",
        f"Esta {garment_type} de {material} {color} combina estilo y comodidad.",
        f"Prenda de {material} en tono {color}, perfecta para realzar tu look.",
    ]
    description = random.choice(base_templates)
    if material in material_adjectives:
        description += f" Su tejido es {random.choice(material_adjectives[material])}."
    if pattern != 'liso':
        description += f" Presenta un diseño de {pattern} que aporta personalidad."
    return description


def classify_type(t):
    t_lower = t.lower()
    shoes_kw = ['zapato', 'zapatos', 'bota', 'botas', 'sneaker', 'sneakers', 'pumps',
                'loafers', 'bailarina', 'bailarinas', 'sandalia', 'sandalias', 'mule',
                'mules', 'mocasines']
    bottoms_kw = ['pantalon', 'pantalones', 'jean', 'jeans', 'falda', 'short', 'shorts',
                  'mono', 'monos', 'cargo', 'flare', 'palazzo', 'culotte']
    tops_kw = ['camiseta', 'camisa', 'blusa', 'top', 'hoodie', 'croptop',
               'jersey', 'sueter', 'blazer', 'polo', 'top_liso',
               'blusa_volantes', 'top_encaje', 'top_bordado']

    if any(kw in t_lower for kw in shoes_kw):
        return 'shoes'
    if any(kw in t_lower for kw in bottoms_kw):
        return 'bottom'
    return 'top'


# --- 2. Usuarios ---
print("Generando usuarios...")
users_data = []
for i in range(1, 301):
    gender = random.choice(GENDER)
    users_data.append({
        'id': i,
        'username': fake.user_name(),
        'age': random.randint(18, 65),
        'body_shape': random.choice(BODY_SHAPES),
        'style_preference': random.choice(STYLE_PREFERENCES),
        'skin_tone': random.choice(SKIN_TONES),
        'gender': gender
    })
users_df = pd.DataFrame(users_data)
users_df.to_csv('users.csv', index=False)

# --- 3. Armarios coherentes ---
print("Generando armarios coherentes para cada usuario...")
garments_data = []
garment_id_counter = 1

for _, user in users_df.iterrows():
    user_style = user['style_preference']
    user_gender = user['gender']

    wardrobe_info = STYLE_WARDROBES.get(user_style, {}).get(
        user_gender,
        STYLE_WARDROBES[user_style].get('mujer')
    )

    num_garments = random.randint(25, 35)

    for _ in range(num_garments):
        garment_type = random.choice(wardrobe_info['types'])

        meta = GARMENT_META.get(garment_type, {})
        base_formality = meta.get('base_formality', random.randint(1, 5))

        formality_level = base_formality + random.choice([-1, 0, 0, 1])
        formality_level = max(1, min(5, formality_level))

        color = random.choice(wardrobe_info['colors'])
        if random.random() < 0.30:
            color = random.choice([c for c in COLORS if c not in wardrobe_info['colors']])

        pattern = random.choice(wardrobe_info['patterns'])
        material = random.choice(MATERIALS)

        description = generate_description(garment_type, material, color, pattern)

        garments_data.append({
            'id': garment_id_counter,
            'user_id': user['id'],
            'name': f"{garment_type} de {material} {color}",
            'type': garment_type,
            'primary_color': color,
            'pattern': pattern,
            'material': material,
            'formality_level': formality_level,
            'description': description,
            'type_group': classify_type(garment_type),
        })

        garment_id_counter += 1

garments_df = pd.DataFrame(garments_data)
garments_df.to_csv('garments.csv', index=False)

# --- 4. Contextos ---
print("Generando contextos...")
contexts_data = []
for i in range(1, 601):
    season = random.choice(SEASONS)
    weather = random.choice(SEASONAL_WEATHER[season])

    if season == 'invierno':
        temp = random.uniform(-5, 12)
    elif season == 'verano':
        temp = random.uniform(22, 38)
    elif season == 'primavera':
        temp = random.uniform(12, 25)
    else:
        temp = random.uniform(8, 22)

    activity = random.choice(ACTIVITIES)

    # formalidad coherente con la actividad (incluyendo nuevas)
    if activity in ['deporte', 'gimnasio', 'senderismo', 'relax_en_casa']:
        formality_level = random.randint(1, 2)
    elif activity in ['casual_paseo', 'viaje']:
        formality_level = random.randint(1, 3)
    elif activity in ['trabajo_oficina', 'comida_cena']:
        formality_level = random.randint(2, 4)
    elif activity in ['fiesta_nocturna', 'evento_formal', 'cita_romantica']:
        formality_level = random.randint(3, 5)
    else:
        formality_level = random.randint(1, 5)

    contexts_data.append({
        'id': i,
        'season': season,
        'weather_condition': weather,
        'temperature': round(temp, 1),
        'activity': activity,
        'mood': random.choice(MOODS),
        'formality_level': formality_level
    })

contexts_df = pd.DataFrame(contexts_data)
contexts_df.to_csv('contexts.csv', index=False)

# --- 5. Valoraciones ---
print("Generando valoraciones de outfits únicas y con lógica experta...")
outfit_ratings_data = []
generated_combinations = set()

users = users_df.set_index('id')
garments = garments_df.set_index('id')
contexts = contexts_df.set_index('id')

tops = garments[garments['type_group'] == 'top']
bottoms = garments[garments['type_group'] == 'bottom']
shoes = garments[garments['type_group'] == 'shoes']

max_instances = 16000
attempts = 0
max_attempts = 400000

while len(outfit_ratings_data) < max_instances and attempts < max_attempts:
    attempts += 1
    user_id = random.randint(1, 300)
    context_id = random.randint(1, 600)

    user_tops = tops[tops['user_id'] == user_id]
    user_bottoms = bottoms[bottoms['user_id'] == user_id]
    user_shoes = shoes[shoes['user_id'] == user_id]

    if user_tops.empty or user_bottoms.empty or user_shoes.empty:
        continue

    top = user_tops.sample(1).iloc[0]
    bottom = user_bottoms.sample(1).iloc[0]
    shoe = user_shoes.sample(1).iloc[0]

    combination_key = (user_id, context_id, int(top.name), int(bottom.name), int(shoe.name))
    if combination_key in generated_combinations:
        continue
    generated_combinations.add(combination_key)

    rating = 2.5  # base neutra

    user = users.loc[user_id]
    context = contexts.loc[context_id]
    gender = user['gender']

    top_meta = GARMENT_META.get(top['type'], {})
    bottom_meta = GARMENT_META.get(bottom['type'], {})
    shoes_meta = GARMENT_META.get(shoe['type'], {})

    top_role = top_meta.get('role', '')
    bottom_role = bottom_meta.get('role', '')
    shoes_role = shoes_meta.get('role', '')

    top_form = top['formality_level']
    bottom_form = bottom['formality_level']
    shoes_form = shoe['formality_level']

    activity = context['activity']
    form_ctx = context['formality_level']

    # -------- Regla 1: adecuación a la actividad --------
    if activity == 'trabajo_oficina':
        if top_form >= 3 and bottom_form >= 3:
            rating += 1.0
        if top_role in ['formal', 'smart'] and bottom_role in ['formal', 'smart']:
            rating += 0.7
        if shoes_role == 'sport' and form_ctx >= 3:
            rating -= 1.0
        if top['type'] in ['camiseta', 'croptop', 'hoodie'] and top_form < 3:
            rating -= 1.2

    elif activity == 'evento_formal':
        if top_form >= 4 and bottom_form >= 4 and shoes_form >= 4:
            rating += 1.5
        if top_role == 'formal' and bottom_role == 'formal' and shoes_role in ['formal', 'smart']:
            rating += 1.0
        if shoes_role == 'sport':
            rating -= 2.0

    elif activity in ['deporte', 'gimnasio', 'senderismo']:
        if top_role in ['sport', 'sport_casual']:
            rating += 1.0
        else:
            rating -= 0.8
        if bottom_role in ['sport', 'sport_casual']:
            rating += 1.0
        else:
            rating -= 0.8
        if shoes_role == 'sport':
            rating += 1.5
        elif shoes_role in ['formal', 'smart']:
            rating -= 2.0
        if top_form >= 4:
            rating -= 1.0
        if bottom_form >= 4:
            rating -= 1.0
        if shoes_form >= 4:
            rating -= 1.0

    elif activity == 'relax_en_casa':
        if top_form <= 2 and bottom_form <= 2:
            rating += 0.8
        if top_form >= 4 or bottom_form >= 4 or shoes_form >= 4:
            rating -= 1.2

    elif activity == 'casual_paseo':
        if 1 <= top_form <= 3 and 1 <= bottom_form <= 3:
            rating += 0.7
        if top_role == 'formal' and bottom_role == 'formal' and form_ctx <= 2:
            rating -= 0.8

    elif activity == 'fiesta_nocturna':
        if 2 <= top_form <= 4 and 2 <= bottom_form <= 4:
            rating += 0.5
        if shoes_role == 'sport' and form_ctx >= 3:
            rating -= 0.5

    elif activity == 'cita_romantica':
        # similar a fiesta, pero un pelín más exigente en calzado
        if 3 <= top_form <= 5 and 3 <= bottom_form <= 5:
            rating += 0.8
        if shoes_role in ['formal', 'smart']:
            rating += 0.8
        if shoes_role == 'sport':
            rating -= 1.5

    # -------- Regla 2: clima / material --------
    if context['season'] == 'invierno':
        if top['material'] in ['lana', 'tweed']:
            rating += 1.0
        if top['material'] in ['lino', 'seda']:
            rating -= 2.0
    elif context['season'] == 'verano':
        if top['material'] in ['lino', 'algodon']:
            rating += 1.0
        if top['material'] in ['lana', 'tweed']:
            rating -= 2.0

    if context['weather_condition'] in ['lluvia ligera', 'lluvioso', 'lluvia']:
        if 'bota' in shoe['type'] or 'botas' in shoe['type']:
            rating += 0.5
        if 'sneaker' in shoe['type'] and shoe['material'] not in ['cuero', 'sintetico']:
            rating -= 0.5

    # -------- Regla 3: estampados --------
    top_complexity = PATTERN_COMPLEXITY.get(top['pattern'], 0)
    bottom_complexity = PATTERN_COMPLEXITY.get(bottom['pattern'], 0)
    if top_complexity > 1 and bottom_complexity > 1:
        rating -= 1.2
    if top['pattern'] in ['floral', 'animal_print'] and bottom['pattern'] != 'liso':
        rating -= 1.2

    # -------- Regla 4: color --------
    if top['primary_color'] == bottom['primary_color'] and top['pattern'] != 'liso':
        rating -= 1.0

    # -------- Regla 5: cuerpo --------
    body_shape_rules = BODY_SHAPE_RULES.get(user['body_shape'], {})
    if body_shape_rules.get('top_bonus'):
        top_bonus_types = body_shape_rules['top_bonus'].get('type', [])
        if any(tk in top['type'] for tk in top_bonus_types):
            rating += 0.4
    if body_shape_rules.get('bottom_bonus'):
        bottom_bonus_types = body_shape_rules['bottom_bonus'].get('type', [])
        if any(bk in bottom['type'] for bk in bottom_bonus_types):
            rating += 0.4
    if body_shape_rules.get('penalties'):
        penalty_types = body_shape_rules['penalties'].get('type', [])
        if any(pk in top['type'] for pk in penalty_types):
            rating -= 0.5

    # -------- Regla 6: colorimetría --------
    colorimetry_rules = COLORIMETRY.get(user['skin_tone'], {})
    outfit_colors = [top['primary_color'], bottom['primary_color']]
    if any((c in colorimetry_rules.get('avoid', [])) for c in outfit_colors if c):
        rating -= 0.5
    if any((c in colorimetry_rules.get('recommended', [])) for c in outfit_colors if c):
        rating += 0.3

    # -------- Regla 7: género --------
    if gender == 'mujer':
        if top['type'] == 'vestido' and form_ctx > 2:
            rating += 0.7
        if 'falda' in bottom['type'] or 'falda' in bottom['name']:
            rating += 0.2
    elif gender == 'hombre':
        if top['type'] == 'camisa' and bottom['type'] == 'pantalones_sastre':
            rating += 0.8
        if top['type'] == 'blazer' and bottom['type'] == 'jeans':
            rating += 0.4
    elif gender == 'no_binario':
        if top_form > 3 and bottom_form < 3:
            rating += 0.3
        if 'estructurado' in top['name'] or 'asimetrico' in top['name']:
            rating += 0.5

    # -------- Ruido y recorte --------
    rating += random.uniform(-0.4, 0.4)
    final_rating = max(1.0, min(5.0, round(rating)))

    outfit_ratings_data.append({
        'id': len(outfit_ratings_data) + 1,
        'user_id': user_id,
        'context_id': context_id,
        'top_garment_id': int(top.name),
        'bottom_garment_id': int(bottom.name),
        'shoes_garment_id': int(shoe.name),
        'rating': final_rating
    })

print(f"\nSe generaron {len(outfit_ratings_data)} valoraciones en {attempts} intentos (máx buscado: {max_instances}).")

outfit_ratings_df = pd.DataFrame(outfit_ratings_data)
outfit_ratings_df.to_csv('outfit_ratings.csv', index=False)

print("Archivos creados: users.csv, garments.csv, contexts.csv, outfit_ratings.csv")
print("✅ ¡Base de datos generada con éxito con lógica experta, de género y roles de prendas!")
