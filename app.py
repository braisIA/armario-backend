# backend/app.py
import os
import time
import pickle

from flask import Flask, request, jsonify, url_for
from flask_restful import Api, Resource
from flask_migrate import Migrate
from flask_jwt_extended import (
    JWTManager,
    jwt_required,
    create_access_token,
    get_jwt_identity,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from config import Config
from models import db, User, Garment, Context
from gemini_service import extract_context_from_message, generate_assistant_message
from recommendation_service import get_recommendation

# =====================================================
#          INICIALIZACIÓN APP Y CONFIGURACIÓN
# =====================================================

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
migrate = Migrate(app, db)
api = Api(app)
jwt = JWTManager(app)

CORS(app, resources={r"/api/*": {"origins": "*"}})

# Carpeta base del backend
BASE_DIR = os.path.dirname(__file__)

# ==========================
#   CONFIG SUBIDA IMÁGENES
# ==========================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =====================================================
#            CARGA DEL MODELO DE RECOMENDACIÓN
# =====================================================

try:
    model_path = os.path.join(BASE_DIR, "ml", "models", "recommendation_model.pkl")
    preprocessor_path = os.path.join(BASE_DIR, "ml", "models", "preprocessor.pkl")

    print(f"Buscando modelo en: {model_path}")
    print(f"Buscando preprocesador en: {preprocessor_path}")

    with open(model_path, "rb") as f:
        recommendation_model = pickle.load(f)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    print("✅ Modelo de ML y preprocesador cargados exitosamente.")
except FileNotFoundError:
    recommendation_model = None
    preprocessor = None
    print(
        "⚠️ ADVERTENCIA: No se encontraron los archivos del modelo de ML. "
        "La recomendación no estará disponible."
    )


# =====================================================
#                      RECURSOS API
# =====================================================


class UserRegistration(Resource):
    def post(self):
        data = request.get_json()
        if not data or not all(k in data for k in ("username", "email", "password")):
            return {
                "message": "Faltan datos requeridos (username, email, password)"
            }, 400

        if User.query.filter_by(email=data["email"]).first():
            return {"message": "El email ya está en uso"}, 400

        new_user = User(username=data["username"], email=data["email"])
        new_user.set_password(data["password"])

        # Datos adicionales (opcionales)
        new_user.age = data.get("age")
        new_user.body_shape = data.get("body_shape")
        new_user.style_preference = data.get("style_preference")
        new_user.skin_tone = data.get("skin_tone")
        new_user.gender = data.get("gender")

        db.session.add(new_user)
        db.session.commit()

        return {"message": "Usuario creado correctamente"}, 201


class UserLogin(Resource):
    def post(self):
        data = request.get_json()
        if not data or not all(k in data for k in ("email", "password")):
            return {
                "message": "Faltan datos requeridos (email, password)"
            }, 400

        user = User.query.filter_by(email=data["email"]).first()

        if user and user.check_password(data["password"]):
            # identity como string para evitar "Subject must be a string"
            access_token = create_access_token(identity=str(user.id))
            return {
                "message": "Login exitoso",
                "access_token": access_token,
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "age": user.age,
                "body_shape": user.body_shape,
                "style_preference": user.style_preference,
                "skin_tone": user.skin_tone,
                "gender": user.gender,
            }, 200
        else:
            return {"message": "Credenciales inválidas"}, 401


class UserProfile(Resource):
    @jwt_required()
    def get(self):
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        if not user:
            return {"message": "Usuario no encontrado"}, 404

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "body_shape": user.body_shape,
            "style_preference": user.style_preference,
            "skin_tone": user.skin_tone,
            "gender": user.gender,
        }, 200

    @jwt_required()
    def put(self):
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        if not user:
            return {"message": "Usuario no encontrado"}, 404

        data = request.get_json() or {}

        if "username" in data:
            user.username = data["username"]
        if "age" in data:
            user.age = data["age"]
        if "body_shape" in data:
            user.body_shape = data["body_shape"]
        if "style_preference" in data:
            user.style_preference = data["style_preference"]
        if "skin_tone" in data:
            user.skin_tone = data["skin_tone"]
        if "gender" in data:
            user.gender = data["gender"]

        db.session.commit()

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "body_shape": user.body_shape,
            "style_preference": user.style_preference,
            "skin_tone": user.skin_tone,
            "gender": user.gender,
        }, 200


class GarmentList(Resource):
    @jwt_required()
    def get(self):
        """Devuelve todas las prendas del usuario actual."""
        user_id = int(get_jwt_identity())
        garments = Garment.query.filter_by(user_id=user_id).all()

        result = [
            {
                "id": g.id,
                "name": g.name,
                "type": g.type,
                "primary_color": g.primary_color,
                "pattern": g.pattern,
                "material": g.material,
                "formality_level": g.formality_level,
                "description": g.description,
                "type_group": g.type_group,
                "image_url": g.image_url,
            }
            for g in garments
        ]
        return jsonify(result)

    @jwt_required()
    def post(self):
        """
        Añade una prenda al armario del usuario vía JSON puro (sin imagen).
        Para subida con imagen usar: POST /api/garments/upload (multipart/form-data)
        """
        user_id = int(get_jwt_identity())
        data = request.get_json()

        if not data or not all(k in data for k in ("name", "type", "primary_color")):
            return {
                "message": "Faltan datos requeridos (name, type, primary_color)"
            }, 400

        new_garment = Garment(
            user_id=user_id,
            name=data["name"],
            type=data["type"],
            primary_color=data["primary_color"],
            pattern=data.get("pattern"),
            material=data.get("material"),
            formality_level=data.get("formality_level", 3),
            description=data.get("description", ""),
            type_group=data.get("type_group", "top"),
            image_url=data.get("image_url", ""),
        )
        db.session.add(new_garment)
        db.session.commit()
        return {"message": "Prenda añadida", "garment_id": new_garment.id}, 201


class GarmentDetail(Resource):
    @jwt_required()
    def delete(self, garment_id):
        user_id = int(get_jwt_identity())
        garment = Garment.query.filter_by(id=garment_id, user_id=user_id).first()

        if not garment:
            return {"message": "Prenda no encontrada"}, 404

        db.session.delete(garment)
        db.session.commit()
        return {"message": "Prenda eliminada correctamente"}, 200

    @jwt_required()
    def put(self, garment_id):
        """Editar una prenda existente del usuario (JSON, sin nueva imagen)."""
        user_id = int(get_jwt_identity())
        garment = Garment.query.filter_by(id=garment_id, user_id=user_id).first()

        if not garment:
            return {"message": "Prenda no encontrada"}, 404

        data = request.get_json() or {}

        if "name" in data:
            garment.name = data["name"]
        if "type" in data:
            garment.type = data["type"]
        if "primary_color" in data:
            garment.primary_color = data["primary_color"]
        if "pattern" in data:
            garment.pattern = data["pattern"]
        if "material" in data:
            garment.material = data["material"]
        if "formality_level" in data:
            garment.formality_level = data["formality_level"]
        if "description" in data:
            garment.description = data["description"]
        if "type_group" in data:
            garment.type_group = data["type_group"]
        if "image_url" in data:
            garment.image_url = data["image_url"]

        db.session.commit()

        return {
            "message": "Prenda actualizada correctamente",
            "garment": {
                "id": garment.id,
                "name": garment.name,
                "type": garment.type,
                "primary_color": garment.primary_color,
                "pattern": garment.pattern,
                "material": garment.material,
                "formality_level": garment.formality_level,
                "description": garment.description,
                "type_group": garment.type_group,
                "image_url": garment.image_url,
            },
        }, 200


# =====================================================
#   ENDPOINT ESPECIAL: CREAR PRENDA + SUBIR IMAGEN
# =====================================================

@app.route("/api/garments/upload", methods=["POST"])
@jwt_required()
def create_garment_with_image():
    user_id = int(get_jwt_identity())

    # Campos de texto (multipart/form-data)
    name = request.form.get("name")
    gtype = request.form.get("type")
    primary_color = request.form.get("primary_color")

    if not name or not gtype or not primary_color:
        return (
            jsonify(
                {
                    "message": "Faltan campos obligatorios (name, type, primary_color)"
                }
            ),
            400,
        )

    pattern = request.form.get("pattern") or None
    material = request.form.get("material") or None
    formality_level_raw = request.form.get("formality_level", "3")
    description = request.form.get("description") or ""
    type_group = request.form.get("type_group", "top")

    try:
        formality_level = int(formality_level_raw)
    except ValueError:
        formality_level = 3

    # Archivo de imagen (opcional)
    file = request.files.get("image")
    image_url = ""

    if file and file.filename:
        if not allowed_file(file.filename):
            return jsonify({"message": "Extensión de imagen no permitida"}), 400

        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[1].lower()

        unique_name = f"user{user_id}_{int(time.time())}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(save_path)

        # URL pública accessible por Flutter: http://host:5000/static/uploads/...
        image_url = url_for("static", filename=f"uploads/{unique_name}", _external=True)

    new_garment = Garment(
        user_id=user_id,
        name=name,
        type=gtype,
        primary_color=primary_color,
        pattern=pattern,
        material=material,
        formality_level=formality_level,
        description=description,
        type_group=type_group,
        image_url=image_url,
    )

    db.session.add(new_garment)
    db.session.commit()

    return (
        jsonify(
            {
                "message": "Prenda creada con imagen",
                "garment_id": new_garment.id,
                "image_url": image_url,
            }
        ),
        201,
    )


# =====================================================
#              RECOMENDACIONES (CHAT + OUTFIT)
# =====================================================

class RecommendationResource(Resource):
    @jwt_required()
    def post(self):
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return {
                "message": "Se requiere un mensaje para generar recomendaciones"
            }, 400

        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        if not user:
            return {"message": "Usuario no encontrado"}, 404

        user_data = {
            "age": user.age,
            "body_shape": user.body_shape,
            "style_preference": user.style_preference,
            "gender": user.gender,
            "skin_tone": user.skin_tone,
        }

        # 1) Obtener contexto desde Gemini (o por defecto)
        context_data = extract_context_from_message(user_message)
        print(f"[API] Mensaje usuario: {user_message}")
        print(f"[API] Contexto Gemini: {context_data}")

        # lista de elementos a evitar (puede venir de Gemini)
        avoid_items = context_data.get("avoid_items", [])
        if not isinstance(avoid_items, list):
            avoid_items = [str(avoid_items)]
        avoid_items = [str(x).lower() for x in avoid_items]

        # 2) Si es un saludo simple -> no recomendamos outfit
        if context_data.get("activity") == "saludo":
            assistant_message = generate_assistant_message(
                user_message=user_message,
                context_data=context_data,
                outfit=None,
                predicted_rating=None,
            )
            return {
                "status": "greeting",
                "assistant_message": assistant_message,
                "context": context_data,
                "outfit": None,
            }, 200

        # 3) Normalizar contexto y guardarlo
        season = context_data.get("season") or "primavera"
        weather = context_data.get("weather_condition") or "templado"
        activity = context_data.get("activity") or "casual_paseo"
        mood = context_data.get("mood") or "relajado"
        try:
            temperature = float(context_data.get("temperature", 20.0))
        except (TypeError, ValueError):
            temperature = 20.0
        try:
            formality_level = int(context_data.get("formality_level", 3))
        except (TypeError, ValueError):
            formality_level = 3

        formality_level = max(1, min(5, formality_level))

        ctx = Context(
            user_id=user_id,
            season=season,
            weather_condition=weather,
            temperature=temperature,
            activity=activity,
            mood=mood,
            formality_level=formality_level,
        )
        db.session.add(ctx)
        db.session.commit()

        cleaned_context = {
            "season": season,
            "weather_condition": weather,
            "temperature": temperature,
            "activity": activity,
            "mood": mood,
            "formality_level": formality_level,
            "avoid_items": avoid_items,
        }

        # 4) Llamar al modelo de recomendación
        recommendation_result = get_recommendation(
            user_id,
            cleaned_context,
            user_data,
            recommendation_model,
            preprocessor,
            avoid_items=avoid_items,
        )

        # 5) Preparar datos para Gemini (texto de respuesta)
        if recommendation_result:
            best_outfit, best_score = recommendation_result

            assistant_message = generate_assistant_message(
                user_message=user_message,
                context_data=cleaned_context,
                outfit=best_outfit,
                predicted_rating=float(best_score),
            )

            result = {
                "status": "ok",
                "assistant_message": assistant_message,
                "context": cleaned_context,
                "predicted_rating": round(float(best_score), 2),
                "outfit": {
                    "top": {
                        "id": best_outfit["top"].id,
                        "name": best_outfit["top"].name,
                        "type": best_outfit["top"].type,
                        "primary_color": best_outfit["top"].primary_color,
                        "image_url": best_outfit["top"].image_url,
                        "type_group": best_outfit["top"].type_group,
                    },
                    "bottom": {
                        "id": best_outfit["bottom"].id,
                        "name": best_outfit["bottom"].name,
                        "type": best_outfit["bottom"].type,
                        "primary_color": best_outfit["bottom"].primary_color,
                        "image_url": best_outfit["bottom"].image_url,
                        "type_group": best_outfit["bottom"].type_group,
                    },
                    "shoes": {
                        "id": best_outfit["shoes"].id,
                        "name": best_outfit["shoes"].name,
                        "type": best_outfit["shoes"].type,
                        "primary_color": best_outfit["shoes"].primary_color,
                        "image_url": best_outfit["shoes"].image_url,
                        "type_group": best_outfit["shoes"].type_group,
                    },
                },
            }
            return result, 200

        else:
            # No hay outfit (por falta de prendas, o avoid_items muy restrictivo, etc.)
            assistant_message = generate_assistant_message(
                user_message=user_message,
                context_data=cleaned_context,
                outfit=None,
                predicted_rating=None,
            )

            return {
                "status": "no_outfit",
                "assistant_message": assistant_message,
                "context": cleaned_context,
            }, 200


# =====================================================
#                  REGISTRO DE RECURSOS
# =====================================================

api.add_resource(UserRegistration, "/api/register")
api.add_resource(UserLogin, "/api/login")
api.add_resource(UserProfile, "/api/user/profile")
api.add_resource(GarmentList, "/api/garments")
api.add_resource(RecommendationResource, "/api/recommend")
api.add_resource(GarmentDetail, "/api/garments/<int:garment_id>")


@app.route("/hola")
def hola_mundo():
    return "¡Hola mundo! El backend está funcionando."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
