# models.py
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

# ----------------------------
# Tabla de usuarios
# ----------------------------
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    # Datos adicionales para recomendaciones
    age = db.Column(db.Integer)
    body_shape = db.Column(db.String(50))
    style_preference = db.Column(db.String(50))
    skin_tone = db.Column(db.String(50))
    gender = db.Column(db.String(50))

    # Relación con prendas
    garments = db.relationship('Garment', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# ----------------------------
# Tabla de prendas del usuario
# ----------------------------
class Garment(db.Model):
    __tablename__ = 'garment'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    primary_color = db.Column(db.String(50), nullable=False)
    pattern = db.Column(db.String(50))
    material = db.Column(db.String(50))
    formality_level = db.Column(db.Integer)
    description = db.Column(db.Text)
    type_group = db.Column(db.String(50))  # top, bottom, shoes, etc.
    image_url = db.Column(db.String(255))  # URL de la imagen de la prenda


# ----------------------------
# Tabla de contexto de cada interacción
# ----------------------------
class Context(db.Model):
    __tablename__ = 'context'
    id = db.Column(db.Integer, primary_key=True)

    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    season = db.Column(db.String(50))
    weather_condition = db.Column(db.String(50))
    temperature = db.Column(db.Float)
    activity = db.Column(db.String(50))
    mood = db.Column(db.String(50))
    formality_level = db.Column(db.Integer)
