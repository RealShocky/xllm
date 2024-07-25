from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    tier = db.Column(db.String(50), nullable=False)  # e.g., 'free', 'pro', 'enterprise'
    query_count = db.Column(db.Integer, default=0)
    query_reset_date = db.Column(db.DateTime, nullable=False)
