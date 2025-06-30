# db.py

import sqlalchemy
from sqlalchemy import text
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = sqlalchemy.create_engine(DATABASE_URL)


def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL image to base64-encoded PNG string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def insert_user_input(label: int, image: Image.Image):
    """
    Inserts a user-labeled digit image into the 'user_input' table.
    """
    base64_img = image_to_base64(image)
    timestamp = datetime.now()

    insert_query = text("""
        INSERT INTO user_input (label, image_base64, created_at)
        VALUES (:label, :image_base64, :created_at)
    """)

    with engine.connect() as conn:
        conn.execute(insert_query, {
            "label": label,
            "image_base64": base64_img,
            "created_at": timestamp
        })
        conn.commit()

    print("Inserted label and image into database.")

