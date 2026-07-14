"""
Fire Detection System - Flask Backend
------------------------------------------------
Handles:
- Username entry + logging to MySQL
- Serving the feature selection page
- Photo upload detection
- Live camera detection (via browser webcam + JS, not OpenCV VideoCapture,
  since this needs to work when deployed on a server with no physical camera)

HOW TO RUN LOCALLY:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import base64
import io
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pymysql

from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key"  # needed for session (username)

MODEL_PATH = "model/model.h5"
IMG_SIZE = (224, 224)

# ------------------------------
# DATABASE CONNECTION SETTINGS
# ------------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",   # <-- change this to your MySQL password
    "database": "fire_detection_db",
}


def get_db_connection():
    return pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)


# ------------------------------
# LOAD MODEL ONCE AT STARTUP
# ------------------------------
try:
    model = load_model(MODEL_PATH, compile=False)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"WARNING: could not load model -> {e}")


def predict_fire(img_array):
    """
    Takes a raw RGB image (numpy array) and returns (is_fire, label, confidence)
    """
    img_resized = cv2.resize(img_array, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_input, verbose=0)[0][0]

    # class_indices from training: {'fire_images': 0, 'non_fire_images': 1}
    if prediction < 0.5:
        is_fire = True
        label = "FIRE DETECTED"
        confidence = (1 - prediction) * 100
    else:
        is_fire = False
        label = "NO FIRE DETECTED"
        confidence = prediction * 100

    return is_fire, label, float(confidence)


# ==================================================================
# ROUTE 1: HOME PAGE - USERNAME ENTRY
# ==================================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if not username:
            return render_template("index.html", error="Please enter your name.")

        # Save to session so other pages know who's using the app
        session["username"] = username

        # Log this visit to MySQL
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO users (username, visit_time) VALUES (%s, %s)",
                    (username, datetime.now()),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB logging failed: {e}")  # app still works even if DB is down

        return redirect(url_for("features"))

    return render_template("index.html")


# ==================================================================
# ROUTE 2: FEATURE SELECTION PAGE
# ==================================================================
@app.route("/features")
def features():
    username = session.get("username")
    if not username:
        return redirect(url_for("index"))
    return render_template("features.html", username=username)


# ==================================================================
# ROUTE 3: PHOTO DETECTION PAGE
# ==================================================================
@app.route("/photo")
def photo_page():
    username = session.get("username")
    if not username:
        return redirect(url_for("index"))
    return render_template("photo.html", username=username)


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if not model_loaded:
        return jsonify({"error": "Model not loaded on server"}), 500

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(file.stream).convert("RGB")
    img_array = np.array(img)
    is_fire, label, confidence = predict_fire(img_array)

    log_detection(session.get("username", "guest"), "photo", label, confidence)

    return jsonify({"is_fire": is_fire, "label": label, "confidence": round(confidence, 2)})


# ==================================================================
# ROUTE 4: LIVE CAMERA DETECTION PAGE
# ==================================================================
@app.route("/camera")
def camera_page():
    username = session.get("username")
    if not username:
        return redirect(url_for("index"))
    return render_template("camera.html", username=username)


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """
    Receives a single webcam frame captured by JavaScript (as a base64
    JPEG string), runs it through the model, and returns the result.
    The browser calls this endpoint repeatedly (e.g. every second) to
    simulate live detection.
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded on server"}), 500

    data = request.get_json()
    frame_data = data.get("frame")  # base64 string like "data:image/jpeg;base64,...."

    if not frame_data:
        return jsonify({"error": "No frame received"}), 400

    # Strip the "data:image/jpeg;base64," prefix
    header, encoded = frame_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_array = np.array(img)

    is_fire, label, confidence = predict_fire(img_array)

    return jsonify({"is_fire": is_fire, "label": label, "confidence": round(confidence, 2)})


# ==================================================================
# HELPER: LOG EACH DETECTION RESULT TO MYSQL
# ==================================================================
def log_detection(username, mode, label, confidence):
    try:
        result = "FIRE" if "FIRE DETECTED" in label and "NO" not in label else "NO_FIRE"
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO detection_logs (username, detection_mode, result, confidence, detected_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (username, mode, result, confidence, datetime.now()),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Detection log failed: {e}")


# ==================================================================
# ROUTE 5: SIMPLE STATS PAGE (how many people used the site)
# ==================================================================
@app.route("/stats")
def stats():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total_visits FROM users")
            total_visits = cursor.fetchone()["total_visits"]

            cursor.execute("SELECT COUNT(DISTINCT username) AS unique_users FROM users")
            unique_users = cursor.fetchone()["unique_users"]

            cursor.execute(
                "SELECT username, visit_time FROM users ORDER BY visit_time DESC LIMIT 20"
            )
            recent_visits = cursor.fetchall()
        conn.close()
    except Exception as e:
        return f"Could not load stats: {e}"

    return render_template(
        "stats.html",
        total_visits=total_visits,
        unique_users=unique_users,
        recent_visits=recent_visits,
    )


if __name__ == "__main__":
    app.run(debug=True)
