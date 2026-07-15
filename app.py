"""
Fire Detection System - Flask Backend (TFLite version)
------------------------------------------------
Same as before, but uses TensorFlow Lite for inference instead of full
Keras/TensorFlow. This uses dramatically less memory, which matters a lot
on Render's free 512MB tier.

HOW TO RUN LOCALLY:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"

import base64
import io
import traceback
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pymysql

# tflite_runtime is a much smaller, lighter-weight package than full
# tensorflow — it only contains what's needed to RUN a model, not train
# one. This is the key memory saving over the previous version.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for local development if tflite_runtime isn't installed —
    # uses the tflite interpreter bundled inside full tensorflow instead.
    from tensorflow.lite import Interpreter

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key"  # needed for session (username)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

MODEL_PATH = "model/model.tflite"
IMG_SIZE = (224, 224)
MAX_UPLOAD_SIDE = 400

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
# LOAD TFLITE MODEL ONCE AT STARTUP
# ------------------------------
try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    interpreter = None
    model_loaded = False
    print(f"WARNING: could not load TFLite model -> {e}")


def load_and_shrink_image(file_or_bytes):
    img = Image.open(file_or_bytes)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.LANCZOS)
    return np.array(img)


def predict_fire(img_array):
    """
    Takes a raw RGB image (numpy array) and returns (is_fire, label, confidence)
    using the TFLite interpreter.
    """
    img_resized = cv2.resize(img_array, IMG_SIZE)
    img_normalized = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_normalized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

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

        session["username"] = username

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
            print(f"DB logging failed: {e}")

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

    try:
        img_array = load_and_shrink_image(file.stream)
        is_fire, label, confidence = predict_fire(img_array)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not process image: {e}"}), 500

    try:
        log_detection(session.get("username", "guest"), "photo", label, confidence)
    except Exception as e:
        print(f"Detection log failed (non-fatal): {e}")

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
    if not model_loaded:
        return jsonify({"error": "Model not loaded on server"}), 500

    data = request.get_json(silent=True) or {}
    frame_data = data.get("frame")

    if not frame_data:
        return jsonify({"error": "No frame received"}), 400

    try:
        header, encoded = frame_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img_array = load_and_shrink_image(io.BytesIO(img_bytes))
        is_fire, label, confidence = predict_fire(img_array)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not process frame: {e}"}), 500

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


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Image is too large. Please upload a smaller photo."}), 413


if __name__ == "__main__":
    app.run(debug=True)