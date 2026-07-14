# Fire Detection Full-Stack Web App

Flask + HTML/CSS/JS + MySQL + your trained CNN model, all in one deployable app.

## Flow
1. User visits your link → enters name → logged to MySQL
2. Picks "Live Camera Detection" or "Upload Photo Detection"
3. Sees a live result (Fire / No Fire) with confidence %
4. If fire detected → alarm beep plays in the browser
5. Every visit + detection gets logged to MySQL so you can check `/stats`

## Folder Structure
```
fire_webapp/
├── app.py                  <- Flask backend (routes + model + MySQL)
├── requirements.txt
├── database/
│   └── schema.sql           <- run this once in MySQL to create tables
├── model/
│   └── model.h5              <- put your trained model here (from train_model.py)
├── templates/                <- HTML pages (Flask renders these)
│   ├── index.html
│   ├── features.html
│   ├── camera.html
│   ├── photo.html
│   └── stats.html
└── static/
    ├── css/style.css
    └── js/alarm.js
```

## PART 1: Local Setup & Testing

### 1. Copy your trained model
Copy `model.h5` (created earlier by `train_model.py`) into this project's `model/` folder.

### 2. Install MySQL locally (for testing before deployment)
Easiest option: install **XAMPP** (https://www.apachefriends.org) which includes MySQL,
or install MySQL Community Server directly (https://dev.mysql.com/downloads/mysql/).

### 3. Create the database
Open MySQL (via XAMPP's phpMyAdmin, or `mysql -u root -p` in terminal) and run the
contents of `database/schema.sql`. This creates the database and both tables.

### 4. Update your MySQL password in app.py
Open `app.py`, find this section near the top, and change the password:
```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_MYSQL_PASSWORD",   # <-- put your real MySQL password here
    "database": "fire_detection_db",
}
```

### 5. Install Python libraries
```
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 6. Run it locally
```
python app.py
```
Open **http://127.0.0.1:5000** in your browser. Test both detection modes.
Visit **http://127.0.0.1:5000/stats** to see logged visits.

---

## PART 2: Deployment (so the URL works without your laptop running)

You need 3 things hosted online: the Flask app, a MySQL database, and the model file.

### Recommended free/cheap options:
- **App hosting:** Render.com (free tier) or Railway.app
- **MySQL hosting:** Railway.app (has free MySQL) or PlanetScale (free MySQL-compatible tier)

### General deployment steps (Render example):
1. Push this whole project to a **GitHub repository**
2. Create a **MySQL database on Railway** (or similar) — note the host, user, password, database name it gives you
3. Update `DB_CONFIG` in `app.py` with those cloud MySQL credentials (better: use environment variables instead of hardcoding — ask me and I'll show you how)
4. Run the `schema.sql` against that cloud MySQL database (most providers have a web console for this)
5. On Render.com: **New → Web Service** → connect your GitHub repo
6. Set the **Start Command** to:
   ```
   gunicorn app:app
   ```
7. Deploy. Render gives you a live URL like `https://your-app.onrender.com`

### Important notes for deployment:
- **Model file size:** `model.h5` can be large. Most free hosting tiers handle a few hundred MB fine, but check your provider's limits.
- **Secret key:** change `app.secret_key` in `app.py` to a long random string before deploying (don't use the placeholder).
- **Database password:** never commit your real password to GitHub — use environment variables in production (I can show you this next if you want).
- **Camera detection** uses the *visitor's own webcam* through their browser (not your laptop's camera) — this works fine once deployed, since it's all client-side JavaScript talking to your server.

## Checking usage stats
Visit `https://your-deployed-url/stats` any time to see total visits, unique users, and the most recent 20 visits.
