from flask import Flask, render_template, request, jsonify, session
import pickle
import sqlite3
import numpy as np
from datetime import datetime
from collections import defaultdict
import os

app = Flask(__name__)
app.secret_key = "fraud9000secretkey"

# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# ─────────────────────────────────────────
#  SQLITE SETUP
# ─────────────────────────────────────────

def get_db():
    db = sqlite3.connect("transactions_log.db")
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT,
            account_id          INTEGER,
            amount              INTEGER,
            hour_of_day         INTEGER,
            pin_attempts        INTEGER,
            is_new_location     INTEGER,
            txn_frequency_10min INTEGER,
            amount_vs_avg       REAL,
            verdict             TEXT,
            confidence          REAL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id  INTEGER PRIMARY KEY,
            card_number TEXT,
            pin         TEXT
        )
    """)
    db.commit()
    db.close()

init_db()

# ─────────────────────────────────────────
#  IN-MEMORY TRACKERS
# ─────────────────────────────────────────

recent_txns     = defaultdict(list)
account_history = defaultdict(list)

def get_txn_frequency(account_id):
    now = datetime.now()
    recent_txns[account_id] = [
        t for t in recent_txns[account_id]
        if (now - t).seconds < 600
    ]
    return len(recent_txns[account_id])

def get_amount_vs_avg(account_id, amount):
    history = account_history[account_id]
    if not history:
        return 0.0
    avg = np.mean(history)
    return round(abs(amount - avg) / avg, 3)

def update_trackers(account_id, amount):
    recent_txns[account_id].append(datetime.now())
    account_history[account_id].append(amount)
    if len(account_history[account_id]) > 50:
        account_history[account_id].pop(0)

# ─────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────

def predict(account_id, amount, hour, pin_attempts, is_new_location):
    txn_freq   = get_txn_frequency(account_id)
    amt_vs_avg = get_amount_vs_avg(account_id, amount)

    features = np.array([[
        amount, hour, pin_attempts,
        is_new_location, txn_freq, amt_vs_avg
    ]])

    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    verdict     = "FRAUD" if prediction == 1 else "SAFE"

    update_trackers(account_id, amount)
    return verdict, round(float(probability), 4), txn_freq, amt_vs_avg

def log_txn(account_id, amount, hour, pin_attempts,
            is_new_location, txn_freq, amt_vs_avg, verdict, confidence):
    db = get_db()
    db.execute("""
        INSERT INTO transactions (
            timestamp, account_id, amount, hour_of_day, pin_attempts,
            is_new_location, txn_frequency_10min, amount_vs_avg,
            verdict, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        account_id, amount, hour, pin_attempts,
        is_new_location, txn_freq, amt_vs_avg,
        verdict, confidence
    ))
    db.commit()
    db.close()

# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")

@app.route("/api/card", methods=["POST"])
def verify_card():
    data       = request.json
    card       = data.get("card", "").strip()

    if not card.isdigit() or len(card) < 4:
        return jsonify({"success": False, "error": "Card number must be at least 4 digits."})

    account_id = int(card) % 200
    session["account_id"] = account_id
    session["card"]       = card

    db      = get_db()
    account = db.execute(
        "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
    ).fetchone()
    db.close()

    is_new = account is None
    session["is_new_account"] = is_new

    return jsonify({
        "success":    True,
        "account_id": account_id,
        "is_new":     is_new,
        "message":    "New account detected. Please set a PIN." if is_new else "Card recognised. Enter your PIN."
    })

@app.route("/api/pin", methods=["POST"])
def verify_pin():
    data       = request.json
    pin        = data.get("pin", "").strip()
    is_new     = session.get("is_new_account", False)
    account_id = session.get("account_id")

    if account_id is None:
        return jsonify({"success": False, "error": "Session expired. Start again."})

    if is_new:
        db = get_db()
        db.execute(
            "INSERT INTO accounts (account_id, card_number, pin) VALUES (?, ?, ?)",
            (account_id, session.get("card"), pin)
        )
        db.commit()
        db.close()
        session["pin_attempts"] = 1
        return jsonify({"success": True, "message": "PIN registered successfully."})

    db      = get_db()
    account = db.execute(
        "SELECT pin FROM accounts WHERE account_id = ?", (account_id,)
    ).fetchone()
    db.close()

    attempts = session.get("pin_attempts", 0) + 1
    session["pin_attempts"] = attempts

    if account and account["pin"] == pin:
        return jsonify({"success": True, "message": "PIN accepted."})
    else:
        remaining = 5 - attempts
        if remaining <= 0:
            hour       = datetime.now().hour
            txn_freq   = get_txn_frequency(account_id)
            log_txn(account_id, 0, hour, attempts, 0, txn_freq, 0.0, "FRAUD", 0.99)
            session.clear()
            return jsonify({
                "success":   False,
                "locked":    True,
                "error":     "Too many wrong PIN attempts. Card blocked.",
                "verdict":   "FRAUD",
                "confidence": 0.99
            })
        return jsonify({
            "success":   False,
            "error":     f"Wrong PIN. {remaining} attempt(s) remaining."
        })

@app.route("/api/transaction", methods=["POST"])
def process_transaction():
    data            = request.json
    account_id      = session.get("account_id")
    pin_attempts    = session.get("pin_attempts", 1)

    if account_id is None:
        return jsonify({"success": False, "error": "Session expired. Start again."})

    try:
        amount          = int(data.get("amount", 0))
        is_new_location = int(data.get("is_new_location", 0))
    except ValueError:
        return jsonify({"success": False, "error": "Invalid amount."})

    if amount <= 0 or amount > 100000:
        return jsonify({"success": False, "error": "Amount must be between Rs.1 and Rs.100000."})

    # --- IMPOSSIBLE TRAVEL LOGIC ---
    last_txns = recent_txns.get(account_id, [])
    time_since_last_txn = None
    if last_txns:
        time_since_last_txn = (datetime.now() - last_txns[-1]).seconds

    hour = datetime.now().hour

    verdict, confidence, txn_freq, amt_vs_avg = predict(
        account_id, amount, hour, pin_attempts, is_new_location
    )

    warning_message = None
    # If used under 5 mins (300s) AND at a different location
    if time_since_last_txn is not None and time_since_last_txn < 300 and is_new_location == 1:
        confidence = min(0.98, confidence + 0.40) # Boost fraud confidence by 40%
        verdict = "FRAUD"
        warning_message = "Suspicious activity: This card was used at a different location just moments ago. This may be a fraudulent transaction."

    log_txn(account_id, amount, hour, pin_attempts,
            is_new_location, txn_freq, amt_vs_avg, verdict, confidence)

    def risk(value, low, high):
        if value <= low:  return "green"
        if value <= high: return "yellow"
        return "red"

    breakdown = [
        {"label": "Amount",                   "value": f"Rs.{amount}",                              "risk": risk(amount, 5000, 20000)},
        {"label": "Hour of transaction",      "value": f"{hour}:00 ({'normal' if 6<=hour<=22 else 'odd hours'})", "risk": "green" if 6<=hour<=22 else "red"},
        {"label": "PIN attempts",             "value": str(pin_attempts),                           "risk": risk(pin_attempts, 1, 2)},
        {"label": "New location",             "value": "Yes" if is_new_location else "No",          "risk": "red" if is_new_location else "green"},
        {"label": "Transactions (10 min)",    "value": str(txn_freq),                               "risk": risk(txn_freq, 2, 5)},
        {"label": "Amount vs average",        "value": f"{amt_vs_avg:.3f}",                         "risk": risk(amt_vs_avg, 0.3, 0.8)},
    ]

    # Add the warning to the visible UI breakdown
    if warning_message:
        breakdown.append({
            "label": "Velocity Mismatch", 
            "value": f"Location changed in {time_since_last_txn}s", 
            "risk": "red"
        })

    session.clear()

    return jsonify({
        "success":    True,
        "verdict":    verdict,
        "confidence": confidence,
        "breakdown":  breakdown,
        "warning":    warning_message
    })

@app.route("/api/log")
def get_log():
    db   = get_db()
    rows = db.execute("""
        SELECT id, timestamp, account_id, amount, verdict, confidence
        FROM transactions ORDER BY id DESC LIMIT 10
    """).fetchall()
    db.close()
    return jsonify([dict(row) for row in rows])

if __name__ == "__main__":
    app.run(debug=True)