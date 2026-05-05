import serial
import serial.tools.list_ports
import pickle
import sqlite3
import numpy as np
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────
#  STEP 1 — LOAD THE TRAINED MODEL
# ─────────────────────────────────────────

print("=" * 50)
print("  ATM FRAUD DETECTION — SERIAL HANDLER")
print("=" * 50)

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

print("\n✅ Model loaded from fraud_model.pkl")

# ─────────────────────────────────────────
#  STEP 2 — SETUP SQLITE LOGGING
#  Every transaction gets logged here
#  regardless of verdict
# ─────────────────────────────────────────

conn = sqlite3.connect("transactions_log.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT,
        amount              INTEGER,
        hour_of_day         INTEGER,
        pin_attempts        INTEGER,
        is_new_location     INTEGER,
        txn_frequency_10min INTEGER,
        amount_vs_avg       REAL,
        verdict             TEXT
    )
""")
conn.commit()
print("✅ SQLite database ready → transactions_log.db")

# ─────────────────────────────────────────
#  STEP 3 — AUTO DETECT ARDUINO PORT
# ─────────────────────────────────────────

def find_arduino_port():
    """Scan all COM ports and return the one most likely to be Arduino."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Arduino usually shows up as USB Serial Device or CH340
        if any(keyword in port.description for keyword in
               ["Arduino", "CH340", "USB Serial", "ttyUSB", "ttyACM"]):
            return port.device
    # If nothing matched, list available ports for manual selection
    if ports:
        print("\n⚠️  Could not auto-detect Arduino. Available ports:")
        for i, p in enumerate(ports):
            print(f"   [{i}] {p.device} — {p.description}")
        idx = int(input("   Enter port number: "))
        return ports[idx].device
    return None

# ─────────────────────────────────────────
#  STEP 4 — TRANSACTION FREQUENCY TRACKER
#  Tracks how many transactions each
#  account did in the last 10 minutes
# ─────────────────────────────────────────

# account_id → list of timestamps
recent_transactions = defaultdict(list)

def get_txn_frequency(account_id):
    """Return how many transactions this account did in last 10 mins."""
    now = datetime.now()
    # Keep only last 10 minutes
    recent_transactions[account_id] = [
        t for t in recent_transactions[account_id]
        if (now - t).seconds < 600
    ]
    return len(recent_transactions[account_id])

def record_transaction(account_id):
    """Log this transaction timestamp for frequency tracking."""
    recent_transactions[account_id].append(datetime.now())

# ─────────────────────────────────────────
#  STEP 5 — AVERAGE AMOUNT TRACKER
#  Tracks each account's average withdrawal
#  to compute amount_vs_avg in real time
# ─────────────────────────────────────────

account_history = defaultdict(list)   # account_id → list of amounts

def get_amount_vs_avg(account_id, amount):
    """How far is this amount from the account's historical average?"""
    history = account_history[account_id]
    if len(history) == 0:
        return 0.0   # first transaction — no baseline yet
    avg = np.mean(history)
    return round(abs(amount - avg) / avg, 3)

def update_amount_history(account_id, amount):
    account_history[account_id].append(amount)
    # Keep only last 50 transactions to avoid memory bloat
    if len(account_history[account_id]) > 50:
        account_history[account_id].pop(0)

# ─────────────────────────────────────────
#  STEP 6 — PARSE INCOMING SERIAL DATA
#  Arduino sends a string like:
#  "ACCT:42,AMT:5000,PIN:1,LOC:0,HR:14"
# ─────────────────────────────────────────

def parse_transaction(raw):
    """Parse the serial string from Arduino into a dict."""
    try:
        parts = raw.strip().split(",")
        data = {}
        for part in parts:
            key, val = part.split(":")
            data[key.strip()] = val.strip()
        return {
            "account_id":   int(data["ACCT"]),
            "amount":       int(data["AMT"]),
            "pin_attempts": int(data["PIN"]),
            "is_new_location": int(data["LOC"]),
            "hour_of_day":  int(data["HR"]),
        }
    except Exception as e:
        print(f"⚠️  Parse error: {e} | Raw: {raw}")
        return None

# ─────────────────────────────────────────
#  STEP 7 — RUN FRAUD PREDICTION
# ─────────────────────────────────────────

def predict(txn):
    """Build feature vector and run through the loaded model."""
    account_id = txn["account_id"]
    amount     = txn["amount"]

    txn_freq    = get_txn_frequency(account_id)
    amt_vs_avg  = get_amount_vs_avg(account_id, amount)

    features = np.array([[
        amount,
        txn["hour_of_day"],
        txn["pin_attempts"],
        txn["is_new_location"],
        txn_freq,
        amt_vs_avg
    ]])

    prediction   = model.predict(features)[0]
    probability  = model.predict_proba(features)[0][1]  # fraud probability
    verdict      = "FRAUD" if prediction == 1 else "SAFE"

    # Update trackers after prediction
    record_transaction(account_id)
    update_amount_history(account_id, amount)

    return verdict, probability, txn_freq, amt_vs_avg

# ─────────────────────────────────────────
#  STEP 8 — LOG TO SQLITE
# ─────────────────────────────────────────

def log_to_db(txn, txn_freq, amt_vs_avg, verdict):
    cursor.execute("""
        INSERT INTO transactions (
            timestamp, amount, hour_of_day, pin_attempts,
            is_new_location, txn_frequency_10min, amount_vs_avg, verdict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        txn["amount"],
        txn["hour_of_day"],
        txn["pin_attempts"],
        txn["is_new_location"],
        txn_freq,
        amt_vs_avg,
        verdict
    ))
    conn.commit()

# ─────────────────────────────────────────
#  STEP 9 — MAIN LOOP
#  Connects to Arduino and processes
#  transactions in real time
# ─────────────────────────────────────────

def main():
    port = find_arduino_port()
    if not port:
        print("\n❌ No Arduino found. Make sure it's plugged in via USB.")
        return

    print(f"\n🔌 Connecting to Arduino on {port}...")

    try:
        ser = serial.Serial(port, 9600, timeout=5)
        print(f"✅ Connected. Waiting for transactions...\n")
        print("-" * 50)

        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()

            if not raw or not raw.startswith("ACCT:"):
                # Skip empty lines or debug messages from Arduino
                if raw:
                    print(f"[Arduino] {raw}")
                continue

            print(f"\n📨 Received: {raw}")

            txn = parse_transaction(raw)
            if txn is None:
                ser.write(b"ERROR\n")
                continue

            verdict, prob, txn_freq, amt_vs_avg = predict(txn)

            # Log to database
            log_to_db(txn, txn_freq, amt_vs_avg, verdict)

            # Print to terminal
            status_icon = "🚨" if verdict == "FRAUD" else "✅"
            print(f"   Account       : {txn['account_id']}")
            print(f"   Amount        : ₹{txn['amount']}")
            print(f"   Hour          : {txn['hour_of_day']}:00")
            print(f"   PIN Attempts  : {txn['pin_attempts']}")
            print(f"   New Location  : {'Yes' if txn['is_new_location'] else 'No'}")
            print(f"   Txn Freq      : {txn_freq} in last 10 min")
            print(f"   Amt vs Avg    : {amt_vs_avg}")
            print(f"   Fraud Prob    : {prob:.2%}")
            print(f"   Verdict       : {status_icon} {verdict}")
            print("-" * 50)

            # Send verdict back to Arduino
            ser.write(f"{verdict}\n".encode())

    except serial.SerialException as e:
        print(f"\n❌ Serial error: {e}")
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        conn.close()
        print("✅ Database connection closed.")


if __name__ == "__main__":
    main()