import socket
import time
import random
from datetime import datetime

# ─────────────────────────────────────────
#  CONFIGURATION
#  Must match serial_handler.py settings
# ─────────────────────────────────────────

HOST = "127.0.0.1"   # localhost
PORT = 65432         # any unused port

# ─────────────────────────────────────────
#  FRAUD SCENARIO PRESETS
#  For quick testing without manual input
# ─────────────────────────────────────────

SCENARIOS = {
    "1": {
        "name": "Normal Transaction",
        "data": {"acct": 42, "amt": 3000, "pin": 1, "loc": 0, "hr": 14},
    },
    "2": {
        "name": "Odd Hours",
        "data": {"acct": 42, "amt": 4000, "pin": 1, "loc": 0, "hr": 2},
    },
    "3": {
        "name": "PIN Brute Force",
        "data": {"acct": 42, "amt": 5000, "pin": 4, "loc": 0, "hr": 15},
    },
    "4": {
        "name": "High Amount + New Location",
        "data": {"acct": 42, "amt": 45000, "pin": 1, "loc": 1, "hr": 13},
    },
    "5": {
        "name": "Combined Fraud (all signals)",
        "data": {"acct": 42, "amt": 60000, "pin": 3, "loc": 1, "hr": 3},
    },
    "6": {
        "name": "Rapid Succession (send 5 times fast)",
        "data": {"acct": 99, "amt": 2000, "pin": 1, "loc": 0, "hr": 16},
        "repeat": 5
    },
}

# ─────────────────────────────────────────
#  FORMAT TRANSACTION
#  Matches exactly what Arduino will send
# ─────────────────────────────────────────

def format_txn(acct, amt, pin, loc, hr):
    return f"ACCT:{acct},AMT:{amt},PIN:{pin},LOC:{loc},HR:{hr}"

# ─────────────────────────────────────────
#  DISPLAY MENU
# ─────────────────────────────────────────

def show_menu():
    print("\n" + "=" * 50)
    print("  ATM FRAUD SIMULATOR — INPUT TERMINAL")
    print("=" * 50)
    print("  Choose a scenario or enter manually:\n")
    for key, val in SCENARIOS.items():
        print(f"  [{key}] {val['name']}")
    print("  [M] Manual entry")
    print("  [R] Random transaction")
    print("  [Q] Quit")
    print("=" * 50)

# ─────────────────────────────────────────
#  MANUAL ENTRY
# ─────────────────────────────────────────

def manual_entry():
    print("\n  Enter transaction details:")
    try:
        acct = int(input("  Account ID (0-199)   : "))
        amt  = int(input("  Amount (₹)           : "))
        pin  = int(input("  PIN attempts (1-5)   : "))
        loc  = int(input("  New location? (0/1)  : "))
        hr   = int(input("  Hour (0-23)          : "))
        return {"acct": acct, "amt": amt, "pin": pin, "loc": loc, "hr": hr}
    except ValueError:
        print("  ⚠️  Invalid input. Try again.")
        return None

# ─────────────────────────────────────────
#  RANDOM TRANSACTION GENERATOR
# ─────────────────────────────────────────

def random_transaction():
    now = datetime.now()
    return {
        "acct": random.randint(0, 199),
        "amt":  random.choice([
            random.randint(500, 8000),      # normal range
            random.randint(20000, 80000),   # suspicious range
        ]),
        "pin": random.choices([1, 2, 3, 4, 5], weights=[70, 15, 8, 4, 3])[0],
        "loc": random.choices([0, 1], weights=[85, 15])[0],
        "hr":  now.hour,
    }

# ─────────────────────────────────────────
#  SEND TRANSACTION AND GET VERDICT
# ─────────────────────────────────────────

def send_transaction(sock, txn_data, label=""):
    message = format_txn(
        txn_data["acct"],
        txn_data["amt"],
        txn_data["pin"],
        txn_data["loc"],
        txn_data["hr"]
    )

    print(f"\n  📤 Sending : {message}")
    sock.sendall((message + "\n").encode())

    # Wait for verdict from serial_handler.py
    verdict = sock.recv(1024).decode().strip()

    icon = "🚨" if verdict == "FRAUD" else "✅"
    print(f"  📥 Verdict : {icon} {verdict}")
    if label:
        print(f"  📋 Scenario: {label}")

    return verdict

# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────

def main():
    print("\n🔌 Connecting to serial_handler.py...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        print("✅ Connected to backend!\n")

    except ConnectionRefusedError:
        print("\n❌ Could not connect.")
        print("   Make sure serial_handler.py is running first.")
        print("   Run it in a separate terminal with: python serial_handler.py")
        return

    try:
        while True:
            show_menu()
            choice = input("\n  Your choice: ").strip().upper()

            if choice == "Q":
                print("\n👋 Exiting simulator.")
                break

            elif choice == "M":
                txn_data = manual_entry()
                if txn_data:
                    send_transaction(sock, txn_data, "Manual Entry")

            elif choice == "R":
                txn_data = random_transaction()
                send_transaction(sock, txn_data, "Random")

            elif choice in SCENARIOS:
                scenario = SCENARIOS[choice]
                repeat = scenario.get("repeat", 1)

                for i in range(repeat):
                    if repeat > 1:
                        print(f"\n  🔁 Sending {i+1}/{repeat}...")
                    send_transaction(sock, scenario["data"], scenario["name"])
                    if repeat > 1:
                        time.sleep(0.5)   # small delay between rapid transactions

            else:
                print("  ⚠️  Invalid choice. Try again.")

    except KeyboardInterrupt:
        print("\n\n👋 Simulator stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()