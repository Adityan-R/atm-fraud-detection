import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# ─────────────────────────────────────────
#  configuration
# ─────────────────────────────────────────
TOTAL_TRANSACTIONS = 10000
FRAUD_RATIO = 0.10          # 10% fraud — realistic and keeps dataset useful
RANDOM_SEED = 42
OUTPUT_FILE = "transactions.csv"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ─────────────────────────────────────────
#  user profile pool
#  simulates different spending habits
# ─────────────────────────────────────────
USER_PROFILES = {
    "low_spender":    {"avg_amount": 1500,  "std": 500},
    "mid_spender":    {"avg_amount": 5000,  "std": 1500},
    "high_spender":   {"avg_amount": 15000, "std": 4000},
}

def generate_user_pool(n=200):
    """Create a pool of users, each with a spending profile and home location."""
    users = {}
    profiles = list(USER_PROFILES.keys())
    for uid in range(n):
        profile_name = random.choice(profiles)
        profile = USER_PROFILES[profile_name]
        users[uid] = {
            "profile": profile_name,
            "avg_amount": profile["avg_amount"],
            "std_amount": profile["std"],
            "home_atm": random.randint(0, 49),
        }
    return users

# ─────────────────────────────────────────
#  transaction generator
# ─────────────────────────────────────────

def generate_legitimate_transaction(user_id, users, recent_txn_times):
    """Generate a realistic, non-fraudulent transaction."""
    user = users[user_id]

    hour = int(np.clip(np.random.normal(14, 4), 6, 22))
    amount = max(100, int(np.random.normal(user["avg_amount"], user["std_amount"])))
    amount = min(amount, 50000)

    atm_id = user["home_atm"] if random.random() < 0.85 else random.randint(0, 49)
    is_new_location = int(atm_id != user["home_atm"])

    pin_attempts = 1
    if random.random() < 0.05:
        pin_attempts = 2

    now = datetime.now()
    recent = [t for t in recent_txn_times.get(user_id, []) if (now - t).seconds < 600]
    txn_frequency = len(recent)

    amount_vs_avg = round(abs(amount - user["avg_amount"]) / user["avg_amount"], 3)

    return {
        "user_id": user_id,
        "amount": amount,
        "hour_of_day": hour,
        "pin_attempts": pin_attempts,
        "is_new_location": is_new_location,
        "txn_frequency_10min": min(txn_frequency, 10),
        "amount_vs_avg": amount_vs_avg,
        "label": 0
    }


def generate_fraudulent_transaction(user_id, users, recent_txn_times):
    """Generate a fraudulent transaction with realistic fraud patterns."""
    user = users[user_id]

    pattern = random.choices(
        ["odd_hours", "high_amount", "pin_bruteforce", "rapid_succession", "combined"],
        weights=[0.20, 0.25, 0.20, 0.15, 0.20]
    )[0]

    hour = random.randint(8, 20)
    amount = int(np.random.normal(user["avg_amount"], user["std_amount"]))
    pin_attempts = 1
    atm_id = user["home_atm"]
    is_new_location = 0

    if pattern == "odd_hours":
        hour = random.choice(list(range(0, 5)) + list(range(23, 24)))
        amount = int(np.random.normal(user["avg_amount"] * 1.5, user["std_amount"]))

    elif pattern == "high_amount":
        amount = int(np.random.normal(user["avg_amount"] * 3.5, user["std_amount"] * 0.5))
        is_new_location = 1

    elif pattern == "pin_bruteforce":
        pin_attempts = random.randint(3, 5)
        amount = int(np.random.normal(user["avg_amount"] * 2, user["std_amount"]))

    elif pattern == "rapid_succession":
        now = datetime.now()
        recent_txn_times[user_id] = [now - timedelta(seconds=s) for s in range(0, 600, 90)]
        amount = int(np.random.normal(user["avg_amount"], user["std_amount"] * 0.3))

    elif pattern == "combined":
        hour = random.choice(list(range(0, 6)))
        amount = int(np.random.normal(user["avg_amount"] * 4, user["std_amount"]))
        pin_attempts = random.randint(2, 4)
        is_new_location = 1

    amount = max(100, min(amount, 100000))
    pin_attempts = max(1, min(pin_attempts, 5))

    now = datetime.now()
    recent = [t for t in recent_txn_times.get(user_id, []) if (now - t).seconds < 600]
    txn_frequency = len(recent)

    amount_vs_avg = round(abs(amount - user["avg_amount"]) / user["avg_amount"], 3)

    return {
        "user_id": user_id,
        "amount": amount,
        "hour_of_day": hour,
        "pin_attempts": pin_attempts,
        "is_new_location": is_new_location,
        "txn_frequency_10min": min(txn_frequency, 10),
        "amount_vs_avg": amount_vs_avg,
        "label": 1
    }


# ─────────────────────────────────────────
#  main loop for generation 
# ─────────────────────────────────────────

def generate_dataset():
    users = generate_user_pool(n=200)
    recent_txn_times = {}
    records = []

    n_fraud = int(TOTAL_TRANSACTIONS * FRAUD_RATIO)
    n_legit = TOTAL_TRANSACTIONS - n_fraud

    print(f"Generating {n_legit} legitimate transactions...")
    for _ in range(n_legit):
        uid = random.randint(0, 199)
        txn = generate_legitimate_transaction(uid, users, recent_txn_times)
        records.append(txn)
        recent_txn_times.setdefault(uid, []).append(datetime.now())

    print(f"Generating {n_fraud} fraudulent transactions...")
    for _ in range(n_fraud):
        uid = random.randint(0, 199)
        txn = generate_fraudulent_transaction(uid, users, recent_txn_times)
        records.append(txn)

    random.shuffle(records)

    df = pd.DataFrame(records)

    print(f"\nDataset shape     : {df.shape}")
    print(f"Fraud count       : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Legit count       : {(df['label']==0).sum()}")
    print(f"\nFeature stats:\n{df.drop(columns=['user_id','label']).describe().round(2)}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    generate_dataset()