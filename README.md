# 🏧 ATM Fraud Detection Simulator

A real-time ATM fraud detection system combining **Arduino hardware** with a **Python/scikit-learn ML backend**. Simulates an ATM interface using a keypad and LCD display, while a trained Random Forest model flags suspicious transactions on the fly.

---

## 🔍 Overview

This project bridges embedded hardware and machine learning to demonstrate how fraud detection works at the transaction level. A physical ATM interface (Arduino + keypad + LCD) sends transaction data to a Python backend, which runs inference using a trained classifier and returns a fraud/legitimate verdict in real time.

---

## 🏗️ Architecture

```
[Arduino + Keypad + LCD]
        │
        │  Serial / Socket
        ▼
[Python Backend (PC)]
        │
        ├── simulator.py       ← PC-side socket simulator (no hardware needed)
        ├── app.py             ← Main Flask app & API
        ├── serial_handler.py  ← Arduino serial communication
        ├── fraud_detecter_9000.py  ← Inference engine
        └── fraud_model.pkl    ← Trained Random Forest model (generated locally)
```

---

## ✨ Features

- 🔐 Real-time transaction fraud classification
- 🤖 Random Forest model with weighted loss for imbalanced fraud data
- 🖥️ PC-side socket simulator — test without physical hardware
- 📊 Confusion matrix comparison and model evaluation tools
- 🔌 Arduino serial interface for physical ATM simulation
- 📁 SQLite transaction logging

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Hardware | Arduino (Keypad + LCD) |
| Backend | Python 3, Flask |
| ML Model | scikit-learn (Random Forest) |
| Communication | Serial (Arduino) / Sockets (Simulator) |
| Data | Synthetic transaction dataset |
| Database | SQLite |

---

## 📁 Project Structure

```
atm-fraud-detection/
│
├── arduino/               # Arduino sketch for keypad + LCD interface
├── templates/
│   └── index.html         # Web dashboard
│
├── app.py                 # Flask app entry point
├── fraud_detecter_9000.py # Core fraud detection logic
├── generate_data.py       # Synthetic training data generator
├── train_model.py         # Model training script
├── simulator.py           # PC-side ATM simulator (no hardware needed)
├── serial_handler.py      # Arduino serial communication handler
├── confusion_matrix_comparison.py  # Model evaluation utility
└── transactions.csv       # Synthetic transaction dataset
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Arduino IDE (only if using physical hardware)

### Installation

```bash
# Clone the repo
git clone https://github.com/Adityan-R/atm-fraud-detection.git
cd atm-fraud-detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Generate Data & Train Model

```bash
python generate_data.py   # Creates synthetic transaction data
python train_model.py     # Trains the Random Forest model → saves fraud_model.pkl
```

### Run (No Hardware)

```bash
python simulator.py   # Starts the PC-side ATM socket simulator
python app.py         # Starts the Flask backend
```

### Run (With Arduino)
1. Upload the sketch from `/arduino/` to your Arduino board
2. Connect via USB
3. Run `python app.py` — serial communication is handled automatically

---

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Training Data**: Synthetically generated transaction records
- **Class Imbalance Handling**: Weighted loss / class weights
- **Evaluation**: Confusion matrix, precision, recall, F1-score

---

## 🗺️ Roadmap

- [x] Phase 1 — Synthetic data generation & model training
- [x] Phase 2 — Flask backend & fraud inference engine
- [ ] Phase 3 — PC-side socket simulator (`simulator.py`)
- [ ] Phase 4 — Full Arduino hardware deployment

---

## 👤 Author

**Adityan R**  
BTech CSE (AI/ML) — Lovely Professional University  
[GitHub](https://github.com/Adityan-R)

---

## 📄 License

This project is for academic and educational purposes.
