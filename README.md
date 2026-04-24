# 🧠 MindGuard – AI-Based Attention Tracking System

MindGuard is an AI-powered attention monitoring system that analyzes user focus in real-time using computer vision, audio processing, and intelligent interventions.

---

## 🚀 Features

- 🎥 Real-time face tracking with MediaPipe  
- 👁️ Eye Aspect Ratio (EAR) for attention detection  
- 😴 Drowsiness detection (blink rate, eye closure, yawning)  
- 🧭 Head pose estimation (yaw & pitch)  
- 🎯 Hybrid attention scoring system (EAR + Head Pose + Drowsiness)  
- 📊 Real-time attention smoothing & state machine logic  
- 🎤 Speech-to-text using Whisper (ASR)  
- 🧠 Keyword extraction (basic NLP)  
- ❓ Dynamic quiz system:  
  - 📚 Predefined quiz bank  
  - 🤖 LLM-based question generation (Ollama)  
- 🔁 Smart intervention system (quiz triggered when attention drops)  
- 📝 Logging system (CSV & JSONL)  

---

## 🧠 How It Works

1. Webcam captures facial landmarks using MediaPipe  
2. Attention score is calculated based on:  
   - Eye movement (EAR)  
   - Head position  
   - Drowsiness indicators (blink, yawn, eye closure)  
3. Audio input is transcribed using Whisper  
4. Keywords are extracted from speech  
5. If attention drops below a threshold:  
   - A quiz is triggered  
   - Questions are selected or generated based on topic  
6. All interactions are logged for analysis  

---

## ⚙️ Setup & Run

```bash
# Clone the repository
git clone https://github.com/eksiesen/MindGuard.git
cd MindGuard

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# (Mac/Linux için)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
## 🛠️ Technologies

- Python
- Streamlit
- OpenCV
- MediaPipe
- Whisper (ASR)
- Ollama (LLM)
- NumPy

## 📌 Use Case

Designed as a prototype for:
- Smart classroom systems
- Student attention monitoring
- Adaptive learning environments

## ⚠️ Note

This project is a prototype and does not include production-level optimization or privacy guarantees.
