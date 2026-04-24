import streamlit as st
import cv2
import mediapipe as mp
import math
import time
import numpy as np
import random
import os
import csv
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Set, FrozenSet
import atexit
import re
import tempfile
import shutil
import traceback
import sys
import wave
import threading
from collections import deque
import requests

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from quiz_bank import QUIZZES   # 👈 MODÜLER QUIZ


# ======================
# EAR / MAR FONKSİYONLARI
# ======================
def euclidean(p1, p2):
    return math.dist(p1, p2)

def compute_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def ear_to_attention_score(ear):
    min_ear = 0.15
    max_ear = 0.35
    ear = max(min(ear, max_ear), min_ear)
    return int((ear - min_ear) / (max_ear - min_ear) * 100)

def compute_mar(mouth_pts):
    """
    Mouth Aspect Ratio (MAR) = vertical / horizontal
    mouth_pts: [left_corner, right_corner, upper_lip, lower_lip] in pixel
    """
    left_corner, right_corner, upper_lip, lower_lip = mouth_pts
    horizontal = euclidean(left_corner, right_corner) + 1e-6
    vertical = euclidean(upper_lip, lower_lip)
    return vertical / horizontal


# ======================
# PARAMETRELER (v0.1)
# ======================
LOW_THRESHOLD = 40              # düşük dikkat eşiği (FINAL score için)
LOW_DURATION_REQUIRED = 3.0     # saniye: düşük kalma süresi
COOLDOWN_DURATION = 120.0       # saniye: quiz sonrası kilit
SMOOTH_WINDOW_SECONDS = 3.0     # saniye: smoothing penceresi

# Quiz tekrar önleme: son N soruda tekrar etme
QUIZ_RECENT_WINDOW = 5

# Loglama
LOG_DIR = "logs"
SAMPLE_LOG_INTERVAL_SEC = 1.0  # her kaç saniyede bir sample yazılsın
RECOVERY_THRESHOLD = 60        # quiz sonrası "toparlanma" eşiği (final_smoothed)
POST_QUIZ_AVG_WINDOW_SEC = 30.0

# Hibrit skor ağırlıkları (EAR + HeadPose + Drowsy)
W_EAR = 0.55
W_HEADPOSE = 0.30
W_DROWSY = 0.15

# Drowsy eşikleri
EAR_CLOSED_SCORE_TH = 20        # ear_score <= 20 -> "kapalı gibi"
EYE_CLOSURE_REQUIRED = 1.5      # saniye: uzun göz kapanma -> drowsy

MAR_YAWN_TH = 0.65              # MAR >= 0.65 -> esneme adayı
YAWN_REQUIRED = 1.0             # saniye: esneme süresi

BLINK_WINDOW_SEC = 30.0         # blink sayımı penceresi
BLINK_BPM_DROWSY_TH = 35.0      # blinks per minute yüksekse drowsy riski

# Head pose skor aralıkları
YAW_TIGHT = 15
PITCH_TIGHT = 10
YAW_MED = 30
PITCH_MED = 20


# ======================
# SESSION STATE (INIT)
# ======================
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False

# quiz selection/session
if "active_quiz_id" not in st.session_state:
    st.session_state.active_quiz_id = None

if "quiz_session_id" not in st.session_state:
    st.session_state.quiz_session_id = 0  # her yeni quizte artar (radio key reset)

if "quiz_history_ids" not in st.session_state:
    st.session_state.quiz_history_ids = []  # gösterilmiş quiz id'leri (sıralı)

if "last_quiz_result" not in st.session_state:
    # phase-1: sadece hafif bir özet tutuyoruz (phase-2'de loga dönecek)
    st.session_state.last_quiz_result = None

# logging session state
if "logging_enabled" not in st.session_state:
    st.session_state.logging_enabled = True

if "intervention_enabled" not in st.session_state:
    # Quiz müdahalesi varsayılan açık (mevcut davranışı bozmasın)
    st.session_state.intervention_enabled = True

# NLP v0.1 (manuel metin -> keyword -> tag seçimi)
if "lesson_text" not in st.session_state:
    st.session_state.lesson_text = ""

if "nlp_keywords" not in st.session_state:
    st.session_state.nlp_keywords = []

if "preferred_quiz_tags" not in st.session_state:
    st.session_state.preferred_quiz_tags = []

# Whisper ASR (Faz 5)
if "asr_text" not in st.session_state:
    st.session_state.asr_text = ""

if "whisper_model_name" not in st.session_state:
    st.session_state.whisper_model_name = "base"

if "whisper_loaded_name" not in st.session_state:
    st.session_state.whisper_loaded_name = None

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

if "asr_last_error" not in st.session_state:
    st.session_state.asr_last_error = None

# Live ASR (Faz 5) - mikrofon
if "live_asr_enabled" not in st.session_state:
    st.session_state.live_asr_enabled = False
if "live_asr_window_sec" not in st.session_state:
    st.session_state.live_asr_window_sec = 20.0
if "live_asr_transcribe_every_sec" not in st.session_state:
    st.session_state.live_asr_transcribe_every_sec = 10.0

if "live_asr_stop_event" not in st.session_state:
    st.session_state.live_asr_stop_event = None
if "live_asr_thread" not in st.session_state:
    st.session_state.live_asr_thread = None
if "live_asr_worker_thread" not in st.session_state:
    st.session_state.live_asr_worker_thread = None
if "live_asr_running" not in st.session_state:
    st.session_state.live_asr_running = False
if "live_asr_audio_lock" not in st.session_state:
    st.session_state.live_asr_audio_lock = None
if "live_asr_audio_buffer" not in st.session_state:
    st.session_state.live_asr_audio_buffer = None
if "live_asr_audio_samples" not in st.session_state:
    st.session_state.live_asr_audio_samples = 0

if "live_asr_last_update_ts" not in st.session_state:
    st.session_state.live_asr_last_update_ts = None
if "live_asr_last_used_text" not in st.session_state:
    st.session_state.live_asr_last_used_text = ""
if "live_asr_tags_updated" not in st.session_state:
    st.session_state.live_asr_tags_updated = False
if "live_asr_device_index" not in st.session_state:
    st.session_state.live_asr_device_index = None  # None = sounddevice default
if "live_asr_level_peak" not in st.session_state:
    st.session_state.live_asr_level_peak = 0.0
if "live_asr_level_rms" not in st.session_state:
    st.session_state.live_asr_level_rms = 0.0
if "live_asr_last_error" not in st.session_state:
    st.session_state.live_asr_last_error = None

if "llm_quiz_enabled" not in st.session_state:
    st.session_state.llm_quiz_enabled = True
if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = "llama3.2:3b"
if "llm_ollama_url" not in st.session_state:
    st.session_state.llm_ollama_url = "http://localhost:11434"
if "llm_temperature" not in st.session_state:
    st.session_state.llm_temperature = 0.2
if "llm_last_error" not in st.session_state:
    st.session_state.llm_last_error = None

if "log_session_id" not in st.session_state:
    st.session_state.log_session_id = str(uuid.uuid4())

if "log_csv_path" not in st.session_state:
    st.session_state.log_csv_path = None

if "log_jsonl_path" not in st.session_state:
    st.session_state.log_jsonl_path = None

if "last_sample_log_ts" not in st.session_state:
    st.session_state.last_sample_log_ts = 0.0

if "quiz_shown_ts" not in st.session_state:
    st.session_state.quiz_shown_ts = None
if "active_generated_quiz" not in st.session_state:
    st.session_state.active_generated_quiz = None

# derived metric trackers (online)
if "last_trigger_latency_sec" not in st.session_state:
    st.session_state.last_trigger_latency_sec = None

if "post_quiz_start_ts" not in st.session_state:
    st.session_state.post_quiz_start_ts = None

if "post_quiz_scores" not in st.session_state:
    st.session_state.post_quiz_scores = []  # (ts, score)

if "post_quiz_avg_done" not in st.session_state:
    st.session_state.post_quiz_avg_done = False

if "recovery_start_ts" not in st.session_state:
    st.session_state.recovery_start_ts = None

if "recovered_done" not in st.session_state:
    st.session_state.recovered_done = False

# state machine
if "mg_state" not in st.session_state:
    st.session_state.mg_state = "IDLE"  # IDLE, LOW, TRIGGERED, COOLDOWN

if "low_start_time" not in st.session_state:
    st.session_state.low_start_time = None

if "cooldown_until" not in st.session_state:
    st.session_state.cooldown_until = 0.0

# smoothing buffer: (timestamp, score)
if "score_buffer" not in st.session_state:
    st.session_state.score_buffer = []

# drowsy trackers
if "eye_closed_start" not in st.session_state:
    st.session_state.eye_closed_start = None

if "yawn_start" not in st.session_state:
    st.session_state.yawn_start = None

if "blink_times" not in st.session_state:
    st.session_state.blink_times = []

if "prev_eye_closed" not in st.session_state:
    st.session_state.prev_eye_closed = False


# ======================
# UI
# ======================
st.title("MindGuard – Dikkat Takibi Prototipi")
run = st.checkbox("Kamerayı Başlat")

with st.expander("📄 Loglama / Ölçüm (Pilot için)"):
    st.session_state.logging_enabled = st.checkbox("Loglamayı etkinleştir", value=st.session_state.logging_enabled)
    st.session_state.intervention_enabled = st.checkbox(
        "Müdahale (Quiz) aktif",
        value=st.session_state.intervention_enabled,
        help="Kapalıysa dikkat düşüşü izlenir/loglanır ama quiz tetiklenmez (kontrol bloğu için).",
    )
    st.caption(
        "Kayıtlar sadece sayısal metrik içerir (görüntü kaydı yok). "
        "CSV: saniyelik örnekler | JSONL: event + sample birlikte."
    )

with st.expander("📝 Ders Metni (NLP v0.1)"):
    st.session_state.lesson_text = st.text_area(
        "Ders metnini buraya yapıştırın (manuel).",
        value=st.session_state.lesson_text,
        height=160,
        placeholder="Örn: Üçgenin iç açıları toplamı 180 derecedir...",
    )

with st.expander("🎙️ Whisper (ASR)"):
    st.caption("Ses dosyası yükleyip Whisper ile metne çevirir. (Faz 5)")

    st.session_state.whisper_model_name = st.selectbox(
        "Whisper modeli",
        options=["tiny", "base", "small", "medium"],
        index=["tiny", "base", "small", "medium"].index(st.session_state.whisper_model_name)
        if st.session_state.whisper_model_name in ["tiny", "base", "small", "medium"]
        else 1,
        help="Model büyüdükçe doğruluk artar ama daha yavaş ve daha ağır olur.",
    )

    uploaded_audio = st.file_uploader(
        "Ses dosyası yükle (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a", "mp4"],
        accept_multiple_files=False,
    )

    st.session_state.live_asr_enabled = st.checkbox(
        "Mikrofondan canlı transkripsiyon (Faz 5)",
        value=st.session_state.live_asr_enabled,
        help="Kamerayı başlatınca arka planda mikrofon kaydı alıp periyodik Whisper transkripsiyonu yapar.",
    )
    # Mikrofon cihaz seçimi (teşhis için çok kritik)
    # - Cihaz listesini sadece gerekince sorgula
    try:
        import sounddevice as sd  # type: ignore

        _asr_input_devs = [
            (i, d)
            for (i, d) in enumerate(sd.query_devices())
            if int(d.get("max_input_channels") or 0) > 0
        ]
        _asr_dev_labels = [
            f"{i} — {d.get('name')} (sr:{int(float(d.get('default_samplerate') or 0))})"
            for (i, d) in _asr_input_devs
        ]
        # default input (may be None on some systems)
        _default_in = None
        try:
            _default_in = sd.default.device[0]
        except Exception:
            _default_in = None

        if _asr_input_devs:
            # Selectbox index: keep stable by mapping device_index -> position in list
            _idx_by_dev = {dev_i: pos for (pos, (dev_i, _d)) in enumerate(_asr_input_devs)}
            _current_dev = st.session_state.live_asr_device_index
            if _current_dev is None and _default_in in _idx_by_dev:
                _current_pos = _idx_by_dev[_default_in]
            elif _current_dev in _idx_by_dev:
                _current_pos = _idx_by_dev[_current_dev]
            else:
                _current_pos = 0

            _chosen_label = st.selectbox(
                "Mikrofon cihazı (Live ASR)",
                options=_asr_dev_labels,
                index=int(_current_pos),
                help="Transkript hiç gelmiyorsa genelde yanlış giriş cihazı seçilidir. Buradan doğru mikrofonu seçin.",
            )
            _chosen_dev = int(_chosen_label.split("—", 1)[0].strip())
            st.session_state.live_asr_device_index = _chosen_dev
        else:
            st.info("Live ASR için giriş mikrofonu bulunamadı (sounddevice cihaz listesi boş).")
    except Exception:
        # sounddevice yoksa veya cihaz listesi sorgulanamadıysa sessiz geç
        pass

    st.session_state.live_asr_window_sec = st.slider(
        "Transkript penceresi (sn)",
        min_value=5.0,
        max_value=40.0,
        value=float(st.session_state.live_asr_window_sec),
        step=1.0,
    )
    st.session_state.live_asr_transcribe_every_sec = st.slider(
        "Ne sıklıkla transkripte çevirilsin (sn)",
        min_value=5.0,
        max_value=30.0,
        value=float(st.session_state.live_asr_transcribe_every_sec),
        step=1.0,
    )

    st.markdown("---")
    st.subheader("🧠 Soru üretimi (LLM) – kontrollü")
    st.session_state.llm_quiz_enabled = st.checkbox(
        "Konuya göre yeni soru üret (4 seçenekli)",
        value=bool(st.session_state.llm_quiz_enabled),
        help="Tag/anahtar kelimelere göre LLM'den yeni soru üretir. Üretim başarısızsa quiz bank'a düşer.",
    )
    st.caption(
        "Ollama (lokal) üzerinden üretim dener. Ollama çalışmıyorsa veya çıktı geçersizse otomatik olarak quiz bank'a düşer."
    )
    st.session_state.llm_ollama_url = st.text_input(
        "Ollama URL",
        value=str(st.session_state.llm_ollama_url),
        help="Varsayılan: http://localhost:11434",
    )
    st.session_state.llm_model_name = st.text_input(
        "Model adı",
        value=str(st.session_state.llm_model_name),
        help="Ollama model adı. Örn: llama3.1:8b veya llama3.2:3b",
    )
    st.session_state.llm_temperature = st.slider(
        "Üretim sıcaklığı",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.llm_temperature),
        step=0.05,
    )
    if st.session_state.llm_last_error:
        st.warning(f"LLM hata (son): {st.session_state.llm_last_error}")
    def _ensure_ffmpeg_on_path() -> Optional[str]:
        """
        Whisper, WAV dahil olmak üzere ffmpeg'i subprocess ile çağırır.
        Önce sistem PATH'inde arar; yoksa imageio-ffmpeg ile otomatik getirir.
        """
        p = shutil.which("ffmpeg")
        if p:
            return p
        # fallback: python package that ships an ffmpeg binary
        try:
            import imageio_ffmpeg  # type: ignore
        except Exception:
            return None
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
        if exe and os.path.exists(exe):
            exe_dir = str(Path(exe).parent)
            os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
            return exe
        return None

    ffmpeg_path = _ensure_ffmpeg_on_path()
    if ffmpeg_path is None:
        st.info(
            "FFmpeg bulunamadı. Whisper ses okumak için FFmpeg'e ihtiyaç duyar (WAV dahil).\n"
            "Çözüm: `python -m pip install -U imageio-ffmpeg` (önerilen) veya sisteminize FFmpeg kurun."
        )

    def _ensure_whisper_model(model_name: str):
        # Whisper (openai-whisper) kurulumu yoksa uygulamayı bozmasın
        try:
            import whisper  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Whisper kurulamadı/kurulu değil. Kurulum: `python -m pip install -U openai-whisper`\n"
                "Not: Whisper için FFmpeg gerekebilir (Windows)."
            ) from e

        if st.session_state.whisper_model is None or st.session_state.whisper_loaded_name != model_name:
            # Streamlit bazen sys.stderr'i "geçersiz handle" ile sarabiliyor.
            # Whisper download sırasında tqdm stderr'e yazınca WinError 6 görülebiliyor.
            old_stderr, old_stdout = sys.stderr, sys.stdout
            try:
                if sys.stderr is None or not callable(getattr(sys.stderr, "flush", None)):
                    sys.stderr = sys.__stderr__
                if sys.stdout is None or not callable(getattr(sys.stdout, "flush", None)):
                    sys.stdout = sys.__stdout__
                st.session_state.whisper_model = whisper.load_model(model_name)
                st.session_state.whisper_loaded_name = model_name
            finally:
                sys.stderr, sys.stdout = old_stderr, old_stdout
        return st.session_state.whisper_model

    def transcribe_with_whisper(audio_path: str, model_name: str) -> str:
        model = _ensure_whisper_model(model_name)
        try:
            # Whisper str path verince ffmpeg ile okur. WAV için ffmpeg'e ihtiyaç duymamak adına
            # wav'ı python ile okuyup numpy array olarak veriyoruz.
            audio_in = audio_path
            if str(audio_path).lower().endswith(".wav"):
                with wave.open(audio_path, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    fr = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)

                if fr != 16000:
                    # En sık görülen durum: 48kHz -> 16kHz (tam kat 3). Basit downsample uygulayalım.
                    if fr == 48000:
                        pass
                    else:
                        raise RuntimeError(
                            f"WAV sample rate {fr}Hz. Şimdilik 16kHz destekleniyor (48kHz -> 16kHz otomatik iner).\n"
                            "Çözüm: 16kHz mono WAV'a dönüştürüp tekrar deneyin veya FFmpeg kurun."
                        )
                if sampwidth != 2:
                    raise RuntimeError(
                        f"WAV sample width {sampwidth} bytes. Şimdilik 16-bit PCM WAV bekleniyor."
                    )

                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                elif n_channels != 1:
                    raise RuntimeError(f"Desteklenmeyen kanal sayısı: {n_channels}")

                # 48kHz -> 16kHz downsample (hızlı/kurulumsuz; konuşma için yeterli)
                if fr == 48000:
                    audio = audio[::3]

                # Basit seviye ölçümü + normalize (çok düşük seste boş sonuç riskini azaltır)
                duration_sec = float(len(audio)) / 16000.0
                rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
                peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

                # Çok kısa veya neredeyse sessiz ise erken uyarı
                if duration_sec < 0.5:
                    raise RuntimeError("Ses çok kısa (<0.5 sn). Daha uzun bir kayıt deneyin.")

                if peak > 0 and peak < 0.08:
                    gain = min(12.0, 0.7 / peak)  # aşırı yükseltme olmasın
                    audio = np.clip(audio * gain, -1.0, 1.0)
                    rms = float(np.sqrt(np.mean(np.square(audio))))
                    peak = float(np.max(np.abs(audio)))

                # UI'ye yazdırmak için session'a koy
                st.session_state.asr_audio_stats = {
                    "duration_sec": round(duration_sec, 2),
                    "rms": round(rms, 4),
                    "peak": round(peak, 4),
                    "sr": 16000,
                }

                audio_in = audio

            result = model.transcribe(
                audio_in,
                fp16=False,
                language="tr",
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.95,
                logprob_threshold=-1.0,
            )
        except Exception as e:
            raise RuntimeError(
                "Whisper transkripsiyon hatası. FFmpeg eksik olabilir veya dosya biçimi desteklenmiyor.\n"
                "Çözüm: FFmpeg kurup tekrar deneyin; alternatif olarak WAV yükleyin."
            ) from e
        text = (result.get("text") or "").strip()
        return text

    def start_live_asr_threads():
        """
        Kamera çalışırken arka planda mikrofon kaydı + periyodik Whisper transkripsiyonu.
        Not: Bu thread içinden lesson_text ve preferred_quiz_tags güncellenir.
        """
        if st.session_state.live_asr_running:
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Live mikrofon modu için `sounddevice` kurulmalı. "
                "Çözüm: `python -m pip install -U sounddevice`"
            ) from e

        # stop event + buffer hazırlığı
        st.session_state.live_asr_stop_event = threading.Event()
        st.session_state.live_asr_audio_lock = threading.Lock()
        st.session_state.live_asr_audio_buffer = deque()
        st.session_state.live_asr_audio_samples = 0
        st.session_state.live_asr_level_peak = 0.0
        st.session_state.live_asr_level_rms = 0.0
        st.session_state.live_asr_last_error = None

        sr = 16000
        window_sec = float(st.session_state.live_asr_window_sec)
        max_samples = int(sr * window_sec)
        transcribe_every = float(st.session_state.live_asr_transcribe_every_sec)
        min_audio_sec = 1.5
        min_samples = int(sr * min_audio_sec)

        def audio_callback(indata, frames, time_info, status):
            if st.session_state.live_asr_stop_event is None:
                return
            if status:
                # sessiz kalalım; stream bazen küçük uyarılar basar
                pass
            # indata: float32 [frames, channels]
            mono = indata[:, 0].copy()
            # seviye meteri (EMA ile yumuşat)
            try:
                peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
                rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
                # EMA
                alpha = 0.2
                st.session_state.live_asr_level_peak = (1 - alpha) * float(st.session_state.live_asr_level_peak) + alpha * peak
                st.session_state.live_asr_level_rms = (1 - alpha) * float(st.session_state.live_asr_level_rms) + alpha * rms
            except Exception:
                pass
            with st.session_state.live_asr_audio_lock:
                st.session_state.live_asr_audio_buffer.append(mono)
                st.session_state.live_asr_audio_samples += len(mono)
                # ring buffer trim
                while st.session_state.live_asr_audio_samples > max_samples and st.session_state.live_asr_audio_buffer:
                    old = st.session_state.live_asr_audio_buffer.popleft()
                    st.session_state.live_asr_audio_samples -= len(old)

        def worker():
            # model ilk kez yükleniyorsa gecikebilir; bu yüzden sadece burada çeviriyoruz
            while st.session_state.live_asr_stop_event is not None and not st.session_state.live_asr_stop_event.is_set():
                # belirli aralıklarla transkripte çevir
                time.sleep(transcribe_every)
                try:
                    if st.session_state.live_asr_audio_samples < min_samples:
                        continue

                    with st.session_state.live_asr_audio_lock:
                        audio = np.concatenate(list(st.session_state.live_asr_audio_buffer), axis=0).astype(np.float32)

                    text = transcribe_with_whisper(audio, st.session_state.whisper_model_name)
                    if not text:
                        continue

                    # kısa/boş sonuç filtreleme
                    if len(text) < 3:
                        continue

                    # Canlı transkript göster (UI için)
                    st.session_state.asr_text = text

                    kws = extract_keywords_simple(text, top_k=8)
                    new_tags = keywords_to_tags(kws)

                    # Eğer yeni tag'ler quiz bankasıyla eşleşmiyorsa topic kaymasın diye
                    # lesson_text/nlp_keywords/preferred_quiz_tags'i güncellemiyoruz.
                    overlap_ok = has_any_quiz_tag_overlap(new_tags)
                    if overlap_ok:
                        st.session_state.lesson_text = text
                        st.session_state.nlp_keywords = kws
                        st.session_state.preferred_quiz_tags = list(
                            _normalize_preferred_tags_to_bank(new_tags)
                        )
                        st.session_state.live_asr_tags_updated = True
                    else:
                        st.session_state.live_asr_tags_updated = False

                    # Thread bağlamında Streamlit log_event bazen yazmayabiliyor.
                    # Bu yüzden JSONL'e doğrudan event append ediyoruz.
                    # Not: log path thread başlarken hazır olmayabilir; her seferinde yeniden bak.
                    try:
                        if st.session_state.get("logging_enabled"):
                            if not st.session_state.get("log_jsonl_path"):
                                _ensure_log_paths()
                            jsonl_path = st.session_state.get("log_jsonl_path")
                            sid = st.session_state.get("log_session_id")
                            if jsonl_path and sid:
                                ts_now = time.time()
                                evt = {
                                    "ts": ts_now,
                                    "ts_iso": _iso_now(ts_now),
                                    "session_id": sid,
                                    "type": "live_asr_keywords",
                                    "payload": {
                                        "updated": bool(overlap_ok),
                                        "text_snippet": (text or "")[:120],
                                        "keywords": kws[:12] if kws else [],
                                        "preferred_tags": list(
                                            _normalize_preferred_tags_to_bank(new_tags)
                                        )[:12],
                                        "n_keywords": len(kws) if kws else 0,
                                    },
                                }
                                _append_jsonl(jsonl_path, evt)
                    except Exception:
                        pass

                    st.session_state.live_asr_last_update_ts = time.time()
                    st.session_state.live_asr_last_used_text = text

                except Exception:
                    # canlı modda bir hata olursa döngüyü öldürmeyelim
                    continue

        # start
        st.session_state.live_asr_running = True
        st.session_state.live_asr_worker_thread = threading.Thread(target=worker, daemon=True)
        st.session_state.live_asr_worker_thread.start()

        # main audio stream'i ayrı bir thread'de başlatıp çalıştıracağız
        def stream_thread():
            # ses giriş cihazı seçimi otomatik olsun; gerekirse kullanıcı ayarlayabilir
            dev = st.session_state.get("live_asr_device_index", None)
            try:
                # JSONL'e durum yaz (UI dışında da teşhis edilebilsin)
                try:
                    log_event(
                        "live_asr_started",
                        {
                            "device_index": dev,
                            "sr": sr,
                            "window_sec": window_sec,
                            "transcribe_every_sec": transcribe_every,
                        },
                    )
                except Exception:
                    pass

                with sd.InputStream(
                    device=dev,
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    callback=audio_callback,
                    blocksize=1024,
                ):
                    while st.session_state.live_asr_stop_event is not None and not st.session_state.live_asr_stop_event.is_set():
                        time.sleep(0.25)
            except Exception as e:
                st.session_state.live_asr_last_error = str(e)
                try:
                    log_event(
                        "live_asr_error",
                        {
                            "device_index": dev,
                            "error": str(e),
                        },
                    )
                except Exception:
                    pass

        st.session_state.live_asr_thread = threading.Thread(target=stream_thread, daemon=True)
        st.session_state.live_asr_thread.start()
        st.session_state.live_asr_worker_thread = st.session_state.live_asr_worker_thread
        return

    if uploaded_audio is not None:
        if st.button("🧾 Metne çevir (Whisper)"):
            st.session_state.asr_last_error = None
            try:
                suffix = Path(uploaded_audio.name).suffix or ".wav"
                # FFmpeg yoksa erken uyarı ver (Whisper WAV'de bile ffmpeg çağırır)
                if ffmpeg_path is None:
                    raise RuntimeError(
                        "FFmpeg bulunamadı. Whisper ses okumak için FFmpeg'e ihtiyaç duyar (WAV dahil).\n"
                        "Çözüm 1: `python -m pip install -U imageio-ffmpeg`\n"
                        "Çözüm 2: Sistem FFmpeg kurun (örn. winget)."
                    )
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_audio.getbuffer())
                    tmp_path = tmp.name

                text = transcribe_with_whisper(tmp_path, st.session_state.whisper_model_name)
                st.session_state.asr_text = text
                if text:
                    # Konu kirlenmesin: dosya transkriptinde sadece son metni kullan
                    st.session_state.lesson_text = text

                    kws = extract_keywords_simple(st.session_state.lesson_text, top_k=8)
                    st.session_state.nlp_keywords = kws
                    new_tags = keywords_to_tags(kws)
                    if has_any_quiz_tag_overlap(new_tags):
                        st.session_state.preferred_quiz_tags = list(
                            _normalize_preferred_tags_to_bank(new_tags)
                        )
                    else:
                        st.session_state.preferred_quiz_tags = []
                    log_event(
                        "asr_transcribed",
                        {
                            "model": st.session_state.whisper_model_name,
                            "text_len": len(text),
                            "preferred_tags": st.session_state.preferred_quiz_tags,
                        },
                    )
                    st.success("Transkripsiyon tamamlandı; ders metni son transkript ile güncellendi.")
                else:
                    st.warning("Metin boş döndü (ses düşük olabilir).")
            except Exception as e:
                # Kullanıcıya kısa, anlaşılır mesaj; detay için traceback
                msg = str(e)
                if "WinError 6" in msg or "işleyici geçersiz" in msg.lower():
                    msg = (
                        "Whisper ses dosyasını çözerken hata aldı (WinError 6). "
                        "Bu genelde FFmpeg/codec kaynaklı olur.\n"
                        "Öneri: WAV yükleyin veya FFmpeg kurun."
                    )
                st.session_state.asr_last_error = msg
                st.error(st.session_state.asr_last_error)
                with st.expander("Hata detayı"):
                    st.code(traceback.format_exc())

    if st.session_state.asr_text:
        st.text_area("Son transkript", value=st.session_state.asr_text, height=120)
    if st.session_state.live_asr_enabled:
        if st.session_state.live_asr_last_update_ts is not None:
            age = int(max(0.0, time.time() - float(st.session_state.live_asr_last_update_ts)))
            st.caption(
                f"Live ASR: son güncelleme {age} sn önce | tag'ler güncellendi: {st.session_state.live_asr_tags_updated} "
                f"| peak:{st.session_state.live_asr_level_peak:.4f} rms:{st.session_state.live_asr_level_rms:.4f}"
            )
        else:
            st.caption(
                f"Live ASR: henüz transkript yok | peak:{st.session_state.live_asr_level_peak:.4f} "
                f"rms:{st.session_state.live_asr_level_rms:.4f}"
            )
        if st.session_state.live_asr_last_error:
            st.warning(f"Live ASR hata: {st.session_state.live_asr_last_error}")
        if st.session_state.preferred_quiz_tags:
            st.caption(f"Live ASR tag önceliği: {', '.join(st.session_state.preferred_quiz_tags)}")
    if st.session_state.get("asr_audio_stats"):
        s = st.session_state.asr_audio_stats
        st.caption(f"Ses: {s['duration_sec']} sn | sr:{s['sr']} | rms:{s['rms']} | peak:{s['peak']}")

attention_placeholder = st.empty()
status_placeholder = st.empty()
state_placeholder = st.empty()
debug_placeholder = st.empty()
frame_slot = st.image([])


# ======================
# MEDIAPIPE
# ======================
MODEL_PATH = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

def get_landmarker():
    lm = st.session_state.get("landmarker")
    if lm is None:
        st.session_state.landmarker = vision.FaceLandmarker.create_from_options(options)
    return st.session_state.landmarker

if st.session_state.get("landmarker") is None:
    st.session_state.landmarker = vision.FaceLandmarker.create_from_options(options)


# ======================
# CAMERA
# ======================
if "cap" not in st.session_state or st.session_state.cap is None:
    st.session_state.cap = cv2.VideoCapture(0)

def cleanup_runtime_resources():
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.cap = None

    # stop live asr threads if running
    try:
        stop_event = st.session_state.get("live_asr_stop_event")
        st.session_state.live_asr_running = False
        if stop_event is not None:
            stop_event.set()
        t1 = st.session_state.get("live_asr_thread")
        t2 = st.session_state.get("live_asr_worker_thread")
        for t in [t1, t2]:
            if t is not None and hasattr(t, "join"):
                try:
                    t.join(timeout=1.0)
                except Exception:
                    pass
    except Exception:
        pass
    st.session_state.live_asr_stop_event = None
    st.session_state.live_asr_thread = None
    st.session_state.live_asr_worker_thread = None
    st.session_state.live_asr_audio_lock = None
    st.session_state.live_asr_audio_buffer = None
    try:
        lm = st.session_state.get("landmarker")
        if lm is not None and hasattr(lm, "close"):
            lm.close()
    except Exception:
        pass
    st.session_state.landmarker = None

if not st.session_state.get("atexit_registered", False):
    atexit.register(cleanup_runtime_resources)
    st.session_state.atexit_registered = True

def _iso_now(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def _ensure_log_paths():
    if st.session_state.log_csv_path and st.session_state.log_jsonl_path:
        return
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    sid = st.session_state.log_session_id
    st.session_state.log_csv_path = str(Path(LOG_DIR) / f"mindguard_{sid}_samples.csv")
    st.session_state.log_jsonl_path = str(Path(LOG_DIR) / f"mindguard_{sid}_events.jsonl")

def _append_csv_row(path: str, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def _append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None):
    if not st.session_state.logging_enabled:
        return
    _ensure_log_paths()
    evt = {
        "ts": time.time(),
        "ts_iso": _iso_now(),
        "session_id": st.session_state.log_session_id,
        "type": event_type,
        "payload": payload or {},
    }
    _append_jsonl(st.session_state.log_jsonl_path, evt)

def maybe_log_sample(sample: dict):
    if not st.session_state.logging_enabled:
        return
    now = time.time()
    if (now - st.session_state.last_sample_log_ts) < SAMPLE_LOG_INTERVAL_SEC:
        return
    st.session_state.last_sample_log_ts = now
    _ensure_log_paths()

    row = {
        "ts": now,
        "ts_iso": _iso_now(now),
        "session_id": st.session_state.log_session_id,
        **sample,
    }
    _append_csv_row(st.session_state.log_csv_path, row)
    _append_jsonl(st.session_state.log_jsonl_path, {"type": "sample", **row})


def _quiz_by_id(qid: str):
    for q in QUIZZES:
        if q.get("id") == qid:
            return q
    return None


_BANK_TAGS_FROZEN: Optional[FrozenSet[str]] = None


def _quiz_bank_tag_set() -> FrozenSet[str]:
    """Quiz bankasında gerçekten kullanılan tag kümesi (küçük harf, trim)."""
    global _BANK_TAGS_FROZEN
    if _BANK_TAGS_FROZEN is None:
        s: Set[str] = set()
        for q in QUIZZES:
            for t in q.get("tags") or []:
                if t is None:
                    continue
                u = str(t).lower().strip()
                if u:
                    s.add(u)
        _BANK_TAGS_FROZEN = frozenset(s)
    return _BANK_TAGS_FROZEN


def _normalize_preferred_tags_to_bank(preferred_tags) -> Set[str]:
    """Sadece bankada var olan tag'ler; gürültü kelimeleri elenir."""
    bank = _quiz_bank_tag_set()
    out: Set[str] = set()
    for t in preferred_tags or []:
        if t is None:
            continue
        u = str(t).lower().strip()
        if u in bank:
            out.add(u)
    return out


def pick_next_quiz(*, recent_ids, preferred_tags=None):
    """
    - recent_ids: en son gösterilmiş quiz id listesi (tam liste)
    - preferred_tags: örn ["isim","fiil"]; boş/None ise tüm banka (tekrar önleme korunur)
    - En az bir banka-tag'i istenmiş ama hiçbiri bankada yoksa: None
    - Banka-tag'leri ile hiç quiz örtüşmüyorsa (max_overlap==0): None — rastgele alakasız soru yok
    """
    raw_incoming = list(preferred_tags or [])
    wanted_tag_based = len(raw_incoming) > 0
    preferred_tags_set = _normalize_preferred_tags_to_bank(raw_incoming)

    all_quizzes = [q for q in QUIZZES if q.get("id")]
    if wanted_tag_based:
        if not preferred_tags_set:
            return None
        scored = []
        max_overlap = 0
        for q in all_quizzes:
            qtags = {str(x).lower().strip() for x in (q.get("tags") or []) if x}
            overlap = len(qtags & preferred_tags_set)
            max_overlap = max(max_overlap, overlap)
            scored.append((q, overlap))
        if max_overlap == 0:
            return None
        candidates = [q for (q, ov) in scored if ov == max_overlap]
    else:
        candidates = all_quizzes

    recent_window = set(recent_ids[-QUIZ_RECENT_WINDOW:]) if recent_ids else set()
    non_recent = [q for q in candidates if q.get("id") not in recent_window]
    pool = non_recent if non_recent else candidates
    return random.choice(pool) if pool else None


def has_any_quiz_tag_overlap(preferred_tags) -> bool:
    """keywords_to_tags çıktısı bankadaki gerçek tag'lerle en az bir kez örtüşüyor mu?"""
    return len(_normalize_preferred_tags_to_bank(preferred_tags)) > 0


def _validate_generated_quiz(q: dict) -> Optional[str]:
    """
    Üretilen quiz formatını doğrula.
    Beklenen: question(str), options(list[str] len=4 unique), answer(str in options)
    """
    if not isinstance(q, dict):
        return "not_a_dict"
    question = q.get("question")
    options = q.get("options")
    answer = q.get("answer")
    if not isinstance(question, str) or len(question.strip()) < 5:
        return "bad_question"
    if not isinstance(options, list) or len(options) != 4:
        return "bad_options_len"
    if not all(isinstance(o, str) and o.strip() for o in options):
        return "bad_option_type"
    norm = [o.strip() for o in options]
    if len(set(norm)) != 4:
        return "options_not_unique"
    if not isinstance(answer, str) or answer.strip() not in norm:
        return "answer_not_in_options"
    return None


def _norm_choice(s: str) -> str:
    # cevap/şık eşleştirmede boşluk/tırnak farkları için
    return re.sub(r"\s+", " ", str(s).strip().strip("\"'")).casefold()


def _normalize_generated_quiz_inplace(q: dict) -> None:
    """options/answer whitespace normalizasyonu + answer'ı seçeneklere hizalama."""
    if not isinstance(q, dict):
        return
    if isinstance(q.get("question"), str):
        q["question"] = q["question"].strip()
    if isinstance(q.get("options"), list):
        q["options"] = [str(o).strip().strip("\"'") for o in q["options"]]
    if isinstance(q.get("answer"), str):
        q["answer"] = q["answer"].strip().strip("\"'")
    # answer'ı seçeneklerle normalize eşleştir
    opts = q.get("options") if isinstance(q.get("options"), list) else None
    ans = q.get("answer") if isinstance(q.get("answer"), str) else None
    if opts and ans:
        want = _norm_choice(ans)
        for o in opts:
            if _norm_choice(o) == want:
                q["answer"] = o
                break


def _repair_quiz_json_via_ollama(*, base_url: str, model: str, temperature: float, bad_json_text: str) -> Optional[dict]:
    """Geçersiz JSON'u Ollama ile 1 kez düzeltmeyi dene."""
    repair_prompt = (
        "Aşağıdaki JSON bir çoktan seçmeli soru şeması. Hatalı olabilir.\n"
        "Görev: JSON'u SADECE düzelt ve yalnızca JSON döndür.\n"
        "Kurallar: options tam 4 adet ve benzersiz olsun; answer options içindeki bir değerle birebir aynı olsun.\n\n"
        f"JSON:\n{bad_json_text}\n"
    )
    r = requests.post(
        f"{base_url}/api/generate",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": repair_prompt,
            "stream": False,
            "options": {"temperature": float(temperature)},
        },
        timeout=180,
    )
    if r.status_code >= 400:
        return None
    data = r.json()
    text = (data.get("response") if isinstance(data, dict) else None)
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip().strip("`").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def generate_quiz_llm(*, preferred_tags, keywords) -> Optional[dict]:
    """
    LLM ile 4 şıklı quiz üret.
    - Ollama çalışmıyorsa None
    - geçersiz format ise None
    """
    tags = [str(t) for t in (preferred_tags or []) if t]
    kws = [str(k) for k in (keywords or []) if k]
    if not tags and not kws:
        return None

    sys_prompt = (
        "Sen bir eğitim asistanısın. Türkçe, kısa ve net 4 seçenekli (A/B/C/D) çoktan seçmeli soru üret. "
        "Soru tek bir konuya odaklı olsun ve ortaokul seviyesine yakın, genel bir düzeyde kalsın. "
        "Sadece aşağıdaki JSON şemasını döndür; ekstra metin yazma.\n\n"
        "{\n"
        "  \"question\": \"...\",\n"
        "  \"options\": [\"...\",\"...\",\"...\",\"...\"],\n"
        "  \"answer\": \"...\",\n"
        "  \"difficulty\": 1\n"
        "}\n\n"
        "Kurallar:\n"
        "- options tam 4 adet olmalı ve birbirinden farklı olmalı.\n"
        "- answer options içindeki bir değerle birebir aynı olmalı.\n"
        "- Özel isim/kişisel veri sorma.\n"
    )

    user_prompt = (
        f"Konu etiketleri: {tags}\n"
        f"Anahtar kelimeler: {kws}\n"
        "Bu konuya uygun 1 soru üret."
    )

    try:
        base = str(st.session_state.get("llm_ollama_url") or "http://localhost:11434").rstrip("/")
        def _ollama_generate():
            return requests.post(
                f"{base}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": str(st.session_state.llm_model_name),
                    "prompt": sys_prompt + "\n\n" + user_prompt,
                    "stream": False,
                    "options": {
                        "temperature": float(st.session_state.llm_temperature),
                    },
                },
                # Yavaş CPU / ilk-token gecikmesi için daha yüksek timeout
                timeout=180,
            )

        try:
            r = _ollama_generate()
        except requests.exceptions.ReadTimeout:
            # bir kez daha dene (model ısınma / yükleme gecikmesi)
            time.sleep(1.0)
            r = _ollama_generate()
        if r.status_code >= 400:
            st.session_state.llm_last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            return None
        data = r.json()
        text = (data.get("response") if isinstance(data, dict) else None)
        if not isinstance(text, str) or not text.strip():
            st.session_state.llm_last_error = "Ollama boş çıktı döndürdü"
            return None

        # JSON parse
        try:
            q = json.loads(text)
        except Exception:
            # bazen etrafında markdown olabilir; kaba temizleme
            cleaned = text.strip()
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            q = json.loads(cleaned)

        # normalize + validate + 1 kez self-repair
        _normalize_generated_quiz_inplace(q)
        err = _validate_generated_quiz(q)
        if err:
            repaired = _repair_quiz_json_via_ollama(
                base_url=base,
                model=str(st.session_state.llm_model_name),
                temperature=float(st.session_state.llm_temperature),
                bad_json_text=text,
            )
            if repaired is not None:
                _normalize_generated_quiz_inplace(repaired)
                err2 = _validate_generated_quiz(repaired)
                if not err2:
                    q = repaired
                else:
                    st.session_state.llm_last_error = f"Üretilen quiz geçersiz: {err2}"
                    return None
            else:
                st.session_state.llm_last_error = f"Üretilen quiz geçersiz: {err}"
                return None

        # normalize + attach metadata
        gen_id = f"gen_{uuid.uuid4().hex[:10]}"
        out_quiz = {
            "id": gen_id,
            "question": q["question"].strip(),
            "options": [o.strip() for o in q["options"]],
            "answer": str(q["answer"]).strip(),
            "difficulty": int(q.get("difficulty") or 1),
            "tags": list(_normalize_preferred_tags_to_bank(tags)) or tags,
            "source": "llm",
        }
        st.session_state.llm_last_error = None
        return out_quiz
    except Exception as e:
        st.session_state.llm_last_error = str(e)
        return None


def get_next_quiz(*, recent_ids, preferred_tags, keywords) -> Optional[dict]:
    """
    Kontrollü seçim:
    1) LLM açık + tag/keyword varsa üretmeyi dene
    2) Üretim yoksa/başarısızsa quiz bank'tan seç
    """
    preferred_tags = list(preferred_tags or [])
    if st.session_state.get("llm_quiz_enabled"):
        q = generate_quiz_llm(preferred_tags=preferred_tags, keywords=keywords)
        if q is not None:
            log_event("llm_quiz_generated", {"quiz_id": q.get("id"), "tags": preferred_tags, "keywords": list(keywords or [])})
            return q
        log_event("llm_quiz_failed", {"tags": preferred_tags, "keywords": list(keywords or []), "error": st.session_state.get("llm_last_error")})
    return pick_next_quiz(recent_ids=recent_ids, preferred_tags=preferred_tags)


# ======================
# NLP v0.1 helpers
# ======================
_TR_STOPWORDS = {
    "ve", "veya", "ile", "de", "da", "ki", "bu", "şu", "o", "bir", "birkaç", "çok",
    "için", "ama", "fakat", "ancak", "gibi", "daha", "en", "mi", "mı", "mu", "mü",
    "ne", "neden", "nasıl", "hangi", "kim", "kaç", "olan", "olarak", "üzerine",
    "ile", "ya", "ya da", "ise", "hem", "çünkü", "diye",
}

_TAG_ALIASES = {
    # matematik
    "toplama": ["matematik", "toplama"],
    "çıkarma": ["matematik"],
    "carpma": ["matematik"],
    "çarpma": ["matematik"],
    "bolme": ["matematik", "bolme"],
    "bölme": ["matematik", "bolme"],
    "üçgen": ["matematik", "geometri"],
    "ucgen": ["matematik", "geometri"],
    "üçgenin": ["matematik", "geometri"],
    "ucgenin": ["matematik", "geometri"],
    "geometri": ["matematik", "geometri"],
    "açı": ["matematik", "geometri"],
    "acı": ["matematik", "geometri"],
    "açılar": ["matematik", "geometri"],
    "acilar": ["matematik", "geometri"],
    "açıları": ["matematik", "geometri"],
    "acilari": ["matematik", "geometri"],
    "derece": ["matematik", "geometri"],
    "derecedir": ["matematik", "geometri"],
    "180": ["matematik", "geometri"],
    "alan": ["matematik", "geometri"],
    "çevre": ["matematik", "geometri"],
    "cevre": ["matematik", "geometri"],
    "açıortay": ["matematik", "geometri"],
    "aciortay": ["matematik", "geometri"],
    "kenarortay": ["matematik", "geometri"],
    "yükseklik": ["matematik", "geometri"],
    "yukseklik": ["matematik", "geometri"],
    "dik": ["matematik", "geometri"],
    "formül": ["matematik"],
    "formul": ["matematik"],
    "teorem": ["matematik", "geometri"],
    # türkçe/dilbilgisi
    "isim": ["turkce", "dilbilgisi", "isim"],
    "fiil": ["turkce", "dilbilgisi", "fiil"],
    "sıfat": ["turkce", "dilbilgisi", "sifat"],
    "sifat": ["turkce", "dilbilgisi", "sifat"],
    "zarf": ["turkce", "dilbilgisi", "zarf"],
    "dilbilgisi": ["turkce", "dilbilgisi"],
    # fen
    "fotosentez": ["fen", "biyoloji", "fotosentez"],
    "bitki": ["fen", "biyoloji"],
    "karbondioksit": ["fen", "biyoloji", "fotosentez"],
    "donma": ["fen", "madde", "hal_degisimleri"],
    # tarih
    "cumhuriyet": ["tarih", "cumhuriyet"],
    "1923": ["tarih", "cumhuriyet"],
}

def _stem_tr_token(w: str) -> str:
    # Çok basit suffix kırpma (NLP v0.1). Tam bir kök bulucu değil; pratik eşleştirme için.
    suffixes = [
        "leri", "ları", "lar", "ler",
        "inin", "ının", "unun", "ünün",
        "nin", "nın", "nun", "nün",
        "in", "ın", "un", "ün",
        "si", "sı", "su", "sü",
        "dir", "dır", "dur", "dür",
        "tir", "tır", "tur", "tür",
        "den", "dan", "ten", "tan",
    ]
    for suf in suffixes:
        if w.endswith(suf) and len(w) > len(suf) + 1:
            return w[: -len(suf)]
    return w


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^0-9a-zçğıöşü\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_keywords_simple(text: str, top_k: int = 8):
    """
    Basit keyword çıkarımı (NLP v0.1):
    - token frekansı (stopword filtreli)
    - 2+ karakter
    """
    t = _normalize_text(text)
    if not t:
        return []
    tokens = [w for w in t.split(" ") if len(w) >= 2 and w not in _TR_STOPWORDS]
    if not tokens:
        return []
    freq: Dict[str, int] = {}
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for (w, _c) in ranked[:top_k]]


def keywords_to_tags(keywords):
    tags = []
    for kw in keywords:
        k = kw
        ks = _stem_tr_token(k)

        # 1) doğrudan alias
        if k in _TAG_ALIASES:
            tags.extend(_TAG_ALIASES[k])
            continue
        # 2) stem alias
        if ks in _TAG_ALIASES:
            tags.extend(_TAG_ALIASES[ks])
            continue
        # 3) substring alias (örn: "açıları" -> "açı")
        matched = False
        for alias_key, alias_tags in _TAG_ALIASES.items():
            if alias_key and (alias_key in k or alias_key in ks):
                tags.extend(alias_tags)
                matched = True
                break
        if matched:
            continue
        # 4) keyword doğrudan tag ile eşleşiyorsa
        tags.append(k)
    # unique + stable
    out = []
    for t in tags:
        if t and t not in out:
            out.append(t)
    return out


if st.session_state.lesson_text.strip():
    if st.button("🔎 Anahtar kelimeleri çıkar / güncelle"):
        kws = extract_keywords_simple(st.session_state.lesson_text, top_k=8)
        st.session_state.nlp_keywords = kws
        candidate_tags = keywords_to_tags(kws)
        if has_any_quiz_tag_overlap(candidate_tags):
            st.session_state.preferred_quiz_tags = list(_normalize_preferred_tags_to_bank(candidate_tags))
        else:
            st.session_state.preferred_quiz_tags = []
        log_event(
            "nlp_updated",
            {"keywords": kws, "preferred_tags": list(st.session_state.preferred_quiz_tags)},
        )

    if st.session_state.nlp_keywords:
        st.caption(f"Anahtar kelimeler: {', '.join(st.session_state.nlp_keywords)}")
    if st.session_state.preferred_quiz_tags:
        st.caption(f"Quiz tag önceliği: {', '.join(st.session_state.preferred_quiz_tags)}")
else:
    st.session_state.nlp_keywords = []
    st.session_state.preferred_quiz_tags = []


def update_smoothing_buffer(raw_score: int) -> int:
    """Son SMOOTH_WINDOW_SECONDS içindeki skorların ortalamasını döndürür."""
    now = time.time()
    st.session_state.score_buffer.append((now, raw_score))

    cutoff = now - SMOOTH_WINDOW_SECONDS
    st.session_state.score_buffer = [(t, s) for (t, s) in st.session_state.score_buffer if t >= cutoff]

    if not st.session_state.score_buffer:
        return raw_score

    avg = sum(s for (_, s) in st.session_state.score_buffer) / len(st.session_state.score_buffer)
    return int(avg)


def mindguard_state_machine(smoothed_score: int) -> Optional[str]:
    """IDLE → LOW → TRIGGERED → COOLDOWN döngüsü."""
    now = time.time()

    # COOLDOWN kontrolü
    if st.session_state.mg_state == "COOLDOWN":
        if now >= st.session_state.cooldown_until:
            st.session_state.mg_state = "IDLE"
        else:
            return None

    # Quiz aktifse TRIGGERED
    if st.session_state.quiz_active:
        st.session_state.mg_state = "TRIGGERED"
        return None

    # IDLE / LOW
    if smoothed_score >= LOW_THRESHOLD:
        st.session_state.mg_state = "IDLE"
        st.session_state.low_start_time = None
        return None

    # smoothed_score < LOW_THRESHOLD
    if st.session_state.mg_state != "LOW":
        st.session_state.mg_state = "LOW"
        st.session_state.low_start_time = now
        return None

    # LOW süre kontrolü
    if st.session_state.low_start_time is not None:
        elapsed = now - st.session_state.low_start_time
        if elapsed >= LOW_DURATION_REQUIRED:
            if st.session_state.intervention_enabled:
                st.session_state.quiz_active = True
                st.session_state.mg_state = "TRIGGERED"
                return "TRIGGERED"
            # kontrol modunda spam olmasın: cooldown'a al, ama quiz tetikleme
            st.session_state.mg_state = "COOLDOWN"
            st.session_state.cooldown_until = time.time() + COOLDOWN_DURATION
            st.session_state.low_start_time = None
            return "SUPPRESSED"
    return None


# ======================
# HEAD POSE (solvePnP)
# ======================
def estimate_head_pose_yaw_pitch(face_landmarks, w: int, h: int):
    """MediaPipe Face landmarks'tan yaklaşık yaw/pitch (derece) tahmini."""
    idx_nose = 1
    idx_chin = 152
    idx_left_eye_outer = 33
    idx_right_eye_outer = 263
    idx_left_mouth = 61
    idx_right_mouth = 291

    image_points = np.array([
        [face_landmarks[idx_nose].x * w, face_landmarks[idx_nose].y * h],
        [face_landmarks[idx_chin].x * w, face_landmarks[idx_chin].y * h],
        [face_landmarks[idx_left_eye_outer].x * w, face_landmarks[idx_left_eye_outer].y * h],
        [face_landmarks[idx_right_eye_outer].x * w, face_landmarks[idx_right_eye_outer].y * h],
        [face_landmarks[idx_left_mouth].x * w, face_landmarks[idx_left_mouth].y * h],
        [face_landmarks[idx_right_mouth].x * w, face_landmarks[idx_right_mouth].y * h],
    ], dtype="double")

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rvec, _tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)

    yaw_deg = float(yaw * 180.0 / math.pi)
    pitch_deg = float(pitch * 180.0 / math.pi)
    return yaw_deg, pitch_deg


def headpose_to_score(yaw_deg: float, pitch_deg: float) -> int:
    """0/50/100 baş-pozu skoru."""
    ay = abs(yaw_deg)
    ap = abs(pitch_deg)

    if ay <= YAW_TIGHT and ap <= PITCH_TIGHT:
        return 100
    if ay <= YAW_MED and ap <= PITCH_MED:
        return 50
    return 0


# ======================
# DROWSY / BLINK / YAWN
# ======================
def update_blink_and_drowsy(ear_score: int, mar: float):
    """
    ear_score ve mar değerlerinden:
    - long eye closure flag
    - yawn flag
    - blink rate (bpm)
    üretir ve drowsy_score (0-100) döndürür.
    """
    now = time.time()

    # ----- Eye closed tracking -----
    eye_closed = ear_score <= EAR_CLOSED_SCORE_TH

    # blink: açık->kapalı geçişini yakala
    if eye_closed and not st.session_state.prev_eye_closed:
        st.session_state.blink_times.append(now)

    st.session_state.prev_eye_closed = eye_closed

    # long closure
    long_closure = False
    if eye_closed:
        if st.session_state.eye_closed_start is None:
            st.session_state.eye_closed_start = now
        else:
            if (now - st.session_state.eye_closed_start) >= EYE_CLOSURE_REQUIRED:
                long_closure = True
    else:
        st.session_state.eye_closed_start = None

    # ----- Yawn tracking -----
    yawn = False
    if mar >= MAR_YAWN_TH:
        if st.session_state.yawn_start is None:
            st.session_state.yawn_start = now
        else:
            if (now - st.session_state.yawn_start) >= YAWN_REQUIRED:
                yawn = True
    else:
        st.session_state.yawn_start = None

    # ----- Blink rate (BPM) -----
    cutoff = now - BLINK_WINDOW_SEC
    st.session_state.blink_times = [t for t in st.session_state.blink_times if t >= cutoff]
    blink_count = len(st.session_state.blink_times)
    bpm = (blink_count * 60.0) / BLINK_WINDOW_SEC

    # ----- Drowsy score -----
    # 100 = iyi (uykulu değil), 0 = çok risk
    penalty = 0.0
    if long_closure:
        penalty += 40.0
    if yawn:
        penalty += 40.0
    if bpm > BLINK_BPM_DROWSY_TH:
        penalty += min(20.0, (bpm - BLINK_BPM_DROWSY_TH) * 2.0)

    drowsy_score = int(max(0, min(100, 100 - penalty)))

    # flag'ler
    drowsy_flag = long_closure or yawn or (bpm > BLINK_BPM_DROWSY_TH)
    return drowsy_score, drowsy_flag, long_closure, yawn, bpm


# ======================
# REAL-TIME LOOP
# ======================
if run:
    if st.session_state.logging_enabled:
        _ensure_log_paths()
        st.caption(f"CSV: `{st.session_state.log_csv_path}`")
        st.caption(f"JSONL: `{st.session_state.log_jsonl_path}`")

    # Live Whisper ASR (Faz 5): kamera çalışırken mikrofon dinle
    if st.session_state.live_asr_enabled and not st.session_state.live_asr_running:
        try:
            start_live_asr_threads()
            st.caption("Live ASR: mikrofon dinleniyor (periyodik transkripsiyon).")
        except Exception as e:
            st.warning(f"Live ASR başlatılamadı: {e}")

    try:
        while run:
            ok, frame = st.session_state.cap.read()
            if not ok:
                st.error("Kamera okunamadı")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame
            )

            result = get_landmarker().detect(mp_image)

            ear_raw_score = None
            headpose_score = None
            drowsy_score = None
            final_raw_score = None
            final_smoothed = None
            yaw_deg = None
            pitch_deg = None
            mar = None
            drowsy_flag = False
            long_closure = False
            yawn = False
            bpm = 0.0

            if result.face_landmarks:
                h, w, _ = frame.shape
                face = result.face_landmarks[0]

                # ---- EAR ----
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]

                left_pts = [(int(face[i].x*w), int(face[i].y*h)) for i in LEFT_EYE]
                right_pts = [(int(face[i].x*w), int(face[i].y*h)) for i in RIGHT_EYE]

                avg_ear = (compute_ear(left_pts) + compute_ear(right_pts)) / 2
                ear_raw_score = ear_to_attention_score(avg_ear)

                # ---- HEAD POSE ----
                yaw_deg, pitch_deg = estimate_head_pose_yaw_pitch(face, w, h)
                if yaw_deg is None or pitch_deg is None:
                    headpose_score = 50  # fallback: nötr
                else:
                    headpose_score = headpose_to_score(yaw_deg, pitch_deg)

                # ---- MAR (yawn) ----
                # mouth corners: 61, 291 ; upper/lower inner lip: 13, 14
                mouth_pts = [
                    (int(face[61].x*w), int(face[61].y*h)),
                    (int(face[291].x*w), int(face[291].y*h)),
                    (int(face[13].x*w), int(face[13].y*h)),
                    (int(face[14].x*w), int(face[14].y*h)),
                ]
                mar = compute_mar(mouth_pts)

                # ---- DROWSY score ----
                drowsy_score, drowsy_flag, long_closure, yawn, bpm = update_blink_and_drowsy(
                    ear_score=ear_raw_score,
                    mar=mar
                )

                # ---- HYBRID FINAL (v0.1) ----
                final_raw_score = int(
                    W_EAR * ear_raw_score +
                    W_HEADPOSE * headpose_score +
                    W_DROWSY * drowsy_score
                )

                # smoothing (FINAL üzerinde)
                final_smoothed = update_smoothing_buffer(final_raw_score)

                # state machine (FINAL üzerinden tetik)
                action = mindguard_state_machine(final_smoothed)
                if action in ("TRIGGERED", "SUPPRESSED"):
                    # trigger latency metriği
                    if st.session_state.low_start_time is not None:
                        st.session_state.last_trigger_latency_sec = time.time() - st.session_state.low_start_time
                    evt_type = "quiz_triggered" if action == "TRIGGERED" else "intervention_suppressed"
                    log_event(
                        evt_type,
                        {
                            "final_smoothed": final_smoothed,
                            "final_raw": final_raw_score,
                            "ear": ear_raw_score,
                            "headpose": headpose_score,
                            "drowsy": drowsy_score,
                            "trigger_latency_sec": st.session_state.last_trigger_latency_sec,
                        },
                    )

                # UI
                attention_placeholder.progress(final_smoothed)

                # durum metni
                if st.session_state.mg_state == "COOLDOWN":
                    remaining = int(max(0, st.session_state.cooldown_until - time.time()))
                    status_placeholder.info(f"Cooldown: {remaining} sn (quiz kilitli)")
                else:
                    if final_smoothed > 60:
                        status_placeholder.success(f"Dikkat iyi ({final_smoothed})")
                    elif final_smoothed > LOW_THRESHOLD:
                        status_placeholder.warning(f"Dikkat düşüyor ({final_smoothed})")
                    else:
                        status_placeholder.error(f"Dikkat düşük ({final_smoothed})")

                state_placeholder.caption(
                    f"State: {st.session_state.mg_state} | "
                    f"EAR:{ear_raw_score} Head:{headpose_score} Drowsy:{drowsy_score} "
                    f"| FINAL(raw):{final_raw_score} FINAL(smooth):{final_smoothed}"
                )

                debug_placeholder.caption(
                    f"MAR:{mar:.2f} | long_closure:{long_closure} | yawn:{yawn} | "
                    f"blink_bpm:{bpm:.1f} | drowsy_flag:{drowsy_flag}"
                )

                # ekrana yazı
                yaw_txt = "NA" if yaw_deg is None else f"{yaw_deg:.1f}"
                pitch_txt = "NA" if pitch_deg is None else f"{pitch_deg:.1f}"

                cv2.putText(
                    frame,
                    f"EAR:{ear_raw_score} Head:{headpose_score} D:{drowsy_score} F:{final_smoothed}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Yaw:{yaw_txt} Pitch:{pitch_txt} MAR:{mar:.2f} BPM:{bpm:.1f}",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                if drowsy_flag:
                    cv2.putText(
                        frame,
                        "DROWSY FLAG",
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )

                # per-second sample log
                maybe_log_sample(
                    {
                        "face_detected": 1,
                        "mg_state": st.session_state.mg_state,
                        "ear_score": ear_raw_score,
                        "headpose_score": headpose_score,
                        "drowsy_score": drowsy_score,
                        "final_raw_score": final_raw_score,
                        "final_smoothed": final_smoothed,
                        "yaw_deg": yaw_deg,
                        "pitch_deg": pitch_deg,
                        "mar": mar,
                        "blink_bpm": bpm,
                        "drowsy_flag": int(bool(drowsy_flag)),
                    }
                )

                # quiz sonrası metrikler (online)
                now = time.time()
                if st.session_state.post_quiz_start_ts is not None and not st.session_state.post_quiz_avg_done:
                    if (now - st.session_state.post_quiz_start_ts) <= POST_QUIZ_AVG_WINDOW_SEC:
                        st.session_state.post_quiz_scores.append((now, final_smoothed))
                    else:
                        scores = [s for (_t, s) in st.session_state.post_quiz_scores]
                        avg30 = (sum(scores) / len(scores)) if scores else None
                        st.session_state.post_quiz_avg_done = True
                        log_event("post_quiz_30s_avg", {"avg_final_smoothed": avg30, "n": len(scores)})

                if st.session_state.recovery_start_ts is not None and not st.session_state.recovered_done:
                    if final_smoothed >= RECOVERY_THRESHOLD:
                        rec = now - st.session_state.recovery_start_ts
                        st.session_state.recovered_done = True
                        log_event(
                            "attention_recovered",
                            {"recovery_sec": rec, "threshold": RECOVERY_THRESHOLD, "final_smoothed": final_smoothed},
                        )

            else:
                # yüz yoksa: yanlış tetik olmasın
                st.session_state.mg_state = "IDLE"
                st.session_state.low_start_time = None

                # drowsy sayaçlarını da sıfırlayalım (yanlış blink/yawn birikmesin)
                st.session_state.eye_closed_start = None
                st.session_state.yawn_start = None
                st.session_state.prev_eye_closed = False
                st.session_state.blink_times = []

                attention_placeholder.progress(0)
                status_placeholder.warning("Yüz algılanamadı")
                state_placeholder.caption(f"State: {st.session_state.mg_state}")
                debug_placeholder.caption("")

                maybe_log_sample(
                    {
                        "face_detected": 0,
                        "mg_state": st.session_state.mg_state,
                        "ear_score": None,
                        "headpose_score": None,
                        "drowsy_score": None,
                        "final_raw_score": None,
                        "final_smoothed": 0,
                        "yaw_deg": None,
                        "pitch_deg": None,
                        "mar": None,
                        "blink_bpm": None,
                        "drowsy_flag": 0,
                    }
                )

            frame_slot.image(frame)

            # ======================
            # QUIZ UI (MODÜLER)
            # ======================
            if st.session_state.quiz_active:
                # quiz aktifken bir kere seç ve "kilitle"
                if st.session_state.active_quiz_id is None:
                    preferred_tags_for_pick = st.session_state.preferred_quiz_tags
                    # Live ASR tetik anında gecikmeli güncellenebildiği için,
                    # quiz seçmeden hemen önce asr_text'ten tag'i tazeleyelim.
                    if st.session_state.live_asr_enabled and st.session_state.asr_text:
                        kws = extract_keywords_simple(st.session_state.asr_text, top_k=10)
                        candidate_tags = keywords_to_tags(kws)
                        if has_any_quiz_tag_overlap(candidate_tags):
                            preferred_tags_for_pick = list(
                                _normalize_preferred_tags_to_bank(candidate_tags)
                            )

                    q = get_next_quiz(
                        recent_ids=st.session_state.quiz_history_ids,
                        preferred_tags=preferred_tags_for_pick,
                        keywords=st.session_state.get("nlp_keywords") or [],
                    )
                    if q is None:
                        st.warning("Anlatılan konuya uygun quiz bulunamadı.")
                        log_event(
                            "quiz_unavailable_no_tag_match",
                            {"preferred_tags": list(preferred_tags_for_pick or [])},
                        )
                        st.session_state.quiz_active = False
                        st.session_state.active_quiz_id = None
                        st.session_state.quiz_shown_ts = None
                        st.session_state.low_start_time = None
                        st.session_state.mg_state = "IDLE"
                        time.sleep(0.15)
                        continue
                    # LLM üretimi ise o anlık quiz'i session'a koy
                    if q.get("source") == "llm":
                        st.session_state.active_generated_quiz = q
                    else:
                        st.session_state.active_generated_quiz = None
                    st.session_state.active_quiz_id = q["id"]
                    st.session_state.quiz_session_id += 1
                    st.session_state.quiz_shown_ts = time.time()
                    log_event("quiz_shown", {"quiz_id": q.get("id")})

                quiz = None
                if st.session_state.get("active_generated_quiz") and st.session_state.active_quiz_id:
                    if st.session_state.active_generated_quiz.get("id") == st.session_state.active_quiz_id:
                        quiz = st.session_state.active_generated_quiz
                if quiz is None:
                    quiz = _quiz_by_id(st.session_state.active_quiz_id) if st.session_state.active_quiz_id else None
                if quiz is None:
                    st.warning("Quiz bulunamadı. Quiz kapatılıyor.")
                    st.session_state.quiz_active = False
                    st.session_state.active_quiz_id = None
                    st.session_state.quiz_shown_ts = None
                    st.session_state.low_start_time = None
                    st.session_state.mg_state = "IDLE"
                    time.sleep(0.15)
                    continue

                st.markdown("---")
                st.subheader("🧠 Mini Quiz")
                st.markdown(f"**{quiz['question']}**")

                selected = st.radio(
                    "Seçiniz:",
                    quiz["options"],
                    key=f"quiz_radio_{st.session_state.quiz_session_id}",
                    index=None
                )

                if st.button("Cevabı Gönder"):
                    if selected is None:
                        st.warning("Lütfen bir seçenek seçin.")
                    elif selected == quiz["answer"]:
                        st.success("✅ Doğru! Tebrikler.")
                        is_correct = True
                    else:
                        st.error(f"❌ Yanlış. Doğru cevap: {quiz['answer']}")
                        is_correct = False

                    if selected is not None:
                        now = time.time()
                        duration_sec = None
                        if st.session_state.quiz_shown_ts is not None:
                            duration_sec = now - st.session_state.quiz_shown_ts

                        st.session_state.last_quiz_result = {
                            "quiz_id": quiz.get("id"),
                            "selected": selected,
                            "answer": quiz.get("answer"),
                            "is_correct": is_correct,
                            "ts": now,
                            "duration_sec": duration_sec,
                        }

                        log_event(
                            "quiz_submitted",
                            {
                                "quiz_id": quiz.get("id"),
                                "selected": selected,
                                "answer": quiz.get("answer"),
                                "is_correct": is_correct,
                                "duration_sec": duration_sec,
                            },
                        )

                        # history güncelle (tekrar önleme için)
                        if quiz.get("id"):
                            st.session_state.quiz_history_ids.append(quiz["id"])

                        # quiz kapat
                        st.session_state.quiz_active = False
                        st.session_state.active_quiz_id = None
                        st.session_state.quiz_shown_ts = None

                        # cooldown başlat
                        st.session_state.mg_state = "COOLDOWN"
                        st.session_state.cooldown_until = time.time() + COOLDOWN_DURATION
                        log_event("cooldown_started", {"cooldown_sec": COOLDOWN_DURATION})

                        # LOW sayaç sıfırla
                        st.session_state.low_start_time = None

                        # post-quiz metrik pencereleri başlat
                        st.session_state.post_quiz_start_ts = now
                        st.session_state.post_quiz_scores = []
                        st.session_state.post_quiz_avg_done = False
                        st.session_state.recovery_start_ts = now
                        st.session_state.recovered_done = False

                # Kamera loop'u içinde widget'lar her frame'de yeniden yaratılınca
                # Streamlit aynı key ile duplicate element hatası verir.
                # Quiz aktifken bu run'ı burada durdurup, kullanıcı etkileşiminde rerun olmasını sağlarız.
                st.stop()

            time.sleep(0.15)
    except KeyboardInterrupt:
        log_event("process_interrupted", {"reason": "KeyboardInterrupt"})
        raise
    finally:
        cleanup_runtime_resources()

else:
    status_placeholder.info("Kamerayı başlatmak için işaretleyin.")
    state_placeholder.caption("State: IDLE")
    debug_placeholder.caption("")
    # Kamera kilitli kalmasın
    if st.session_state.get("cap") is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

if st.session_state.logging_enabled and st.session_state.log_csv_path and os.path.exists(st.session_state.log_csv_path):
    with st.expander("⬇️ Dışa Aktarma"):
        try:
            csv_bytes = Path(st.session_state.log_csv_path).read_bytes()
            jsonl_bytes = Path(st.session_state.log_jsonl_path).read_bytes() if st.session_state.log_jsonl_path else b""
            st.download_button("CSV indir", data=csv_bytes, file_name=Path(st.session_state.log_csv_path).name)
            if jsonl_bytes:
                st.download_button("JSONL indir", data=jsonl_bytes, file_name=Path(st.session_state.log_jsonl_path).name)
        except Exception as e:
            st.warning(f"Log dosyaları okunamadı: {e}")