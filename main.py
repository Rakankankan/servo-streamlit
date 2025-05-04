import streamlit as st
import requests
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from streamlit_autorefresh import st_autorefresh
import cv2
import torch
import time
import datetime
import pytz
from PIL import Image
import numpy as np
import io
import os
from telegram.ext import Application
from telegram import Bot
import asyncio
import logging
import traceback
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Konfigurasi logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Atur cache PyTorch
os.environ['TORCH_HOME'] = '/tmp/torch_hub'

# Zona waktu WIB (UTC+7)
WIB = pytz.timezone('Asia/Jakarta')

# --- KONFIGURASI ---
UBIDOTS_TOKEN = "BBUS-4dkNId6LDOVysK48pdwW8cUGBfAQTK"
DEVICE_LABEL = "rakangaming"
VARIABLES = ["mq2", "humidity", "temperature", "lux", "mic"]

TELEGRAM_BOT_TOKEN = "7941979379:AAEWGtlb87RYkvht8GzL8Ber29uosKo3e4s"
TELEGRAM_CHAT_ID = "5721363432"
NOTIFICATION_INTERVAL = 300  # 5 menit
ALERT_COOLDOWN = 60  # 1 menit

CAMERA_URL = "http://192.168.1.12:81/stream"
FIREBASE_URL = "https://servo-control-f3c90-default-rtdb.asia-southeast1.firebasedatabase.app/servo.json"

GEMINI_API_KEY = "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"
GEMINI_MODEL = "google/gemini-2.0-flash-001"

# --- STYLE CSS ---
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1e2a44; /* Dark navy background */
            color: #e0e6f0;
        }
        .main-container {
            max-width: 1200px;
            margin: auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #3b82f6, #10b981);
            color: white;
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            font-size: 38px;
            font-weight: bold;
            margin-bottom: 30px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            animation: slideIn 0.5s ease-in;
        }
        .narasi {
            background-color: #2a3b5e;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 16px;
            color: #a3bffa;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .data-card {
            background-color: #334876;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .data-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }
        .data-label {
            font-size: 20px;
            font-weight: 600;
            color: #a3bffa;
        }
        .data-value {
            font-size: 28px;
            font-weight: bold;
            color: #60a5fa;
        }
        .status-badge {
            padding: 10px 16px;
            border-radius: 20px;
            font-size: 15px;
            font-weight: 500;
            display: inline-block;
            margin-top: 10px;
        }
        .status-danger { background-color: #f87171; color: #fef2f2; }
        .status-warning { background-color: #f59e0b; color: #fef2f2; }
        .status-success { background-color: #10b981; color: #f0fdf4; }
        .status-info { background-color: #3b82f6; color: #eff6ff; }
        .chat-container {
            max-height: 450px;
            overflow-y: auto;
            background-color: #2a3b5e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
        }
        .chat-message {
            padding: 12px 18px;
            border-radius: 10px;
            margin-bottom: 12px;
            max-width: 85%;
            animation: fadeInChat 0.3s ease-in;
        }
        .user-message {
            background-color: #60a5fa;
            color: white;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #e0e6f0;
            color: #1e2a44;
            margin-right: auto;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 500;
            transition: background-color 0.3s, transform 0.2s;
        }
        .stButton>button:hover {
            background-color: #2563eb;
            transform: scale(1.05);
        }
        .stSlider .st-bx {
            background-color: #3b82f6;
        }
        .stCheckbox label {
            color: #a3bffa;
        }
        .footer {
            text-align: center;
            color: #a3bffa;
            margin-top: 30px;
            font-size: 14px;
            opacity: 0.8;
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        @keyframes fadeInChat {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .fade-in {
            animation: slideIn 0.5s ease-in;
        }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI TELEGRAM ---
async def send_telegram_message(message):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown")
        logger.info("Pesan Telegram dikirim")
        return True
    except Exception as e:
        logger.error(f"Gagal mengirim pesan: {str(e)}")
        st.error(f"Gagal mengirim pesan: {str(e)}")
        return False

async def send_telegram_photo(photo, caption):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode="Markdown")
        logger.info("Foto Telegram dikirim")
        return True
    except Exception as e:
        logger.error(f"Gagal mengirim foto: {str(e)}")
        st.error(f"Gagal mengirim foto: {str(e)}")
        return False

# --- FUNGSI UBIDOTS DENGAN RETRY ---
def get_ubidots_data(variable_label, retries=3, backoff_factor=0.5, timeout=15):
    url = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}/{variable_label}/values"
    headers = {"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"}
    
    # Konfigurasi retry
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    
    try:
        response = session.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            st.error(f"Gagal mengambil data {variable_label}: Status {response.status_code}")
            logger.error(f"Gagal mengambil data {variable_label}: Status {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error mengambil data {variable_label}: {str(e)}")
        logger.error(f"Error mengambil data {variable_label}: {str(e)}")
        return None

# --- PENGAMBILAN DATA SENSOR ---
def fetch_latest_sensor_data():
    data_values = {}
    statuses = {}
    for var_name in VARIABLES:
        data = get_ubidots_data(var_name)
        if data and len(data) > 0:
            value = round(data[0]['value'], 2)
            data_values[var_name] = value
        else:
            data_values[var_name] = None
            statuses[var_name] = f"Data {var_name} tidak tersedia"
    
    statuses["mq2"] = predict_smoke_status(data_values.get("mq2"))
    statuses["lux"] = evaluate_lux_condition(data_values.get("lux"), data_values.get("mq2"))
    statuses["temperature"] = evaluate_temperature_condition(data_values.get("temperature"))
    statuses["humidity"] = (
        f"Kelembapan {data_values.get('humidity', 'N/A')}%: {'tinggi' if data_values.get('humidity') and data_values.get('humidity') > 70 else 'normal' if data_values.get('humidity') and data_values.get('humidity') >= 30 else 'rendah'}"
        if data_values.get("humidity") is not None else "Data kelembapan tidak tersedia"
    )
    statuses["mic"] = evaluate_mic_condition(data_values.get("mic"))
    return {"values": data_values, "statuses": statuses}

# --- SIMULASI DAN MODEL ---
@st.cache_data
def generate_mq2_simulation_data(n_samples=100):
    data = []
    for _ in range(n_samples):
        label = random.choices([0, 1], weights=[0.7, 0.3])[0]
        value = random.randint(400, 1000) if label == 1 else random.randint(100, 400)
        data.append((value, label))
    return pd.DataFrame(data, columns=["mq2_value", "label"])

@st.cache_resource
def train_mq2_model():
    df = generate_mq2_simulation_data()
    X = df[['mq2_value']]
    y = df['label']
    model = LogisticRegression()
    model.fit(X, y)
    return model

model_iot = train_mq2_model()

# --- LOGIKA AI ---
def predict_smoke_status(mq2_value):
    if mq2_value is None:
        return "Data asap tidak tersedia"
    if mq2_value > 800:
        return "Bahaya! Terdeteksi asap rokok!"
    elif mq2_value >= 500:
        return "Mencurigakan: kemungkinan ada asap"
    return "Aman, tidak terdeteksi asap"

def evaluate_lux_condition(lux_value, mq2_value):
    if lux_value is None:
        return "Data cahaya tidak tersedia"
    if lux_value <= 50:
        if "Bahaya" in predict_smoke_status(mq2_value):
            return "Mencurigakan: gelap dan ada asap!"
        return "Ruangan gelap, tapi aman"
    return "Ruangan terang"

def evaluate_temperature_condition(temp_value):
    if temp_value is None:
        return "Data suhu tidak tersedia"
    if temp_value >= 31:
        return "Suhu sangat panas!"
    elif temp_value >= 29:
        return "Suhu cukup panas"
    return "Suhu normal"

def evaluate_mic_condition(mic_value):
    if mic_value is None:
        return "Data amplitudo tidak tersedia"
    if mic_value >= 600:
        return f"Amplitudo {mic_value}: Tinggi"
    elif mic_value >= 200:
        return f"Amplitudo {mic_value}: Sedang"
    return f"Amplitudo {mic_value}: Rendah"

def generate_narrative_report(mq2_status, mq2_value, lux_status, lux_value, temp_status, temp_value, humidity_status, humidity_value, mic_status, mic_value):
    return (
        f"📊 *Laporan Status Ruangan*\n"
        f"- 🚨 Asap: {mq2_status} ({mq2_value if mq2_value is not None else 'N/A'})\n"
        f"- 💡 Cahaya: {lux_status} ({lux_value if lux_value is not None else 'N/A'} lux)\n"
        f"- 🌡️ Suhu: {temp_status} ({temp_value if temp_value is not None else 'N/A'}°C)\n"
        f"- 💧 Kelembapan: {humidity_status}\n"
        f"- 🎙️ Amplitudo: {mic_status}\n"
        f"🕒 Waktu (WIB): {datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')}"
    )

def predict_smoking_risk_rule_based(mq2_value, lux_value, mic_value):
    if None in [mq2_value, lux_value, mic_value]:
        return "Data tidak cukup"
    
    risk_score = 0
    risk_messages = []
    
    if mq2_value > 800:
        risk_score += 50
        risk_messages.append("Asap tinggi: kemungkinan merokok")
    elif mq2_value >= 500:
        risk_score += 30
        risk_messages.append("Asap mencurigakan")
    
    if lux_value <= 50:
        risk_score += 20
        risk_messages.append("Ruangan gelap")
    
    if mic_value >= 600:
        risk_score += 15
        risk_messages.append("Amplitudo tinggi")
    
    risk_level = "Tinggi" if risk_score >= 60 else "Sedang" if risk_score >= 40 else "Rendah"
    recommendation = "Periksa ruangan" if risk_score >= 40 else "Kondisi aman"
    
    return (
        f"🔍 *Prediksi Risiko Merokok*\n"
        f"Tingkat: **{risk_level}** (Skor: {risk_score})\n"
        f"Rincian:\n- {risk_messages[0] if risk_messages else 'Normal'}\n"
        f"Rekomendasi: {recommendation}"
    )

def get_room_condition_summary(mq2_value, lux_value, temperature_value, humidity_value, mic_value):
    mq2_status = predict_smoke_status(mq2_value)
    lux_status = evaluate_lux_condition(lux_value, mq2_value)
    temp_status = evaluate_temperature_condition(temperature_value)
    humidity_status = (
        f"Kelembapan {humidity_value if humidity_value is not None else 'N/A'}%: {'tinggi' if humidity_value and humidity_value > 70 else 'normal' if humidity_value and humidity_value >= 30 else 'rendah'}"
        if humidity_value is not None else "Data tidak tersedia"
    )
    mic_status = evaluate_mic_condition(mic_value)

    if "Bahaya" in mq2_status or "panas" in temp_status:
        return {"status": "Bahaya", "color": "red", "suggestion": "Periksa segera!", "details": {"Asap": mq2_status, "Cahaya": lux_status, "Suhu": temp_status, "Kelembapan": humidity_status, "Amplitudo": mic_status}}
    elif "Mencurigakan" in mq2_status:
        return {"status": "Waspada", "color": "orange", "suggestion": "Pantau terus", "details": {"Asap": mq2_status, "Cahaya": lux_status, "Suhu": temp_status, "Kelembapan": humidity_status, "Amplitudo": mic_status}}
    return {"status": "Aman", "color": "green", "suggestion": "Kondisi baik", "details": {"Asap": mq2_status, "Cahaya": lux_status, "Suhu": temp_status, "Kelembapan": humidity_status, "Amplitudo": mic_status}}

# --- NARASI SINGKAT ---
def generate_narasi_singkat(sensor_data):
    values = sensor_data["values"]
    statuses = sensor_data["statuses"]
    narasi = (
        "Sistem memantau kondisi ruangan secara real-time menggunakan sensor asap (MQ2), cahaya (lux), suhu, kelembapan, dan amplitudo suara. "
        f"Saat ini, status asap: **{statuses['mq2']}** (nilai: {values.get('mq2', 'N/A')}). "
        f"Ruangan {'gelap' if 'gelap' in statuses['lux'].lower() else 'terang'} dengan cahaya {values.get('lux', 'N/A')} lux. "
        f"Suhu {values.get('temperature', 'N/A')}°C ({'panas' if 'panas' in statuses['temperature'].lower() else 'normal'}). "
        f"Kelembapan {values.get('humidity', 'N/A')}% ({'tinggi' if values.get('humidity', 0) > 70 else 'normal' if values.get('humidity', 0) >= 30 else 'rendah'}). "
        f"Amplitudo suara {values.get('mic', 'N/A')} ({'tinggi' if values.get('mic', 0) >= 600 else 'sedang' if values.get('mic', 0) >= 200 else 'rendah'}). "
        "Pantau data di bawah untuk detail lebih lanjut."
    )
    return narasi

# --- NARASI UNTUK TELEGRAM ---
def generate_telegram_narasi(sensor_data):
    values = sensor_data["values"]
    statuses = sensor_data["statuses"]
    narasi = (
        f"📊 *Laporan Kondisi Ruangan* ({datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')})\n\n"
        f"💨 *Asap (MQ2)*: **{statuses['mq2']}** (Nilai: {values.get('mq2', 'N/A')})\n"
        f"Status asap menunjukkan {'bahaya' if 'Bahaya' in statuses['mq2'] else 'mencurigakan' if 'Mencurigakan' in statuses['mq2'] else 'aman'}.\n\n"
        f"💡 *Cahaya*: **{statuses['lux']}** (Nilai: {values.get('lux', 'N/A')} lux)\n"
        f"Ruangan {'gelap, kemungkinan lampu dimatikan' if 'gelap' in statuses['lux'].lower() else 'terang, pencahayaan cukup'}.\n\n"
        f"🌡️ *Suhu*: **{statuses['temperature']}** (Nilai: {values.get('temperature', 'N/A')}°C)\n"
        f"Suhu {'normal dan nyaman' if 'normal' in statuses['temperature'].lower() else 'panas, perhatikan ventilasi'}.\n\n"
        f"💧 *Kelembapan*: **{statuses['humidity']}**\n"
        f"Kelembapan {'tinggi, kemungkinan tidak ada asap rokok' if values.get('humidity', 0) > 70 else 'normal' if values.get('humidity', 0) >= 30 else 'rendah, kemungkinan ada asap rokok'}.\n\n"
        f"🎙️ *Amplitudo*: **{statuses['mic']}**\n"
        f"Suara {'sedang, mungkin ada orang' if values.get('mic', 0) >= 200 and values.get('mic', 0) < 600 else 'tinggi, aktivitas bising' if values.get('mic', 0) >= 600 else 'rendah, ruangan sepi'}."
    )
    return narasi

# --- GEMINI AI CHATBOT ---
def get_gemini_response(messages):
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": GEMINI_MODEL, "messages": messages, "stream": False}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=10)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Error {response.status_code}"
    except requests.exceptions.RequestException:
        return "Error menghubungi AI"

def generate_chatbot_context(mq2_value, lux_value, temperature_value, humidity_value, mic_value):
    return (
        f"Data sensor:\n"
        f"- Asap (MQ2): {mq2_value if mq2_value is not None else 'N/A'}\n"
        f"- Cahaya: {lux_value if lux_value is not None else 'N/A'} lux\n"
        f"- Suhu: {temperature_value if temperature_value is not None else 'N/A'}°C\n"
        f"- Kelembapan: {humidity_value if humidity_value is not None else 'N/A'}%\n"
        f"- Amplitudo: {mic_value if mic_value is not None else 'N/A'}\n"
        f"Waktu (WIB): {datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Jawab sebagai asisten deteksi merokok."
    )

# --- DETEKSI KAMERA ---
@st.cache_resource
def load_yolo_model():
    try:
        device = torch.device('cpu')
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device, force_reload=True)
        return model
    except Exception as e:
        st.error(f"Gagal memuat YOLOv5: {str(e)}")
        return None

async def run_camera_detection(frame_placeholder, status_placeholder):
    try:
        cap = cv2.VideoCapture(CAMERA_URL)
        if not cap.isOpened():
            status_placeholder.error("Tidak dapat membuka kamera")
            return
        last_smoking_notification = 0
        while st.session_state.get("cam_running", False):
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Gagal membaca frame")
                break
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if st.session_state.get("model_cam"):
                results = st.session_state.model_cam(img_pil)
                results.render()
                frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
                df = results.pandas().xyxy[0]
                found_person = 'person' in df['name'].values
                found_smoke = 'smoke' in df['name'].values
            else:
                found_person = found_smoke = False
            _, buffer = cv2.imencode('.jpg', frame)
            st.session_state.latest_frame = buffer.tobytes()
            if found_person and found_smoke and time.time() - last_smoking_notification > ALERT_COOLDOWN:
                caption = f"🚨 *Peringatan*: Merokok terdeteksi!\n🕒 Waktu: {datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')}"
                await send_telegram_photo(st.session_state.latest_frame, caption)
                last_smoking_notification = time.time()
                status_placeholder.warning("Merokok terdeteksi!")
            else:
                status_placeholder.success("Tidak ada aktivitas merokok")
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            await asyncio.sleep(0.1)
        cap.release()
    except Exception as e:
        status_placeholder.error(f"Error kamera: {str(e)}")

# --- NOTIFIKASI PERIODIK ---
async def send_periodic_notification():
    current_time = time.time()
    if 'last_notification_time' not in st.session_state or current_time - st.session_state.last_notification_time >= NOTIFICATION_INTERVAL:
        logger.info("Mengirim notifikasi periodik...")
        sensor_data = fetch_latest_sensor_data()
        caption = generate_telegram_narasi(sensor_data)
        if st.session_state.latest_frame:
            await send_telegram_photo(st.session_state.latest_frame, caption)
        else:
            await send_telegram_message(caption + "\n⚠️ Kamera tidak aktif")
        st.session_state.last_notification_time = current_time
        logger.info("Notifikasi periodik dikirim")

# --- ASYNC WRAPPER ---
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coro)
    loop.close()
    return result

# --- UI UTAMA ---
def main():
    st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="header">Sistem Deteksi Merokok IoT</div>', unsafe_allow_html=True)

    # Inisialisasi session state
    if 'last_notification' not in st.session_state:
        st.session_state.last_notification = {
            'mq2': {'status': None, 'value': None, 'last_alert_sent': 0},
            'lux': {'status': None, 'value': None},
            'temperature': {'status': None, 'value': None},
            'humidity': {'status': None, 'value': None},
            'mic': {'status': None, 'value': None},
            'last_sent': 0
        }
    if 'last_notification_time' not in st.session_state:
        st.session_state.last_notification_time = 0
    if 'latest_frame' not in st.session_state:
        st.session_state.latest_frame = None
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = {
            'values': {var: None for var in VARIABLES},
            'statuses': {var: f"Data {var} tidak tersedia" for var in VARIABLES},
            'last_updated': 0
        }
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "system", "content": "Asisten deteksi merokok"}]
    if 'last_servo_update' not in st.session_state:
        st.session_state.last_servo_update = 0
    if 'prev_sudut' not in st.session_state:
        st.session_state.prev_sudut = 90

    tab1, tab2 = st.tabs(["📊 Sensor IoT", "📸 Kamera ESP32"])

    # --- TAB SENSOR IoT ---
    with tab1:
        st.subheader("Data Sensor Real-Time")
        auto_refresh = st.checkbox("Auto-Refresh Data", value=True)
        if auto_refresh:
            st_autorefresh(interval=5000)

        # Narasi singkat di bawah auto-refresh
        sensor_data = fetch_latest_sensor_data()
        st.session_state.sensor_data = {
            'values': sensor_data['values'],
            'statuses': sensor_data['statuses'],
            'last_updated': time.time()
        }
        st.markdown(
            f'<div class="narasi">{generate_narasi_singkat(st.session_state.sensor_data)}</div>',
            unsafe_allow_html=True
        )

        for var_name in VARIABLES:
            value = st.session_state.sensor_data['values'].get(var_name)
            status = st.session_state.sensor_data['statuses'].get(var_name)
            data = get_ubidots_data(var_name)
            
            var_label, emoji = {
                "mq2": ("Asap/Gas", "💨"),
                "humidity": ("Kelembapan", "💧"),
                "temperature": ("Suhu", "🌡️"),
                "lux": ("Cahaya", "💡"),
                "mic": ("Amplitudo", "🎙️")
            }[var_name]

            status_class = {
                "Bahaya": "status-danger",
                "Mencurigakan": "status-warning",
                "Aman": "status-success",
                "normal": "status-success",
                "tinggi": "status-warning",
                "rendah": "status-info",
                "panas": "status-warning",
                "dingin": "status-info",
                "Sedang": "status-info",
                "Tinggi": "status-warning",
                "Rendah": "status-success"
            }.get(status.split()[0] if status else "normal", "status-info")

            st.markdown(
                f"""
                <div class="data-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="data-label">{emoji} {var_label}</span>
                        <span class="data-value">{value if value is not None else "N/A"}</span>
                    </div>
                    <div class="status-badge {status_class}">{status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(WIB)
                st.line_chart(df[['timestamp', 'value']].set_index('timestamp'))

            current_time = time.time()
            if var_name == "mq2" and "Bahaya" in status and current_time - st.session_state.last_notification['mq2']['last_alert_sent'] > ALERT_COOLDOWN:
                caption = f"🚨 *Peringatan Asap*: {status}\n📊 Nilai: {value if value is not None else 'N/A'}\n🕒 Waktu: {datetime.datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')}"
                if st.session_state.latest_frame:
                    run_async(send_telegram_photo(st.session_state.latest_frame, caption))
                else:
                    run_async(send_telegram_message(caption + "\n⚠️ Kamera tidak aktif"))
                st.session_state.last_notification['mq2']['last_alert_sent'] = current_time

        # Ringkasan
        values = st.session_state.sensor_data['values']
        if all(values.get(v) for v in VARIABLES):
            summary = get_room_condition_summary(
                values['mq2'], values['lux'], values['temperature'], values['humidity'], values['mic']
            )
            st.markdown(
                f"""
                <div class="data-card">
                    <h3 style="color: {summary['color']};">Status Ruangan: {summary['status']}</h3>
                    <p><strong>Saran:</strong> {summary['suggestion']}</p>
                    <p><strong>Detail:</strong></p>
                    <ul>
                        {"".join(f"<li>{k}: {v}</li>" for k, v in summary['details'].items())}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Chatbot
        st.subheader("💬 AI Chatbot")
        with st.form("chat_form", clear_on_submit=True):
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for msg in st.session_state.chat_messages[1:]:
                st.markdown(
                    f'<div class="chat-message {msg["role"]}-message">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            user_input = st.text_input("Tanya tentang kondisi ruangan...")
            if st.form_submit_button("Kirim"):
                st.session_state.chat_messages = [{
                    "role": "system",
                    "content": generate_chatbot_context(
                        values.get('mq2'), values.get('lux'), values['temperature'],
                        values.get('humidity'), values.get('mic')
                    )
                }]
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                with st.spinner("Menunggu AI..."):
                    response = get_gemini_response(st.session_state.chat_messages)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()

        # Prediksi Risiko
        st.subheader("🔍 Prediksi Risiko")
        if st.button("Analisis Risiko"):
            prediction = predict_smoking_risk_rule_based(values.get('mq2'), values.get('lux'), values.get('mic'))
            st.markdown(f'<div class="data-card">{prediction}</div>', unsafe_allow_html=True)

        # Panggil notifikasi periodik
        run_async(send_periodic_notification())

    # --- TAB KAMERA ---
    with tab2:
        st.subheader("Deteksi Kamera Real-Time")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            start_cam = st.checkbox("Mulai Deteksi", key="cam_start")
        with col2:
            auto_refresh = st.checkbox("Auto-Refresh", value=True, key="cam_refresh")

        st.write("Kontrol Sudut Kamera")
        sudut = st.slider("Sudut Servo (0-180)", 0, 180, 90)
        current_time = time.time()
        if st.session_state.prev_sudut != sudut and current_time - st.session_state.last_servo_update > 1:  # Debouncing: minimal 1 detik
            try:
                response = requests.put(FIREBASE_URL, json=sudut, timeout=5)
                if response.status_code == 200:
                    st.success(f"Servo diatur ke {sudut}°")
                    st.session_state.prev_sudut = sudut
                    st.session_state.last_servo_update = current_time
                else:
                    st.error(f"Gagal mengirim perintah ke Firebase: Status {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Terjadi kesalahan jaringan: {e}")

        if start_cam:
            st.session_state.cam_running = True
            if 'model_cam' not in st.session_state:
                st.session_state.model_cam = load_yolo_model()
            if auto_refresh:
                run_async(run_camera_detection(frame_placeholder, status_placeholder))
        else:
            st.session_state.cam_running = False
            if st.session_state.last_frame:
                frame_placeholder.image(st.session_state.last_frame, channels="RGB")
                status_placeholder.info("Kamera dimatikan")

    st.markdown('<div class="footer">Dibuat dengan ❤️ oleh Tim SIGMA BOYS</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
