import streamlit as st
import requests

# Judul Aplikasi
st.title("Kontrol Servo dari Cloud")

# URL Firebase
FIREBASE_URL = "https://servo-control-f3c90-default-rtdb.asia-southeast1.firebasedatabase.app/servo.json"

# Slider untuk mengatur sudut servo
angle = st.slider("Atur Sudut Servo", min_value=0, max_value=180, value=90)

# Mengirim data ke Firebase
try:
    response = requests.put(FIREBASE_URL, json=angle)

    # Memberikan umpan balik ke pengguna
    if response.status_code == 200:
        st.success(f"Servo diatur ke {angle} derajat")
    else:
        st.error("Gagal mengirim perintah ke Firebase.")
except requests.exceptions.RequestException as e:
    st.error(f"Terjadi kesalahan jaringan: {e}")
