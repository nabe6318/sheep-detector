import streamlit as st
import urllib.request
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Google Driveからファイルをダウンロードする関数
@st.cache_resource
def download_model_from_gdrive(file_id, destination):
    if os.path.exists(destination):
        return destination

    url = f"https://drive.google.com/uc?export=download&id={1WAmeHK5ec7lzilw1_NNag9CKqQcLwnld}"
    with st.spinner("モデルをダウンロード中..."):
        urllib.request.urlretrieve(url, destination)
    return destination

# ファイルIDを指定（ここに自分のファイルIDを入れる）
GDRIVE_FILE_ID = "1WAmeHK5ec7lzilw1_NNag9CKqQcLwnld"  # ←自分のGoogle DriveのIDに置き換えてください
MODEL_PATH = "yolov8x.pt"

# モデルを取得して読み込み
model_file = download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
model = YOLO(model_file)

# Streamlit UI
st.set_page_config(page_title="YOLO 独自モデル - Drive連携", layout="centered")
st.title("📦 YOLOv8 独自モデル（Google Drive経由）")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="元画像", use_container_width=True)

    results = model.predict(image_np)
    result_img = results[0].plot()

    st.subheader("検出結果")
    st.image(result_img, use_container_width=True)

