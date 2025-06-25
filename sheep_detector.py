import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# YOLOv8 モデルを読み込み（yolov8n は軽量版）
model = YOLO("yolov8n.pt")  # 自動でダウンロードされます（初回のみ）

st.set_page_config(page_title="羊検出アプリ", layout="centered")
st.title("🐏 羊検出アプリ（YOLOv8 + OpenCV + Streamlit）")

uploaded_file = st.file_uploader("画像をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像読み込み・変換
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # 表示：元画像
    st.subheader("元画像")
    st.image(image_np, use_column_width=True)

    # 物体検出
    results = model.predict(image_np)

    # 結果画像（バウンディングボックス付き）
    result_img = results[0].plot()
    st.subheader("検出結果")
    st.image(result_img, use_column_width=True)

    # 検出ラベルを取得して「sheep」があるか確認
    boxes = results[0].boxes
    if boxes is not None and len(boxes.cls) > 0:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[cid] for cid in class_ids]

        if "sheep" in class_names:
            st.success("✅ 羊（sheep）が検出されました！")
        else:
            st.info("🔍 羊は検出されませんでした。")
        
        # 詳細出力
        st.write("検出されたクラス:")
        st.write(class_names)
    else:
        st.warning("⚠️ 物体が検出されませんでした。")

