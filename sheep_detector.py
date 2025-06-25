import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# YOLOv8 モデルを読み込み（軽量版）
model = YOLO("yolov8n.pt")  # 自動でDL（sheepを含む）

st.set_page_config(page_title="羊検出アプリ", layout="centered")
st.title("🐏 羊検出アプリ（YOLOv8 + Streamlit）")

uploaded_file = st.file_uploader("画像をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像読み込み・変換
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("元画像")
    st.image(image_np, use_column_width=True)

    # 物体検出
    results = model.predict(image_np)
    boxes = results[0].boxes

    # 結果画像（バウンディングボックス付き）
    result_img = results[0].plot()
    st.subheader("検出結果")
    st.image(result_img, use_column_width=True)

    # sheep のみ抽出
    if boxes is not None and len(boxes.cls) > 0:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names_all = [model.names[cid] for cid in class_ids]

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        # sheep のみフィルター
        sheep_rows = []
        for i, cls in enumerate(class_names_all):
            if cls == "sheep":
                sheep_rows.append({
                    "クラス": cls,
                    "信頼度": round(float(conf[i]), 3),
                    "X1": int(xyxy[i][0]),
                    "Y1": int(xyxy[i][1]),
                    "X2": int(xyxy[i][2]),
                    "Y2": int(xyxy[i][3]),
                })

        if len(sheep_rows) > 0:
            st.success(f"✅ 羊（sheep）が {len(sheep_rows)} 件 検出されました。")

            # データフレーム表示
            df = pd.DataFrame(sheep_rows)
            st.subheader("検出された羊の情報")
            st.dataframe(df, use_container_width=True)

            # CSVダウンロード
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 CSVとしてダウンロード",
                data=csv,
                file_name='sheep_detection.csv',
                mime='text/csv',
            )
        else:
            st.info("🔍 画像に羊は検出されませんでした。")
    else:
        st.warning("⚠️ 物体が検出されませんでした。")

