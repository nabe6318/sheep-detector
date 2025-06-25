import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# YOLOv8 モデルを読み込み（軽量）
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="羊検出アプリ", layout="centered")
st.title("🐏 羊検出アプリ（YOLOv8 + Streamlit）")

uploaded_file = st.file_uploader("画像をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像を読み込んで NumPy 配列へ変換
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("元画像")
    st.image(image_np, use_container_width=True)

    # YOLOv8で物体検出
    results = model.predict(image_np)
    boxes = results[0].boxes

    # 検出結果画像を表示（バウンディングボックス付き）
    result_img = results[0].plot()
    st.subheader("検出結果")
    st.image(result_img, use_container_width=True)

    # 検出結果のテーブル表示（バウンディングボックス含む）
    if boxes is not None and len(boxes.cls) > 0:
        xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        conf = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[cid] for cid in class_ids]

        rows = []
        for i in range(len(class_ids)):
            x1, y1, x2, y2 = xyxy[i]
            width = x2 - x1
            height = y2 - y1
            rows.append({
                "クラス": class_names[i],
                "信頼度": round(float(conf[i]), 3),
                "X1": int(x1),
                "Y1": int(y1),
                "X2": int(x2),
                "Y2": int(y2),
                "幅（px）": int(width),
                "高さ（px）": int(height),
            })

        df = pd.DataFrame(rows)
        st.subheader("検出結果テーブル（座標・サイズ）")
        st.dataframe(df, use_container_width=True)

        # CSVダウンロード
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 検出結果をCSVでダウンロード",
            data=csv,
            file_name="sheep_detection_results.csv",
            mime="text/csv",
        )

        # 羊が検出されたかどうか通知
        if "sheep" in df["クラス"].values:
            st.success(f"✅ 羊（sheep）が {sum(df['クラス'] == 'sheep')} 件 検出されました。")
        else:
            st.info("🔍 羊は検出されませんでした。")
    else:
        st.warning("⚠️ 物体が検出されませんでした。")
