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

        # 検出詳細を表形式で表示
        xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        conf = boxes.conf.cpu().numpy()  # confidence
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[cid] for cid in class_ids]

        # 表データの作成
        rows = []
        for i in range(len(class_ids)):
            rows.append({
                "クラス": class_names[i],
                "信頼度": round(float(conf[i]), 3),
                "X1": int(xyxy[i][0]),
                "Y1": int(xyxy[i][1]),
                "X2": int(xyxy[i][2]),
                "Y2": int(xyxy[i][3]),
            })

        st.subheader("検出結果テーブル")
        st.dataframe(rows, use_container_width=True)
