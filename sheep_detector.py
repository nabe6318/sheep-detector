import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# YOLOv8 モデル読み込み
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="羊検出アプリ", layout="centered")
st.title("🐏 羊検出アプリ（YOLOv8 + Streamlit）")

uploaded_file = st.file_uploader("画像をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像読み込み
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("元画像")
    st.image(image_np, use_container_width=True)

    # YOLOv8 推論
    results = model.predict(image_np)
    boxes = results[0].boxes

    if boxes is not None and len(boxes.cls) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[cid] for cid in class_ids]

        rows = []
        annotated_img = image_np.copy()

        sheep_count = 0
        for i in range(len(class_ids)):
            if class_names[i] == "sheep":
                sheep_count += 1
                x1, y1, x2, y2 = map(int, xyxy[i])
                confidence = round(float(conf[i]), 3)
                label = f"{sheep_count}: sheep ({confidence})"

                # バウンディングボックス描画（緑）
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 表用データに追加
                rows.append({
                    "通し番号": sheep_count,
                    "信頼度": confidence,
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                    "幅（px）": x2 - x1,
                    "高さ（px）": y2 - y1,
                })

        if sheep_count > 0:
            # 結果画像（BGR→RGB）
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.subheader("検出結果（sheep のみ番号付き）")
            st.image(annotated_img, use_container_width=True)

            # テーブル表示
            df = pd.DataFrame(rows)
            st.subheader("検出結果テーブル（sheep のみ）")
            st.dataframe(df, use_container_width=True)

            # CSVダウンロード
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 検出結果をCSVでダウンロード",
                data=csv,
                file_name="sheep_detection_results.csv",
                mime="text/csv",
            )

            st.success(f"✅ 羊（sheep）が {sheep_count} 件 検出されました。")
        else:
            st.info("🔍 画像に羊は検出されませんでした。")
    else:
        st.warning("⚠️ 物体が検出されませんでした。")