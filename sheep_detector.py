import streamlit as st
import numpy as np  
import pandas as pd  
from PIL import Image, ImageDraw, ImageFont  
from ultralytics import YOLO  

# モデル読み込み
model = YOLO("yolov8n.pt")  

st.set_page_config(page_title="羊検出アプリ", layout="centered")  
st.title("🐏 羊検出アプリ（YOLOv8 + Streamlit）")  

uploaded_file = st.file_uploader("画像をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])  

if uploaded_file is not None:  
    image = Image.open(uploaded_file).convert("RGB")  
    image_np = np.array(image)  

    st.subheader("元画像")  
    st.image(image_np, use_container_width=True)  

    # YOLOv8で推論
    results = model.predict(image_np)  
    boxes = results[0].boxes  

    if boxes is not None and len(boxes.cls) > 0:  
        xyxy = boxes.xyxy.cpu().numpy()  
        conf = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy().astype(int)  
        class_names = [model.names[cid] for cid in class_ids]  

        rows = []  
        sheep_count = 0  

        # PILで描画する準備
        annotated_img = image.copy()  
        draw = ImageDraw.Draw(annotated_img)  
        try:
            font = ImageFont.truetype("arial.ttf", size=18)
        except:
            font = ImageFont.load_default()

        for i in range(len(class_ids)):  
            if class_names[i] == "sheep":  
                sheep_count += 1  
                x1, y1, x2, y2 = map(int, xyxy[i])  
                width = x2 - x1  
                height = y2 - y1  

                # 黒色で枠線と通し番号だけ描く
                draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=2)  
                draw.text((x1, y1 - 20), str(sheep_count), fill="black", font=font)

                rows.append({  
                    "クラス": "sheep",  
                    "信頼度": round(float(conf[i]), 3),  
                    "X1": x1,  
                    "Y1": y1,  
                    "X2": x2,  
                    "Y2": y2,  
                    "幅（px）": width,  
                    "高さ（px）": height,  
                })  

        if sheep_count > 0:  
            st.subheader("検出結果（sheep のみ）")  
            st.image(annotated_img, use_container_width=True)  

            df = pd.DataFrame(rows)  
            st.subheader("検出結果テーブル（sheep のみ）")  
            st.dataframe(df, use_container_width=True)  

            csv = df.to_csv(index=False).encode("utf-8-sig")  
            st.download_button(  
                label="📥 検出結果をCSVでダウンロード",  
                data=csv,  
                file_name="sheep_detection_results.csv",  
                mime="text/csv",  
            )  

            st.success(f"✅ 羊（sheep）が {sheep_count} 件 検出されました。")  
        else:  
            st.info("🔍 羊は検出されませんでした。")  
    else:  
        st.warning("⚠️ 物体が検出されませんでした。")  