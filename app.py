import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="InspectFlow AI - Diagnostic de Fissures",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 36px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
    .sub-header { font-size: 18px; text-align: center; color: #4B5563; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏗️ InspectFlow AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Système d\'Inspection Intelligente et de Diagnostic de Sévérité des Fissures Structurelles</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration du Modèle")

# Recherche automatique du modèle présent dans le projet
if os.path.exists('best.pt'):
    model_path = 'best.pt'
elif os.path.exists('yolov8n-seg.pt'):
    model_path = 'yolov8n-seg.pt'
else:
    uploaded_model = st.sidebar.file_uploader("Importer votre modèle YOLO (.pt)", type=["pt"])
    if uploaded_model:
        model_path = "temp_model.pt"
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
    else:
        model_path = None

conf_threshold = st.sidebar.slider("Seuil de Confiance (Confidence)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
threshold_low = st.sidebar.number_input("Seuil Faible (< %)", value=1.5, step=0.5)
threshold_medium = st.sidebar.number_input("Seuil Moyen (< %)", value=4.0, step=0.5)

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

def process_image(img_pil, model, conf):
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    total_pixels = h * w

    results = model.predict(source=img_bgr, conf=conf)
    result = results[0]

    res_plotted = result.plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    crack_pixels = 0
    if result.masks is not None:
        for mask in result.masks.data:
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h))
            crack_pixels += np.sum(mask_resized > 0.5)

    ratio = (crack_pixels / total_pixels) * 100
    return res_rgb, ratio, crack_pixels, total_pixels

if model_path is None:
    st.warning("⚠️ Veuillez uploader votre fichier de modèle dans le menu latéral.")
else:
    try:
        model = load_yolo_model(model_path)
        st.sidebar.success(f"✅ Modèle chargé ({os.path.basename(model_path)}) !")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

    uploaded_file = st.file_uploader("📂 Choisissez une image d'inspection (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if st.button("🔍 Lancer le Diagnostic InspectFlow AI", type="primary"):
            with st.spinner("Analyse et segmentation en cours..."):
                segmented_img, ratio, crack_px, total_px = process_image(image, model, conf_threshold)

            if ratio == 0:
                status, color = "AUCUN DÉFAUT DÉTECTÉ", "green"
                recommendation = "Structure en bon état apparent. Prochaine inspection selon calendrier standard."
            elif ratio < threshold_low:
                status, color = "FAIBLE (Fissure Superficielle)", "green"
                recommendation = "Nettoyage de surface et colmatage préventif recommandé."
            elif ratio < threshold_medium:
                status, color = "MOYEN (Fissure Modérée)", "orange"
                recommendation = "Pose d'un témoin de fissure. Injection de résine époxy recommandée sous 3 mois."
            else:
                status, color = "ÉLEVÉ (Risque Structurel Majeur)", "red"
                recommendation = "🚨 INTERVENTION URGENTE REQUISE. Étayage de sécurité et expertise immédiate."

            st.markdown("---")
            st.subheader("📌 Résultats de l'Analyse")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1: st.metric(label="Surface Totale Analysée", value=f"{total_px:,} px")
            with col_m2: st.metric(label="Surface Impactée", value=f"{crack_px:,} px")
            with col_m3: st.metric(label="Taux de Sévérité", value=f"{ratio:.2f} %")

            st.markdown(f"### Diagnostic : :{color}[**{status}**]")
            st.info(f"**Recommandation Technique :** {recommendation}")

            col_img1, col_img2 = st.columns(2)
            with col_img1: st.image(image, caption="Image originale", use_container_width=True)
            with col_img2: st.image(segmented_img, caption="Détection & Segmentation", use_container_width=True)