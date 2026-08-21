import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="InspectFlow AI", page_icon="🏗️", layout="wide")

st.title("🏗️ InspectFlow AI - Analyse de Dégradation")

# Chargement sécurisé du modèle standard
@st.cache_resource
def load_model():
    # yolov8n-seg.pt est téléchargé automatiquement par Ultralytics sans aucune restriction
    return YOLO("yolov8n-seg.pt")

model = None
try:
    model = load_model()
    st.sidebar.success("✅ Modèle chargé avec succès !")
except Exception as e:
    st.sidebar.error(f"❌ Erreur lors du chargement du modèle : {e}")

# Sidebar : Configuration
st.sidebar.header("⚙️ Configuration du Modèle")
conf_threshold = st.sidebar.slider("Seuil de Confiance (Confidence)", 0.01, 1.0, 0.15, 0.01)

st.sidebar.header("📊 Seuils de Sévérité (%)")
seuil_faible = st.sidebar.number_input("Seuil Faible (< %)", value=1.5, step=0.5)
seuil_moyen = st.sidebar.number_input("Seuil Moyen (< %)", value=4.0, step=0.5)

uploaded_file = st.file_uploader("Choisissez une image à analyser...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("Le modèle n'est pas chargé. Impossible d'effectuer l'analyse.")
    else:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Dimensions
        h, w = img_array.shape[:2]
        total_pixels = h * w
        
        # Prédiction
        results = model.predict(source=img_array, conf=conf_threshold)
        res = results[0]
        
        annotated_frame = res.plot()
        
        # Calcul de surface
        impacted_pixels = 0
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()
            combined_mask = np.zeros((h, w), dtype=bool)
            
            for mask in masks:
                mask_resized = cv2.resize(mask, (w, h)) > 0.5
                combined_mask = np.logical_or(combined_mask, mask_resized)
                
            impacted_pixels = int(np.sum(combined_mask))
        
        severity_rate = (impacted_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Diagnostic
        st.header("📌 Résultats de l'Analyse")
        col1, col2, col3 = st.columns(3)
        col1.metric("Surface Totale Analysée", f"{total_pixels:,} px")
        col2.metric("Surface Impactée", f"{impacted_pixels:,} px")
        col3.metric("Taux de Sévérité", f"{severity_rate:.2f} %")
        
        if impacted_pixels == 0:
            st.success("Diagnostic : AUCUN DÉFAUT DÉTECTÉ")
            st.info("Recommandation Technique : Structure en bon état apparent.")
        elif severity_rate < seuil_faible:
            st.warning(f"Diagnostic : DÉGRADATION FAIBLE ({severity_rate:.2f}%)")
            st.info("Recommandation Technique : Surveillance recommandée lors des prochaines inspections.")
        elif severity_rate < seuil_moyen:
            st.warning(f"Diagnostic : DÉGRADATION MOYENNE ({severity_rate:.2f}%)")
            st.info("Recommandation Technique : Intervention de colmatage recommandée à moyen terme.")
        else:
            st.error(f"Diagnostic : DÉGRADATION SÉVÈRE ({severity_rate:.2f}%)")
            st.info("Recommandation Technique : INTERVENTION RAPIDE REQUISE.")

        st.write("---")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(image, caption="Image originale", use_column_width=True)
        with col_img2:
            st.image(annotated_frame, caption="Détection & Segmentation", use_column_width=True)
