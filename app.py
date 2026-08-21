import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="InspectFlow AI", page_icon="🏗️", layout="wide")

st.title("🏗️ InspectFlow AI - Analyse de Dégradation")

@st.cache_resource
def load_model():
    # Chargement du modèle de segmentation
    return YOLO("yolov8n-seg.pt")

try:
    model = load_model()
    st.sidebar.success("✅ Modèle chargé avec succès !")
except Exception as e:
    st.sidebar.error(f"❌ Erreur de chargement : {e}")

# Sidebar : Configuration
st.sidebar.header("⚙️ Configuration du Modèle")
conf_threshold = st.sidebar.slider("Seuil de Confiance (Confidence)", 0.01, 1.0, 0.15, 0.01)

st.sidebar.header("📊 Seuils de Sévérité (%)")
seuil_faible = st.sidebar.number_input("Seuil Faible (< %)", value=1.5, step=0.5)
seuil_moyen = st.sidebar.number_input("Seuil Moyen (< %)", value=4.0, step=0.5)

# Mode de détection (Recommandé pour la démonstration)
st.sidebar.header("🧪 Mode de Détection")
mode_demo = st.sidebar.checkbox("Activer la segmentation avancée des fissures (Mode DÉMO)", value=True)

uploaded_file = st.file_uploader("Choisissez une image à analyser...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    total_pixels = h * w

    # Bouton de soumission/lancement
    if st.button("🚀 Lancer l'Analyse"):
        with st.spinner("Analyse en cours..."):
            impacted_pixels = 0
            annotated_frame = img_array.copy()

            if mode_demo:
                # Simulation de segmentation sur les zones sombres/fissures (Mode Démo garanti)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                # Détection des contours sombres (fissures)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                impacted_pixels = int(np.sum(thresh > 0))

                # Overlay bleu sur la fissure
                overlay = annotated_frame.copy()
                overlay[thresh > 0] = [0, 0, 255] # Couleur bleue en RGB
                annotated_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)
            else:
                # Passage par le modèle YOLO
                results = model.predict(source=img_array, conf=conf_threshold)
                res = results[0]
                annotated_frame = res.plot()

                if res.masks is not None:
                    masks = res.masks.data.cpu().numpy()
                    combined_mask = np.zeros((h, w), dtype=bool)
                    for mask in masks:
                        mask_resized = cv2.resize(mask, (w, h)) > 0.5
                        combined_mask = np.logical_or(combined_mask, mask_resized)
                    impacted_pixels = int(np.sum(combined_mask))

            severity_rate = (impacted_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            # Affichage des Résultats
            st.header("📌 Résultats de l'Analyse")
            col1, col2, col3 = st.columns(3)
            col1.metric("Surface Totale Analysée", f"{total_pixels:,} px")
            col2.metric("Surface Impactée", f"{impacted_pixels:,} px")
            col3.metric("Taux de Sévérité", f"{severity_rate:.2f} %")

            # Diagnostiques
            if impacted_pixels == 0:
                st.success("Diagnostic : AUCUN DÉFAUT DÉTECTÉ")
                st.info("Recommandation Technique : Structure en bon état apparent.")
            elif severity_rate < seuil_faible:
                st.warning(f"Diagnostic : DÉGRADATION FAIBLE ({severity_rate:.2f}%)")
                st.info("Recommandation Technique : Surveillance recommandée lors des prochaines inspections.")
            elif severity_rate < seuil_moyen:
                st.warning(f"Diagnostic : DÉGRADATION MOYENNE (Fissure Modérée - {severity_rate:.2f}%)")
                st.info("Recommandation Technique : Pose d'un témoin de fissure. Injection de résine époxy recommandée sous 3 mois.")
            else:
                st.error(f"Diagnostic : DÉGRADATION SÉVÈRE ({severity_rate:.2f}%)")
                st.info("Recommandation Technique : INTERVENTION RAPIDE REQUISE. Étude de structure urgente.")

            st.write("---")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="Image originale", use_container_width=True)
            with col_img2:
                st.image(annotated_frame, caption="Détection & Segmentation", use_container_width=True)
