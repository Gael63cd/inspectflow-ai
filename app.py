import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="InspectFlow AI", page_icon="🏗️", layout="wide")

st.title("🏗️ InspectFlow AI - Analyse de Dégradation")

st.sidebar.success("✅ Modèle InspectFlow AI prêt !")

# Sidebar : Configuration
st.sidebar.header("⚙️ Configuration du Modèle")
conf_threshold = st.sidebar.slider("Seuil de Sensibilité", 0.01, 1.0, 0.30, 0.01)

st.sidebar.header("📊 Seuils de Sévérité (%)")
seuil_faible = st.sidebar.number_input("Seuil Faible (< %)", value=1.5, step=0.5)
seuil_moyen = st.sidebar.number_input("Seuil Moyen (< %)", value=4.0, step=0.5)

uploaded_file = st.file_uploader("Choisissez une image à analyser...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Correction automatique si image avec canal Alpha (RGBA)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    h, w = img_array.shape[:2]
    total_pixels = h * w

    if st.button("🚀 Lancer l'Analyse", type="primary"):
        with st.spinner("Analyse de la structure en cours..."):
            # Traitement de l'image pour la détection de fissures
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Seuil dynamique adapté à la sensibilité réglée
            thresh_val = int(255 * (1 - conf_threshold))
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Nettoyage du bruit
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # Calcul exact de la surface impactée
            impacted_pixels = int(np.sum(thresh > 0))
            severity_rate = (impacted_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            # Création du masque visuel bleu sur la fissure
            annotated_frame = img_array.copy()
            blue_mask = np.zeros_like(img_array)
            blue_mask[thresh > 0] = [0, 102, 255] # Bleu brillant
            
            # Superposition du masque sur l'image
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, blue_mask, 0.5, 0)

            # Affichage des Métriques
            st.header("📌 Résultats de l'Analyse")
            col1, col2, col3 = st.columns(3)
            col1.metric("Surface Totale Analysée", f"{total_pixels:,} px".replace(",", " "))
            col2.metric("Surface Impactée", f"{impacted_pixels:,} px".replace(",", " "))
            col3.metric("Taux de Sévérité", f"{severity_rate:.2f} %")

            # Diagnostic & Recommandation
            if impacted_pixels == 0 or severity_rate < 0.1:
                st.success("Diagnostic : AUCUN DÉFAUT MAJEUR DÉTECTÉ")
                st.info("Recommandation Technique : Structure en bon état apparent.")
            elif severity_rate < seuil_faible:
                st.warning(f"Diagnostic : DÉGRADATION FAIBLE ({severity_rate:.2f}%)")
                st.info("Recommandation Technique : Surveillance recommandée lors des prochaines inspections.")
            elif severity_rate < seuil_moyen:
                st.warning(f"Diagnostic : MOYEN (Fissure Modérée - {severity_rate:.2f}%)")
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
