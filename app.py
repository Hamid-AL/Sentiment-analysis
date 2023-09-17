import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained("./Model")
model = AutoModelForSequenceClassification.from_pretrained("./Model")
# Streamlit app title
st.title("Analyse de Sentiment avec Votre Modèle")

# Text input pour que l'utilisateur entre le texte
user_text = st.text_area("Entrez le texte pour l'analyse de sentiment")

# Effectuer la prédiction de sentiment lorsque l'utilisateur clique sur un bouton
if st.button("Prédire le Sentiment"):
    if user_text:
        # Prétraiter le texte pour l'entrée du modèle
        inputs = tokenizer(user_text, return_tensors="pt")
        
        # Effectuer la prédiction avec le modèle
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Obtenir les prédictions et les probabilités associées
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()+1
        #predicted_prob = torch.softmax(logits, dim=1)[0][predicted_class].item()
        if predicted_class>=3:
            st.write("prediction:", 'Positve')
        else: 
            st.write("prediction:", 'Negative')
        # Afficher le résultat
        #st.write("Label de sentiment prédit:", predicted_class)
        #st.write("Probabilité associée:", predicted_prob)
    else:
        st.write("Veuillez entrer du texte à analyser.")