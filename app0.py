import streamlit as st
import bcrypt
import streamlit_authenticator as stauth
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os

# Configuration du client OpenAI avec la clé API et les détails du projet
openai.api_key = os.getenv('OPENAI_API_KEY')
# Vérifier si la clé API est récupérée correctement
if openai.api_key is None:
    print("La clé API n'est pas définie.")
else:
    print("La clé API est définie.")

# Définir les fonctions load_data et preprocess_data
def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df['Questions'] = df['Questions'].astype(str)
    df['Réponses'] = df['Réponses'].astype(str)
    df['Services'] = df['Services'].astype(str)
    df['Problématiques'] = df['Problématiques'].astype(str)
    return df

@st.cache_data
def load_embeddings(df):
    embeddings = []
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=df['Questions'].tolist()
        )
        embeddings = [record['embedding'] for record in response['data']]
    except Exception as e:
        st.error(f"Erreur lors de la création des embeddings : {e}")
        return np.array([])  # Retourner un tableau vide en cas d'erreur
    return np.array(embeddings)

def get_embedding(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return np.array(response['data'][0]['embedding'])
    except Exception as e:
        st.error(f"Erreur lors de la création de l'embedding : {e}")
        return np.array([])  # Retourner un tableau vide en cas d'erreur

def find_most_similar_question(user_question, df, question_embeddings):
    user_vector = get_embedding(user_question)
    if user_vector.size == 0 or question_embeddings.size == 0:
        st.error("Erreur lors du calcul des similarités.")
        return None, 0

    similarities = cosine_similarity([user_vector], question_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    most_similar_question = df['Questions'].iloc[most_similar_idx]
    similarity_score = similarities[most_similar_idx]
    return most_similar_question, similarity_score

def generate_response_with_gpt(question, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant utile."},
                {"role": "user", "content": f"Contexte : {context}\nQuestion : {question}\nRéponse :"}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse avec GPT : {e}")
        return None

def generate_fallback_response(question):
    try:
        context = "Je n'ai pas trouvé de réponse spécifique dans la base de données. Basé sur les informations des FAQ suivantes :"
        urls = [
            "https://site.actionlogement.fr/aide/louer/logement-social/salarie/guides",
            "https://www.actionlogement.fr/faq",
            "https://mobilijeune.actionlogement.fr/faq"
        ]
        for url in urls:
            context += f"\n- {url}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant utile."},
                {"role": "user", "content": f"Question : {question}\nContexte : {context}\nRéponse :"}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse avec GPT : {e}")
        return None

# Définir les mots de passe en clair
plain_passwords = {
    "username1": "eveille1",
    "username2": "eveille2"
}

# Hasher les mots de passe avec bcrypt
hashed_passwords = {username: bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    for username, password in plain_passwords.items()}

# Créer le dictionnaire des identifiants avec les mots de passe hachés
credentials = {
    "usernames": {
        "username1": {"name": "Rym", "password": hashed_passwords["username1"]},
        "username2": {"name": "Sidik", "password": hashed_passwords["username2"]}
    }
}

# Initialiser l'objet d'authentification
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name='ac44899db8472c2e',
    cookie_key='6a69f0e95409402ebc2ba43bec19cd62',
    cookie_expiry_days=200
)

# Ajouter un logo à la barre latérale
st.sidebar.image("lg.png", use_column_width=True)

# Ajouter la présentation avant la connexion
st.sidebar.markdown("""
#### Cette plateforme est destinée au modérateur.
""")

# Utiliser correctement le paramètre fields dans la méthode login
name, authentication_status, username = authenticator.login(
    fields={'username': 'Nom d\'utilisateur', 'password': 'Mot de passe', 'submit': 'Se connecter'}
)

# Afficher le bouton de déconnexion si l'utilisateur est authentifié
if authentication_status:
    # Ajouter les boutons pour générer les messages
    if st.sidebar.button('Message de remontée'):
        st.sidebar.write("""
        Nous avons bien reçu votre message. Afin de pouvoir interroger le service en charge de votre demande, pourriez-vous nous indiquer textuellement votre nom, prénom, adresse mail liée à votre demande et numéro de dossier s’il vous plaît ? En vous remerciant par avance. Bien cordialement, ^
        """)
    if st.sidebar.button('Remontée effectuée'):
        st.sidebar.write("""
        Nous avons bien reçu votre message et vous remercions pour ces informations. Nous allons interroger le service concerné par votre problématique. Nous reviendrons vers vous dès que nous avons un retour. En nous excusant de la gêne occasionnée. Bien cordialement, ^
        """)
    if st.sidebar.button('Téléphone'):
        st.sidebar.write("""
        Bonjour, nous avons bien reçu votre message. Vous pouvez nous contacter au 0970 800 800 (appel non surtaxé), du lundi au vendredi, de 9h à 18h. 
        Autrement, voici la liste de nos agences et leurs coordonnées : https://www.actionlogement.fr/implantations 
        Bien cordialement, ^
        """)
    st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    authenticator.logout('Se déconnecter', 'sidebar')
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    st.success(f'Bienvenue {name} !')
    
    # Charger et prétraiter les données
    df = load_data('datat.csv')
    df = preprocess_data(df)

    # Charger les embeddings des questions
    question_embeddings = load_embeddings(df)

    st.title("I'm Mox 👽, ton pote pour la GRC.")
    user_question = st.text_input("Posez votre question:")

    if st.button('Obtenir une réponse'):
        if user_question:
            most_similar_question, similarity_score = find_most_similar_question(user_question, df, question_embeddings)
            if most_similar_question and similarity_score > 0.7:
                response_row = df[df['Questions'] == most_similar_question].iloc[0]
                st.write(f"Question la plus similaire : {most_similar_question}")
                st.write("Réponses :", response_row['Réponses'])
                st.write("Services :", response_row['Services'])
                st.write("Problématiques :", response_row['Problématiques'])
            elif similarity_score <= 0.5:
                fallback_response = generate_fallback_response(user_question)
                if fallback_response:
                    st.write("Réponse générée par GPT-4 :", fallback_response)
                else:
                    st.error("Impossible de générer une réponse, veuillez contacter le support.")
            else:
                st.error("La question ne correspond à aucune question dans la base de données.")
        else:
            st.error("Veuillez entrer une question pour obtenir une réponse.")
elif authentication_status is False:
    st.error("Accès refusé. Identifiants incorrects.")
elif authentication_status is None:
    st.info("Veuillez entrer vos identifiants pour vous connecter.")
