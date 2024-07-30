print("Initialisation du package src...")

# Import de fonctions spécifiques des modules pour faciliter l'accès via le package
from .data_processing import load_data, preprocess_data
from .model import encode_questions, get_most_similar_question

__all__ = ['load_data', 'preprocess_data', 'encode_questions', 'get_most_similar_question']
