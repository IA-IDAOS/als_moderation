from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import preprocess_data, load_data
import transformers
import tensorflow as tf

# Configuration pour BERT
feature_extractor = pipeline("feature-extraction", model="bert-base-uncased")

def encode_questions(data):
    return {row['Questions']: feature_extractor(row['Full Context'])[0][0] for index, row in data.iterrows()}

def get_most_similar_question(user_question, encoded_questions):
    user_question_vec = feature_extractor(user_question)[0][0]
    similarities = {question: cosine_similarity([user_question_vec], [vec]).flatten()[0] for question, vec in encoded_questions.items()}
    return max(similarities, key=similarities.get), similarities[max(similarities, key=similarities.get)]
