import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_response_with_gpt(user_question, context):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = f"Context: {context}\nQuestions: {user_question}\nRéponses:"
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()
