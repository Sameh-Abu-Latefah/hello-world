import json
import os

from dotenv import load_dotenv
from groq import Groq


def  ask_llm(system_prompt, user_message, model, temperature=0):
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    client = Groq(api_key=key)
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response = client.chat.completions.create( model=model, messages=message, temperature=temperature,)
    return  response.choices[0].message.content

def parse_label(text):


        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return "invalid"

        if not isinstance(data, dict):
            return "invalid"

        label = data.get("label")

        if not isinstance(label, str):
            return "invalid"

        label = label.strip().lower()

        if label in {"positive", "negative", "neutral"}:
            return label
        return "invalid"



