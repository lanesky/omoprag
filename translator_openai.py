# translator_openai.py
import os
from openai import OpenAI, RateLimitError, APIError
# Assuming config_openai.py is in the same directory or accessible path
try:
    from config_openai import OPENAI_API_KEY, OPENAI_TRANSLATION_MODEL # Adjust model if needed
except ImportError:
    # Fallback or raise error if config cannot be imported
    print("Warning: Could not import config_openai. Falling back to environment variables.")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_TRANSLATION_MODEL = "gpt-4.1-nano" # Default translation model

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found. Ensure it's in config_openai.py or environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_translation_openai(text: str, target_language: str = 'en') -> str | None:
    """
    Translates medical terminology and related text using OpenAI API.
    Specifically designed for accurate translation of medical terms and healthcare content.
    Returns the translated text or the original text if translation fails.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return text

    # Basic check: Consider adding more robust language detection if needed
    # or simply attempt translation regardless.

    prompt = f"Translate the following text accurately to {target_language}. Output only the translation, nothing else:\n\n{text}"
    print(f"Requesting translation from OpenAI model: {OPENAI_TRANSLATION_MODEL}...")

    try:
        response = client.chat.completions.create(
            model=OPENAI_TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        translated_text = response.choices[0].message.content.strip()
        if not translated_text:
             print("Warning: OpenAI returned empty translation. Using original text.")
             return text # Fallback to original
        print(f"OpenAI Translation result: '{translated_text}'")
        return translated_text
    except (RateLimitError, APIError) as e:
        print(f"Warning: OpenAI API error during translation ({type(e).__name__}). Using original text.")
        return text # Fallback to original
    except Exception as e:
        print(f"Warning: Unexpected error during translation: {e}. Using original text.")
        return text # Fallback to original