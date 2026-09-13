import os

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-20b"


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    print("ZOOZ Groq diagnostic")
    print("- GROQ_API_KEY present:", bool(api_key))
    print("- Model:", MODEL)

    if not api_key:
        print("RESULT: FAIL")
        print("CAUSE: GROQ_API_KEY is missing from the current environment/.env file.")
        return

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            max_tokens=10,
        )
        text = (response.choices[0].message.content or "").strip()
        print("- Model response:", repr(text))
        print("RESULT: PASS")
    except Exception as exc:
        print("RESULT: FAIL")
        print("ERROR TYPE:", type(exc).__name__)
        print("ERROR:", str(exc))


if __name__ == "__main__":
    main()
