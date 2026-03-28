from openai import OpenAI

API_KEY = "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"

client = OpenAI(api_key=API_KEY)

prompt = """
You are a top-tier startup founder and hackathon judge.
Give me 3 highly original hackathon ideas for the theme 'Better at Home'.
Keep each one concise.
"""

response = client.responses.create(
    model="gpt-5.4",
    input=prompt
)

print(response.output_text)