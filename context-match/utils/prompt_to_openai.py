import json
import pandas as pd
from re import sub
from openai import OpenAI

# Load the API key from the JSON file
with open("keys.json", "r") as f:
    api_key = json.load(f)["openAI_token"]

client = OpenAI(api_key=api_key)

# Load your CSV file
df = pd.read_csv("out.csv")

# Function to generate a prompt and get facts from the OpenAI API
def get_facts(df: pd.DataFrame, n_facts: int = 3):
    # Get the base column names
    colnames = list(df.columns)

    # Check that the number of facts requested is positive
    assert n_facts > 0, "Please input more than zero facts."
    assert n_facts < 12, "Please input less than 12 facts to prevent GPT overruse."

    # Create the prompt for the OpenAI API
    fact_get_prompt = f"""This is my data: 
    
    {df.head().to_string()}
    
    Based on this, come up with {n_facts} column names for additional facts about the data above. 
    Return it as a Python list and nothing else."""

    # Fetch the response from the OpenAI API using GPT-4 Turbo
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": fact_get_prompt},
        ],
        max_tokens=50,
    )
    fact_response = response.choices[0].message.content.strip()
    fact_response = sub("```python", "", fact_response)
    fact_response = sub("```", "", fact_response)

    return fact_response


# Get the facts for the data in the DataFrame
response = get_facts(df)

print(response)
