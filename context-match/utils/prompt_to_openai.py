import json
import pandas as pd
from re import sub
from openai import OpenAI


def prompt_openai(prompt: str, api_key: str, max_tokens: int = 50):

    # Get the OpenAI Client
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


# Function to generate a prompt and get facts from the OpenAI API
def get_facts(df: pd.DataFrame, api_key: str, n_facts: int = 3, max_tokens: int = 50):

    # Check that the number of facts requested is positive
    assert n_facts > 0, "Please input more than zero facts."
    assert n_facts < 12, "Please input less than 12 facts to prevent GPT overruse."

    # Create the prompt for the OpenAI API
    fact_get_prompt = f"""This is my data: 
    
    {df.head().to_string()}
    
    Based on this, come up with {n_facts} column names for additional facts about the data above. 
    Return it as a Python list and nothing else."""

    # Fetch the response from the OpenAI API using GPT-4o
    response = prompt_openai(fact_get_prompt, api_key, max_tokens)
    fact_response = sub("```python", "", response)
    fact_response = sub("```", "", fact_response)

    return fact_response

def augment_column_names(prompt:str, col_names:list, api_key: str, max_tokens: int = 800):
    
    prompt_augumented = prompt + f"""\n\nBased on the columns: {col_names}, 
                                come up with additional column names that could be useful for the data above. 
                                Return it as a Python list and nothing else.
                                Based on these column names which ones can be used to create static facts. 
                                Based on these selected column static names provide me with extra subcategories for each column.
                                And based on this take into account that is for the Netherlands"""
    
    response = prompt_openai(prompt_augumented, api_key, max_tokens)
    response = sub("```python", "", response)
    response = sub("```", "", response)

    return response


def rename_columns(df: pd.DataFrame, api_key: str, max_tokens: int = 50):

    prompt = f"""This is my data: 
    
    {df.head().to_string()}
    
    Based on this dataframe, rename the columns if necessary based on the content of the rows. 
    Do not include underscores in the names, and ensure that there are no duplicates.
    Provide this in a python list (use double quotation marks for each entry) and nothing else."""

    # Fetch the response from the OpenAI API using GPT-4o
    response = prompt_openai(prompt, api_key, max_tokens)
    response = sub("```python", "", response)
    response = sub("```", "", response)

    df.columns = json.loads(response)

    return df


if __name__ == "__main__":

    import json

    with open("keys.json") as f:
        dic_keys = json.load(f)
        OPENAI_TOKEN = dic_keys["openAI_token"]

    df = pd.read_csv("out.csv")
    # df_new = rename_columns(df, OPENAI_TOKEN, max_tokens=70)
    # df_new.to_csv("test.csv", index=False)

    prompt = "Get me a table of firms and their employees"

    base_columns = df.columns
    new_columns = augment_column_names(prompt, base_columns, OPENAI_TOKEN)
    print(prompt)
    print(new_columns)

