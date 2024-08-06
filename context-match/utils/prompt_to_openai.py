import json
import pandas as pd
from re import sub
from openai import OpenAI


def prompt_openai(prompt: str, api_key: str, max_tokens: int = 50) -> str:
    """
    Send a prompt to the OpenAI API and get a response.

    :param prompt: The text prompt to send to the OpenAI API.
    :param api_key: The OpenAI API key for authentication.
    :param max_tokens: The maximum number of tokens in the response.
    :return: The response text from the OpenAI API.
    """

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


def get_facts(
    df: pd.DataFrame, api_key: str, n_facts: int = 3, max_tokens: int = 50
) -> str:
    """
    Generate a list of additional fact column names based on the provided dataframe.

    :param df: The pandas DataFrame containing the data.
    :param api_key: The OpenAI API key for authentication.
    :param n_facts: The number of fact columns to generate.
    :param max_tokens: The maximum number of tokens in the response.
    :return: A string representing a list of suggested fact column names.
    """
    assert 0 < n_facts < 12, "Please request between 1 and 11 facts."

    fact_get_prompt = f"""This is my data: 
    
    {df.head().to_string()}
    
    Based on this, come up with {n_facts} column names for additional facts about the data above. 
    Return it as a Python list and nothing else."""

    response = prompt_openai(fact_get_prompt, api_key, max_tokens)
    fact_response = sub("```python", "", response)
    fact_response = sub("```", "", fact_response)

    return fact_response


def augment_column_names(
    prompt: str, col_names: list, api_key: str, max_tokens: int = 800
):
    """
    Augment column names by suggesting additional columns based on a prompt.

    :param prompt: The initial prompt describing the data.
    :param col_names: A list of existing column names in the data.
    :param api_key: The OpenAI API key for authentication.
    :param max_tokens: The maximum number of tokens in the response.
    :return: A string representing the suggested additional column names and subcategories.
    """

    prompt_augmented = (
        prompt
        + f"""\n\nBased on the columns: {col_names}, 
                                come up with additional column names that could be useful for the data above. 
                                Return it as a Python list and nothing else.
                                Based on these column names which ones can be used to create static facts. 
                                Based on these selected column static names provide me with extra subcategories for each column.
                                And based on this take into account that is for the Netherlands"""
    )

    response = prompt_openai(prompt_augmented, api_key, max_tokens)
    response = sub("```python", "", response)
    response = sub("```", "", response)

    return response


def rename_columns(df: pd.DataFrame, api_key: str, max_tokens: int = 125):
    """
    Rename the columns of a dataframe based on the content of the rows.

    :param df: The pandas DataFrame with columns to rename.
    :param api_key: The OpenAI API key for authentication.
    :param max_tokens: The maximum number of tokens in the response.
    :return: The DataFrame with renamed columns.
    """

    prompt = f"""This is my data: 
    
    {df.head().to_string()}
    
    Based on this dataframe, rename the columns if necessary based on the content of the rows. 
    Do not include underscores in the names, and ensure that there are no duplicates.
    Also, ensure that the number of column names is the same as in the original database.
    Provide this in a python list (use double quotation marks for each entry) and nothing else."""

    response = prompt_openai(prompt, api_key, max_tokens)
    response = sub("```python", "", response)
    response = sub("```", "", response)

    # print(response)

    df.columns = json.loads(response)

    return df


def get_names_columns(
    prompt: str, col_names: list, api_key: str, max_tokens: int = 800
):
    """
    Get additional column names that could be useful for the data.

    :param prompt: The initial prompt describing the data.
    :param col_names: A list of existing column names in the data.
    :param api_key: The OpenAI API key for authentication.
    :param max_tokens: The maximum number of tokens in the response.
    :return: A string representing the suggested additional column names.
    """

    prompt_augmented = (
        prompt
        + f"""\n\nBased on the columns: {col_names}, 
                                come up with additional column names that could be useful for the data above. 
                                Return it as a Python list and nothing else."""
    )

    response = prompt_openai(prompt_augmented, api_key, max_tokens)
    response = sub("```python", "", response)
    response = sub("```", "", response)

    return response


def rerank_similar_dataframes(
    prompt: str, df_ranked: dict, api_key: str, max_tokens: int = 400
):
    """
    Rerank the similar dataframes based on the prompt and description of the dataframes.

    :param prompt: The prompt to rerank the dataframes.
    :param df_ranked: A dictionary of dataframes ranked by similarity score.
    :return: The reranked dataframes.
    """
    desc = ""
    # Iterate through the dictionary
    for _, items in df_ranked.items():
        desc += items[2] + "\n"

    # print(desc)
    prompt = f"""Based on this prompt '{prompt}' and these table descriptions: {desc}.\n
                 Rank the tables according to relevance in descending order.
                 Get me the table names as a Python list and nothing else."""
    response = prompt_openai(prompt, api_key, max_tokens)
    response = sub("```python", "", response)
    response = sub("```", "", response)

    print(response)

    return


if __name__ == "__main__":

    with open("keys.json") as f:
        dic_keys = json.load(f)
        OPENAI_TOKEN = dic_keys["openAI_token"]

    df = pd.read_csv("out.csv")
    prompt = "Get me a table of firms and their employees"
    base_columns = df.columns
    new_columns = augment_column_names(prompt, base_columns, OPENAI_TOKEN)

    print(prompt)
    print(new_columns)
