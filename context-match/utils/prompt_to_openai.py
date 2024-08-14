import sys
import json
import pprint
import argparse
import pandas as pd
from re import sub
from openai import OpenAI


def clean_output(response: str):
    """
    Function to clean the GPT Python code output.

    :param response: The text response from the OpenAI API.
    :return: The cleaned response (str).
    """

    new_response = sub("```python", "", response)
    new_response = sub("```", "", new_response)
    new_response = new_response.replace("'", '"')

    return new_response


def prompt_openai(
    prompt: str,
    api_key: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
):
    """
    Function to send a prompt to the OpenAI API and get a response.

    :param prompt: The text prompt to send to the OpenAI API.
    :param api_key: The OpenAI API key for authentication.
    :param temperature: The randomness of the response (default is 0.0).
    :param max_tokens: The maximum number of tokens in the response.
    :param model: The OpenAI model to use.
    :return: The response text from the OpenAI API (str).
    """

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


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

    return clean_output(response)


def rename_columns(df: pd.DataFrame, api_key: str, max_tokens: int = 200):
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
    df.columns = json.loads(clean_output(response))

    return df


def get_names_columns(
    prompt: str, col_names: list, api_key: str, max_tokens: int = 500
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
    response = clean_output(response)

    return response


def rerank_dataframes(
    original_prompt: str, df_ranked: dict, api_key: str, max_tokens: int = 200
):
    """
    Rerank the similar dataframes based on the prompt and description of the dataframes using ChatGPT.

    :param prompt: The prompt to rerank the dataframes.
    :param df_ranked: A dictionary of dataframes ranked by similarity score.
    :return: The reranked dataframes.
    """
    desc_batches = []
    desc = ""

    # Iterate through the dictionary
    for _, items in df_ranked.items():
        desc += items[2] + "\n"

        # count whether the desc as exceeded 24000 words
        if len(desc) > 24_000:
            desc_batches.append(desc)
            desc = ""

    response_batches = ""
    # The desc should never be more than 2400 words so just append to desc_batches
    if desc_batches == []:
        desc_batches.append(desc)

    for desc in desc_batches:
        prompt = f"""Based on this prompt '{original_prompt}' and these table descriptions: {desc}.\n
                    Select the most relevant tables and then sort them according to relevance in descending order.
                    Get me the table names as a Python list and nothing else.
                    If none of the tables are relevant, return an empty list."""
        response = prompt_openai(prompt, api_key, max_tokens)
        response = clean_output(response)
        response_batches += response

    return json.loads(response_batches)


def get_relevant_columns(
    prompt: str,
    df_ranked: dict,
    api_key: str,
    args: argparse.Namespace = None,
    max_tokens: int = 200,
    verbose: bool = False,
):
    """
    Get and weight the relevant columns from the dataframes based on the prompt.

    :param prompt: The prompt to get the relevant columns.
    :param df_ranked: A dictionary of dataframes ranked by similarity score.
                    : df_ranked[table_name] = (similarity_score, df, desc)
    :param api_key: The OpenAI API key for authentication.
    :param verbose: Whether to print the scores (bool, default = False).
    :return: A dictionary of relevant columns for each dataframe (dict).
    """

    print("\nGetting relevant columns...")
    dict_weights = {}
    for table_name, (_, _, desc) in df_ranked.items():

        prompt = f"""Based on this prompt:\n
                    {prompt}\n
                    Analyze the below dataframe and select the most relevant column names based on the above prompt.
                    {desc}
                    Return only the relevant column names as a python dictionary (outside of a variable), with the key being the
                    column name, and the value being the importance score of said column with respect to the prompt.
                    Ensure that you only return column names that are present in the above dataframe.
                    Ensure the importance scores are all integers (ranging from 1-10, with 10 being the highest score) 
                    and only return the dictionary.
                    """

        response = prompt_openai(
            prompt, api_key, max_tokens=max_tokens, temperature=args.temperature
        )

        dict_weights[table_name] = json.loads(clean_output(response))

    # For printing
    if verbose:
        pprint.pprint(dict_weights)

    return dict_weights


# Unused
def get_facts(df: pd.DataFrame, api_key: str, n_facts: int = 3, max_tokens: int = 50):
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
