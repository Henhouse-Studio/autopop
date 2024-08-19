import os
import json
import warnings
from langchain._api import LangChainDeprecationWarning


# For loading the OpenAI API key
def load_openai_api():

    with open("/home/gregorygo/autopop/llamaindex/keys.json") as f:
        dic_keys = json.load(f)
        os.environ["OPENAI_API_KEY"] = dic_keys["openAI_token"]


# To disable the import deprecation warning from LangChain
def disable_warning():
    warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
