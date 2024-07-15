import json
from utils import *
from argparse import ArgumentParser

# Argparser arguments:


# Execution
if __name__ == "__main__":

    # Notion API Token Get
    with open("notion.json") as f:
        NOTION_TOKEN = json.load(f)["KEY"]

    print(NOTION_TOKEN)
