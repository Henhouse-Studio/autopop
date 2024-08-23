import os
import json

with open("keys.json") as f:
    dic_keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = dic_keys["openAI_token"]


from llama_index.core import TreeIndex, SimpleDirectoryReader

resume = SimpleDirectoryReader(
    "/home/gregorygo/autopop/context-match/databases/table_of_tables"
).load_data()
new_index = TreeIndex.from_documents(resume)

query_engine = new_index.as_query_engine()
response = query_engine.query(
    "Where does TechTrends work? Based on the confidence values, how probable is this fact?"
)
print(response)
