from misc import *
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext

load_openai_api()
disable_warning()

# Load your CSV files
docs = SimpleDirectoryReader("/home/gregorygo/autopop/llamaindex/data").load_data()

# Create a KnowledgeGraphIndex
storage_context = StorageContext.from_defaults()
index = KnowledgeGraphIndex.from_documents(docs, storage_context=storage_context)

# Save the index
storage_context.persist(persist_dir="./storage")

# # To load the index later:
# loaded_storage_context = StorageContext.from_defaults(persist_dir="./storage")
# loaded_index = load_index_from_storage(storage_context=loaded_storage_context)

# # Now you can use the loaded index
# query_engine = loaded_index.as_query_engine(
#     include_text=False, response_mode="tree_summarize"
# )
# response = query_engine.query(
#     "Tell me more about TechTrends and what their real identity is."
# )
# print(response)
