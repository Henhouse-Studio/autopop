import os

dir_path = os.path.dirname(os.path.dirname(__file__))
SAVE_FOLDER = os.path.join(dir_path, "chat_logs")
KEYFILE_LOC = os.path.join(dir_path, "keys.json")
FAVICON_LOC = os.path.join(dir_path, "assets/images/henhouse.png")
STYLE_MAIN = os.path.join(dir_path, "assets/css/main.css")
STYLE_SETTINGS = os.path.join(dir_path, "assets/css/settings.css")

# For database files
DATA_DIR = os.path.join(dir_path, "databases/table_of_tables")
FIELD_DIR = os.path.join(dir_path, "databases/table_of_fields")
MERGED_DIR = os.path.join(dir_path, "databases/merged_cache.csv")
SAVE_MERGE_DIR = os.path.join(dir_path, "databases/merge_cache")
