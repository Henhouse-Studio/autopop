import os

dir_path = os.path.dirname(os.path.dirname(__file__))
SAVE_FOLDER = os.path.join(dir_path, "chat_logs")
KEYFILE_LOC = os.path.join(dir_path, "keys.json")
FAVICON_LOC = os.path.join(dir_path, "assets/images/henhouse.png")
STYLE_MAIN = os.path.join(dir_path, "assets/css/main.css")
STYLE_SETTINGS = os.path.join(dir_path, "assets/css/settings.css")
COMMAND_EMB = os.path.join(dir_path, "assets/command_emb.json")

# For database files
DATA_DIR = os.path.join(dir_path, "databases/table_of_tables")
FIELD_DIR = os.path.join(dir_path, "databases/table_of_fields")
MERGED_DIR = os.path.join(dir_path, "databases/merged_cache.csv")
SAVE_MERGE_DIR = os.path.join(dir_path, "databases/merge_cache")

# For settings
TOL_EXAMPLE = os.path.join(dir_path, "assets/csv/threshold_example.csv")
TABLE_EXAMPLE = os.path.join(dir_path, "assets/csv/matching_example.csv")
