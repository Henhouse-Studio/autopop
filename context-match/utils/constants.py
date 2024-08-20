import os

dir_path = os.path.dirname(os.path.dirname(__file__))
SAVE_FOLDER = os.path.join(dir_path, "chat_logs")
KEYFILE_LOC = os.path.join(dir_path, "keys.json")
FAVICON_LOC = os.path.join(dir_path, "assets/henhouse.png")
