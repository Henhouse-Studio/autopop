import logging
import warnings

def suppress_warnings():

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

    # Suppress SentenceTransformer, transformers, and general logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    # logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    # logging.basicConfig(level=logging.WARNING)