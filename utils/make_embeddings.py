import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_embeddings(embeddings):

    return embeddings / np.linalg.norm(embeddings, axis=0, keepdims=True)


def compute_embedding(text: str, model_encoder: str = "all-MiniLM-L6-v2"):
    """
    Encode text using a transformer.

    :param name: The text to encode (str).
    :param model_encoder: The sentence transformer to use (default is 'all-MiniLM-L6-v2').
    :return: Text Embeddings (np.array).
    """

    # Preprocess text data (simple example, can be enhanced)
    def preprocess_text(text):
        return text.lower()

    # Load pre-trained Sentence-BERT model
    model = SentenceTransformer(model_encoder)

    processed_text = preprocess_text(text)
    text_embeddings = model.encode(processed_text, convert_to_tensor=True)
    # print(text_embeddings.cpu().detach().numpy().shape)
    text_embeddings = normalize_embeddings(text_embeddings.cpu().detach().numpy())

    return text_embeddings

# def compute_embedding(text: str, model_encoder: str = "all-MiniLM-L6-v2"):
#     """
#     Encode text using a transformer.

#     :param text: The text to encode (str).
#     :param model_encoder: The sentence transformer to use (default is 'all-MiniLM-L6-v2').
#     :return: Text Embeddings (np.array).
#     """

#     def load_model():
#         return SentenceTransformer(model_encoder)

#     # Load the model (this will only happen once and will be cached)
#     model = load_model()

#     # Preprocess text data (can be enhanced further)
#     def preprocess_text(text):
#         return text.lower()

#     processed_text = preprocess_text(text)

#     # Compute embeddings
#     text_embeddings = model.encode(processed_text, convert_to_tensor=True)

#     # Normalize embeddings
#     text_embeddings = normalize_embeddings(text_embeddings.cpu().detach().numpy())

#     return text_embeddings

# def normalize_embeddings(embeddings):
#     # Normalize embeddings
#     return embeddings / np.linalg.norm(embeddings, axis=0, keepdims=True)

