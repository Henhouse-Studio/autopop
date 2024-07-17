from sentence_transformers import SentenceTransformer

def compute_embedding(text: str, load_embeddings: bool = False):

    # Preprocess text data (simple example, can be enhanced)
    def preprocess_text(text):
        return text.lower()

    # Load pre-trained Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    processed_text = preprocess_text(text)
    text_embeddings = model.encode(processed_text, convert_to_tensor=True)

    return text_embeddings
