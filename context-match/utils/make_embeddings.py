from transformers import BertTokenizer, BertModel
import nltk


def compute_embedding(text: str, load_embeddings: bool = False):

    # Path to your nltk_data
    nltk_data_path = "~/nltk_data/"
    nltk.data.path.append(nltk_data_path)

    # Ensure punkt tokenizer is available
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        # print(f"Downloading punkt to {nltk_data_path}")
        nltk.download("punkt", download_dir=nltk_data_path, quiet=True)

    def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        return " ".join(tokens)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    text_embeddings = get_bert_embedding(preprocess_text(text))

    return text_embeddings
