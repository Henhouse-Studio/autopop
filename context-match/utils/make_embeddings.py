from transformers import BertTokenizer, BertModel
import nltk

def compute_embeddings(prompt: str, df_columns: list, load_embeddings: bool = False):

    # Path to your nltk_data
    nltk_data_path = '~/nltk_data/'
    nltk.data.path.append(nltk_data_path)

    # Ensure punkt tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print(f"Downloading punkt to {nltk_data_path}")
        nltk.download('punkt', download_dir=nltk_data_path)
        
    def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join(tokens)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    prompt_embedding = get_bert_embedding(preprocess_text(prompt))
    field_embeddings = [get_bert_embedding(preprocess_text(field)) for field in df_columns]

    return prompt_embedding, field_embeddings