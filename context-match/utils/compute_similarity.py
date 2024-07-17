from sentence_transformers import util

def compute_similarity(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).squeeze().cpu().tolist()

def compute_similarity_matrix(embeddings1, embeddings2):
    similarity_scores = {}
    for idx1, emb1 in enumerate(embeddings1):
        for idx2, emb2 in enumerate(embeddings2):
            similarity_scores[(idx1, idx2)] = util.pytorch_cos_sim(emb1, emb2).squeeze().cpu().tolist()
    return similarity_scores