from transformers import BertJapaneseTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 加载日文嵌入模型
tokenizer_bert = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

def get_embeddings(text):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def calculate_similarity(answer, reference):
    answer_emb = get_embeddings(answer)
    reference_emb = get_embeddings(reference)
    similarity = cosine_similarity(answer_emb, reference_emb)
    return similarity[0][0]
