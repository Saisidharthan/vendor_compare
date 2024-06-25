import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import TapexTokenizer, BartForConditionalGeneration

# Directory where models are stored
model_dir = "model_files"

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retriever_path = os.path.join(model_dir, "sentence_transformer")
    if not os.path.exists(retriever_path):
        retriever = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
        retriever.save(retriever_path)
    else:
        retriever = SentenceTransformer(retriever_path)
    
    tokenizer_path = os.path.join(model_dir, "tapex_tokenizer")
    model_path = os.path.join(model_dir, "tapex_model")
    if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
        tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-large')
        model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-large')
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
    else:
        tokenizer = TapexTokenizer.from_pretrained(tokenizer_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    
    model.to(device)
    return retriever, tokenizer, model
