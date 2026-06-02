import torch
import transformers

import numpy as np
import pandas as pd

from other_code.config_delta_get import *
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration, GenerationConfig


def load_models(MODEL_PATH, MODEL_EMBEDDING_NAME):

    model = BigBirdPegasusForConditionalGeneration.from_pretrained(MODEL_PATH)

    model.generation_config.num_beams = 4
    model.generation_config.length_penalty = 0.8
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.max_length = 300
    model.generation_config.min_length = 100
    model.generation_config.early_stopping = True
    model.generation_config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, return_pt = True)
  


    model_embedding = SentenceTransformer(MODEL_EMBEDDING_NAME, trust_remote_code=True)
    model.to('cuda:0')

    model.eval()

    return tokenizer, model, model_embedding

def inference(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        # max_length=model.config.max_position_embeddings  # или tokenizer.model_max_length
    )
    
    inputs.to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(**inputs)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return decoded[0]


def main():

    cache_file = "embeddings_cache.npy"

    if os.path.exists(cache_file):
        print(f"Загрузка векторов из кэша: {cache_file}")
        embeddings = np.load(cache_file)
        print(f"Загружено {len(embeddings)} векторов размерности {embeddings.shape[1]}")
        return embeddings

    tokenizer, model, model_embedding = load_models(MODEL_PATH, MODEL_EMBEDDING_NAME)
    

    df_test = pd.read_csv(PATH_TEST_CSV_CLEF)


    df_test['claims_for_summarize'] = df_test['claims'].apply(lambda x: 'summarize: ' + x)
    df_test['claims_vector'] = df_test['claims_for_summarize'].apply(lambda x : inference(x, tokenizer, model))


    embeddings_claims = model_embedding.encode(df_test['claims'].tolist())
    embeddings_claims_summary = model_embedding.encode(df_test['claims_vector'].tolist())

    embeddings_all = embeddings_claims_summary - embeddings_claims


    delta_avg = np.mean(embeddings_all, axis=0)

    np.save(cache_file, delta_avg)


if __name__ == "__main__":
    main()