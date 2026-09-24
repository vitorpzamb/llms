"""
pipeline_01_llm_base.py
Pipeline 1 — LLM sem RAG (baseline)
Envia perguntas direto ao modelo sem contexto externo.
Salva respostas e métricas em CSV.
"""

import time
import json
import pandas as pd
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
MODELOS       = ["llama3", "mistral", "gemma:2b"]
N_PERGUNTAS   = 10
OLLAMA_URL    = "http://localhost:11434/api/generate"
OUTPUT_CSV    = "./outputs/resultados_pipeline1.csv"
SEED          = 42

# ─── 1. CARREGAR PERGUNTAS ───────────────────────────────────
print("Carregando perguntas BioASQ...")
qa = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
df = pd.DataFrame(qa["test"])
df_amostra = df.sample(N_PERGUNTAS, random_state=SEED).reset_index(drop=True)
print(f"  {N_PERGUNTAS} perguntas selecionadas")

# ─── 2. CARREGAR MODELO DE EMBEDDING (para métrica de cosseno) ──
print("Carregando modelo de embedding...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Pronto!")

# ─── 3. FUNÇÃO: CHAMAR OLLAMA ────────────────────────────────
def chamar_ollama(modelo, pergunta):
    prompt = f"""You are a biomedical expert. Answer the following question concisely and accurately based on your knowledge.

Question: {pergunta}

Answer:"""

    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200}
    }

    inicio = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    tempo = time.time() - inicio

    data = response.json()
    resposta = data.get("response", "").strip()
    tokens_prompt = data.get("prompt_eval_count", 0)
    tokens_resposta = data.get("eval_count", 0)

    return resposta, tempo, tokens_prompt, tokens_resposta

# ─── 4. FUNÇÃO: CALCULAR MÉTRICAS ────────────────────────────
def calcular_metricas(resposta_gerada, ground_truth, emb_model):
    # METEOR
    ref_tokens = nltk.word_tokenize(ground_truth.lower())
    hyp_tokens = nltk.word_tokenize(resposta_gerada.lower())
    meteor = meteor_score([ref_tokens], hyp_tokens)

    # Similaridade de cosseno
    emb_gerada = emb_model.encode([resposta_gerada])
    emb_ref    = emb_model.encode([ground_truth])
    cosseno = cosine_similarity(emb_gerada, emb_ref)[0][0]

    return round(meteor, 4), round(float(cosseno), 4)

# ─── 5. RODAR OS PIPELINES ───────────────────────────────────
print(f"\nRodando Pipeline 1 — LLM sem RAG")
print(f"  Modelos: {MODELOS}")
print(f"  Perguntas: {N_PERGUNTAS}\n")

resultados = []

for modelo in MODELOS:
    print(f"─── Modelo: {modelo} ───")

    for i, row in df_amostra.iterrows():
        pergunta     = row["question"]
        ground_truth = row["answer"]

        print(f"  [{i+1}/{N_PERGUNTAS}] {pergunta[:60]}...")

        try:
            resposta, tempo, tokens_p, tokens_r = chamar_ollama(modelo, pergunta)
            meteor, cosseno = calcular_metricas(resposta, ground_truth, embedding_model)

            resultados.append({
                "pipeline":       "LLM_base",
                "modelo":         modelo,
                "pergunta_id":    row["id"],
                "pergunta":       pergunta,
                "ground_truth":   ground_truth,
                "resposta":       resposta,
                "meteor":         meteor,
                "cosseno":        cosseno,
                "tempo_s":        round(tempo, 2),
                "tokens_prompt":  tokens_p,
                "tokens_resposta":tokens_r,
                "tokens_total":   tokens_p + tokens_r
            })

            print(f"    METEOR: {meteor:.4f} | Cosseno: {cosseno:.4f} | "
                  f"Tempo: {tempo:.1f}s | Tokens: {tokens_p + tokens_r}")

        except Exception as e:
            print(f"    ERRO: {e}")
            resultados.append({
                "pipeline": "LLM_base", "modelo": modelo,
                "pergunta_id": row["id"], "pergunta": pergunta,
                "ground_truth": ground_truth, "resposta": "ERRO",
                "meteor": None, "cosseno": None,
                "tempo_s": None, "tokens_prompt": None,
                "tokens_resposta": None, "tokens_total": None
            })

    print()

# ─── 6. SALVAR RESULTADOS ────────────────────────────────────
import os
os.makedirs("./outputs", exist_ok=True)

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(OUTPUT_CSV, index=False)
print(f"Resultados salvos em: {OUTPUT_CSV}")

# ─── 7. RESUMO ───────────────────────────────────────────────
print("\n─── Resumo por modelo ───")
resumo = df_resultados.groupby("modelo")[["meteor","cosseno","tempo_s","tokens_total"]].mean().round(4)
print(resumo.to_string())