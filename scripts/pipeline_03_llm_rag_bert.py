"""
pipeline_03_llm_rag_bert.py
Pipeline 3 — LLM + RAG + BERT
Recupera trechos via RAG, re-ranqueia com BERT cross-encoder,
envia os melhores ao modelo como contexto.
"""

import time
import os
import pandas as pd
import requests
import chromadb
import numpy as np
import nltk
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
MODELOS        = ["llama3", "mistral", "gemma:2b"]
N_PERGUNTAS    = 10
TOP_K_RAG      = 10   # trechos recuperados pelo RAG
TOP_K_BERT     = 3    # trechos mantidos após filtragem BERT
OLLAMA_URL     = "http://localhost:11434/api/generate"
PASTA_CHROMA   = "./chroma_db"
COLECAO        = "bioasq_corpus"
OUTPUT_CSV     = "./outputs/resultados_pipeline3.csv"
SEED           = 42

# modelo BERT cross-encoder para re-ranking
BERT_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── 1. CARREGAR PERGUNTAS ───────────────────────────────────
print("Carregando perguntas BioASQ...")
qa = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
df = pd.DataFrame(qa["test"])
df_amostra = df.sample(N_PERGUNTAS, random_state=SEED).reset_index(drop=True)
print(f"  {N_PERGUNTAS} perguntas selecionadas")

# ─── 2. CARREGAR MODELOS ─────────────────────────────────────
print("Carregando modelo de embedding...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Pronto!")

print(f"Carregando modelo BERT cross-encoder: {BERT_MODEL}")
print("  (primeira vez faz download — aguarde)")
cross_encoder = CrossEncoder(BERT_MODEL)
print("  Pronto!")

# ─── 3. CONECTAR AO CHROMADB ─────────────────────────────────
print("Conectando ao ChromaDB...")
client  = chromadb.PersistentClient(path=PASTA_CHROMA)
colecao = client.get_collection(COLECAO)
print(f"  {colecao.count():,} trechos indexados")

# ─── 4. FUNÇÃO: RAG + BERT RERANKING ─────────────────────────
def recuperar_e_reranquear(pergunta, top_k_rag=TOP_K_RAG, top_k_bert=TOP_K_BERT):
    # passo 1 — RAG: recupera top_k_rag trechos por similaridade vetorial
    vetor = embedding_model.encode(pergunta).tolist()
    resultados = colecao.query(
        query_embeddings=[vetor],
        n_results=top_k_rag
    )
    trechos_candidatos = resultados["documents"][0]
    dist_rag = resultados["distances"][0]

    # passo 2 — BERT: re-ranqueia os candidatos por relevância
    pares = [(pergunta, t) for t in trechos_candidatos]
    scores_bert = cross_encoder.predict(pares)

    # ordena pelos scores BERT (maior = mais relevante)
    ranking = sorted(
        zip(trechos_candidatos, scores_bert, dist_rag),
        key=lambda x: x[1],
        reverse=True
    )

    # mantém só os top_k_bert melhores
    trechos_filtrados  = [r[0] for r in ranking[:top_k_bert]]
    scores_top         = [float(r[1]) for r in ranking[:top_k_bert]]
    dist_media_rag     = float(np.mean(dist_rag))
    score_medio_bert   = float(np.mean(scores_top))

    return trechos_filtrados, dist_media_rag, score_medio_bert

# ─── 5. FUNÇÃO: CHAMAR OLLAMA COM CONTEXTO FILTRADO ──────────
def chamar_ollama_bert(modelo, pergunta, trechos):
    contexto = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(trechos)])

    prompt = f"""You are a biomedical expert. Use the following highly relevant context passages to answer the question accurately and concisely.

Context:
{contexto}

Question: {pergunta}

Answer:"""

    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200}
    }

    inicio = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    tempo = time.time() - inicio

    data = response.json()
    resposta        = data.get("response", "").strip()
    tokens_prompt   = data.get("prompt_eval_count", 0)
    tokens_resposta = data.get("eval_count", 0)

    return resposta, tempo, tokens_prompt, tokens_resposta

# ─── 6. FUNÇÃO: CALCULAR MÉTRICAS ────────────────────────────
def calcular_metricas(resposta_gerada, ground_truth, emb_model):
    ref_tokens = nltk.word_tokenize(ground_truth.lower())
    hyp_tokens = nltk.word_tokenize(resposta_gerada.lower())
    meteor = meteor_score([ref_tokens], hyp_tokens)

    emb_gerada = emb_model.encode([resposta_gerada])
    emb_ref    = emb_model.encode([ground_truth])
    cosseno = cosine_similarity(emb_gerada, emb_ref)[0][0]

    return round(meteor, 4), round(float(cosseno), 4)

# ─── 7. RODAR OS PIPELINES ───────────────────────────────────
print(f"\nRodando Pipeline 3 — LLM + RAG + BERT")
print(f"  RAG recupera top {TOP_K_RAG}, BERT filtra para top {TOP_K_BERT}")
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
            # RAG + BERT reranking
            trechos, dist_rag, score_bert = recuperar_e_reranquear(pergunta)

            # chamar modelo com trechos filtrados
            resposta, tempo, tokens_p, tokens_r = chamar_ollama_bert(
                modelo, pergunta, trechos
            )

            meteor, cosseno = calcular_metricas(
                resposta, ground_truth, embedding_model
            )

            resultados.append({
                "pipeline":        "LLM_RAG_BERT",
                "modelo":          modelo,
                "pergunta_id":     row["id"],
                "pergunta":        pergunta,
                "ground_truth":    ground_truth,
                "resposta":        resposta,
                "trechos_usados":  " ||| ".join(trechos),
                "dist_media_rag":  round(dist_rag, 4),
                "score_medio_bert":round(score_bert, 4),
                "meteor":          meteor,
                "cosseno":         cosseno,
                "tempo_s":         round(tempo, 2),
                "tokens_prompt":   tokens_p,
                "tokens_resposta": tokens_r,
                "tokens_total":    tokens_p + tokens_r
            })

            print(f"    METEOR: {meteor:.4f} | Cosseno: {cosseno:.4f} | "
                  f"Tempo: {tempo:.1f}s | Tokens: {tokens_p + tokens_r} | "
                  f"BERT score: {score_bert:.4f}")

        except Exception as e:
            print(f"    ERRO: {e}")
            resultados.append({
                "pipeline": "LLM_RAG_BERT", "modelo": modelo,
                "pergunta_id": row["id"], "pergunta": pergunta,
                "ground_truth": ground_truth, "resposta": "ERRO",
                "trechos_usados": None, "dist_media_rag": None,
                "score_medio_bert": None, "meteor": None,
                "cosseno": None, "tempo_s": None,
                "tokens_prompt": None, "tokens_resposta": None,
                "tokens_total": None
            })

    print()

# ─── 8. SALVAR RESULTADOS ────────────────────────────────────
os.makedirs("./outputs", exist_ok=True)
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(OUTPUT_CSV, index=False)
print(f"Resultados salvos em: {OUTPUT_CSV}")

# ─── 9. RESUMO ───────────────────────────────────────────────
print("\n─── Resumo por modelo ───")
resumo = df_resultados.groupby("modelo")[
    ["meteor","cosseno","tempo_s","tokens_total","score_medio_bert"]
].mean().round(4)
print(resumo.to_string())