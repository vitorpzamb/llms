"""
pipeline_completo.py
Script unificado — 4 pipelines x 3 modelos x todas as perguntas BioASQ
- 3 modelos rodam em paralelo dentro de cada pipeline
- Retomada automática por pipeline/modelo/pergunta
- Salva a cada resposta (CSV incremental)
- Métricas: METEOR + similaridade de cosseno
- Registra tempo total e por pipeline
"""

import time
import os
import csv
import threading
import numpy as np
import pandas as pd
import requests
import chromadb
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
MODELOS        = ["llama3", "mistral", "gemma:2b"]
OLLAMA_URL     = "http://localhost:11434/api/generate"
PASTA_CHROMA   = "./chroma_db"
COLECAO        = "bioasq_corpus"
CSV_CLUSTERS   = "./outputs/clusters_resultado.csv"
PASTA_OUTPUTS  = "./outputs"
ARQ_CENTROIDES = "./outputs/centroides.npy"
ARQ_TEMPO      = "./outputs/tempo_execucao.txt"

TOP_K_RAG      = 5
TOP_K_BERT_RAG = 10
TOP_K_BERT_FIN = 3
BERT_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEED           = 42
TIMEOUT_OLLAMA = 300
MAX_WORKERS    = 3

os.makedirs(PASTA_OUTPUTS, exist_ok=True)

PIPELINES = {
    "P1_LLM_BASE":     f"{PASTA_OUTPUTS}/resultados_P1_LLM_BASE.csv",
    "P2_LLM_RAG":      f"{PASTA_OUTPUTS}/resultados_P2_LLM_RAG.csv",
    "P3_LLM_RAG_BERT": f"{PASTA_OUTPUTS}/resultados_P3_LLM_RAG_BERT.csv",
    "P4_LLM_RAG_CLUS": f"{PASTA_OUTPUTS}/resultados_P4_LLM_RAG_CLUS.csv",
}

COLUNAS_BASE = [
    "pipeline","modelo","pergunta_id","pergunta","ground_truth",
    "resposta","meteor","cosseno",
    "tempo_s","tokens_prompt","tokens_resposta","tokens_total"
]
COLUNAS_RAG     = COLUNAS_BASE + ["trechos_usados","dist_media_rag"]
COLUNAS_BERT    = COLUNAS_RAG  + ["score_medio_bert"]
COLUNAS_CLUSTER = COLUNAS_RAG  + ["cluster_escolhido","sim_cluster"]

csv_locks = {p: threading.Lock() for p in PIPELINES}

# ─── 1. CARREGAR RECURSOS ────────────────────────────────────
print("=" * 60)
print("CARREGANDO RECURSOS")
print("=" * 60)

print("\n[1/6] Perguntas BioASQ...")
qa    = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
df_qa = pd.concat([
    pd.DataFrame(qa["train"]),
    pd.DataFrame(qa["test"])
], ignore_index=True)
print(f"  {len(df_qa):,} perguntas carregadas")

print("\n[2/6] Modelo de embedding...")
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Pronto!")

print("\n[3/6] Modelo BERT cross-encoder...")
cross_encoder = CrossEncoder(BERT_MODEL)
print("  Pronto!")

print("\n[4/6] ChromaDB...")
client  = chromadb.PersistentClient(path=PASTA_CHROMA)
colecao = client.get_collection(COLECAO)
print(f"  {colecao.count():,} trechos indexados")

print("\n[5/6] Clusters...")
df_clusters = pd.read_csv(CSV_CLUSTERS)
df_clusters["id"] = df_clusters["id"].astype(str)
ids_por_cluster = df_clusters.groupby("cluster")["id"].apply(list).to_dict()
n_clusters      = df_clusters["cluster"].nunique()
print(f"  {n_clusters} clusters | {len(df_clusters):,} trechos")

print("\n[6/6] Centróides...")
if os.path.exists(ARQ_CENTROIDES):
    arr        = np.load(ARQ_CENTROIDES)
    centroides = {k: arr[k] for k in range(n_clusters)}
    print(f"  Carregados do disco ({ARQ_CENTROIDES})")
else:
    print("  Calculando centróides pela primeira vez...")
    centroides = {}
    for k in sorted(ids_por_cluster.keys()):
        ids_k = ids_por_cluster[k]
        embs  = []
        for i in range(0, len(ids_k), 1000):
            res = colecao.get(ids=ids_k[i:i+1000], include=["embeddings"])
            embs.extend(res["embeddings"])
        centroides[k] = np.mean(embs, axis=0)
        print(f"  Cluster {k}: {len(ids_k):,} trechos")
    arr = np.array([centroides[k] for k in sorted(centroides.keys())])
    np.save(ARQ_CENTROIDES, arr)
    print(f"  Salvos em {ARQ_CENTROIDES}")

# ─── 2. FUNÇÕES AUXILIARES ───────────────────────────────────

def chamar_ollama(modelo, prompt):
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200}
    }
    t0 = time.time()
    r  = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_OLLAMA)
    tempo = time.time() - t0
    d = r.json()
    return (
        d.get("response", "").strip(),
        tempo,
        d.get("prompt_eval_count", 0),
        d.get("eval_count", 0)
    )

def metricas_texto(resposta, ground_truth):
    ref = nltk.word_tokenize(ground_truth.lower())
    hyp = nltk.word_tokenize(resposta.lower())
    m   = meteor_score([ref], hyp)
    eg  = emb_model.encode([resposta])
    er  = emb_model.encode([ground_truth])
    c   = cosine_similarity(eg, er)[0][0]
    return round(m, 4), round(float(c), 4)

def rag_busca(pergunta, top_k=TOP_K_RAG):
    vetor = emb_model.encode(pergunta).tolist()
    res   = colecao.query(query_embeddings=[vetor], n_results=top_k)
    return res["documents"][0], float(np.mean(res["distances"][0]))

def bert_rerank(pergunta, candidatos, top_k=TOP_K_BERT_FIN):
    pares  = [(pergunta, t) for t in candidatos]
    scores = cross_encoder.predict(pares)
    ranked = sorted(zip(candidatos, scores), key=lambda x: x[1], reverse=True)
    top       = ranked[:top_k]
    trechos_f = [r[0] for r in top]
    score_med = float(np.mean([r[1] for r in top]))
    return trechos_f, round(score_med, 4)

def rag_cluster(pergunta, top_k=TOP_K_RAG):
    vetor = emb_model.encode(pergunta)
    sims  = {k: cosine_similarity([vetor], [centroides[k]])[0][0] for k in centroides}
    cl    = max(sims, key=sims.get)
    sim   = round(float(sims[cl]), 4)
    ids_cl = ids_por_cluster[cl]
    res = colecao.query(
        query_embeddings=[vetor.tolist()],
        n_results=min(500, len(ids_cl))
    )
    cl_set  = set(ids_cl)
    trechos = []
    dists   = []
    for doc, id_, d in zip(res["documents"][0], res["ids"][0], res["distances"][0]):
        if id_ in cl_set:
            trechos.append(doc)
            dists.append(d)
        if len(trechos) == top_k:
            break
    dist_med = round(float(np.mean(dists)), 4) if dists else 1.0
    return trechos, dist_med, cl, sim

def prompt_base(pergunta):
    return f"""You are a biomedical expert. Answer the following question concisely and accurately based on your knowledge.

Question: {pergunta}

Answer:"""

def prompt_contexto(pergunta, trechos):
    ctx = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(trechos)])
    return f"""You are a biomedical expert. Use the following context passages to answer the question accurately and concisely.

Context:
{ctx}

Question: {pergunta}

Answer:"""

def ids_ja_processados(csv_path, modelo):
    if not os.path.exists(csv_path):
        return set()
    try:
        df  = pd.read_csv(csv_path)
        sub = df[df["modelo"] == modelo]
        return set(sub["pergunta_id"].astype(str).tolist())
    except Exception:
        return set()

def append_csv(csv_path, row, colunas, lock):
    with lock:
        novo = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=colunas)
            if novo:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in colunas})

# ─── 3. FUNÇÕES POR PIPELINE ─────────────────────────────────

def rodar_p1(modelo, df_perguntas):
    csv_path  = PIPELINES["P1_LLM_BASE"]
    lock      = csv_locks["P1_LLM_BASE"]
    ja_feitos = ids_ja_processados(csv_path, modelo)
    pendentes = df_perguntas[~df_perguntas["id"].astype(str).isin(ja_feitos)]
    print(f"  [P1][{modelo}] {len(pendentes):,} pendentes")
    for _, row in pendentes.iterrows():
        try:
            resposta, t, tp, tr = chamar_ollama(modelo, prompt_base(row["question"]))
            m, c = metricas_texto(resposta, row["answer"])
            append_csv(csv_path, {
                "pipeline":"P1_LLM_BASE","modelo":modelo,
                "pergunta_id":row["id"],"pergunta":row["question"],
                "ground_truth":row["answer"],"resposta":resposta,
                "meteor":m,"cosseno":c,
                "tempo_s":round(t,2),"tokens_prompt":tp,
                "tokens_resposta":tr,"tokens_total":tp+tr
            }, COLUNAS_BASE, lock)
            print(f"    [P1][{modelo}] {row['id']} | {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"    [P1][{modelo}] ERRO {row['id']}: {e}")

def rodar_p2(modelo, df_perguntas):
    csv_path  = PIPELINES["P2_LLM_RAG"]
    lock      = csv_locks["P2_LLM_RAG"]
    ja_feitos = ids_ja_processados(csv_path, modelo)
    pendentes = df_perguntas[~df_perguntas["id"].astype(str).isin(ja_feitos)]
    print(f"  [P2][{modelo}] {len(pendentes):,} pendentes")
    for _, row in pendentes.iterrows():
        try:
            trechos, dist = rag_busca(row["question"])
            resposta, t, tp, tr = chamar_ollama(modelo, prompt_contexto(row["question"], trechos))
            m, c = metricas_texto(resposta, row["answer"])
            append_csv(csv_path, {
                "pipeline":"P2_LLM_RAG","modelo":modelo,
                "pergunta_id":row["id"],"pergunta":row["question"],
                "ground_truth":row["answer"],"resposta":resposta,
                "meteor":m,"cosseno":c,
                "tempo_s":round(t,2),"tokens_prompt":tp,
                "tokens_resposta":tr,"tokens_total":tp+tr,
                "trechos_usados":" ||| ".join(trechos),
                "dist_media_rag":round(dist,4)
            }, COLUNAS_RAG, lock)
            print(f"    [P2][{modelo}] {row['id']} | {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"    [P2][{modelo}] ERRO {row['id']}: {e}")

def rodar_p3(modelo, df_perguntas):
    csv_path  = PIPELINES["P3_LLM_RAG_BERT"]
    lock      = csv_locks["P3_LLM_RAG_BERT"]
    ja_feitos = ids_ja_processados(csv_path, modelo)
    pendentes = df_perguntas[~df_perguntas["id"].astype(str).isin(ja_feitos)]
    print(f"  [P3][{modelo}] {len(pendentes):,} pendentes")
    for _, row in pendentes.iterrows():
        try:
            candidatos, _  = rag_busca(row["question"], top_k=TOP_K_BERT_RAG)
            trechos, score = bert_rerank(row["question"], candidatos)
            resposta, t, tp, tr = chamar_ollama(modelo, prompt_contexto(row["question"], trechos))
            m, c = metricas_texto(resposta, row["answer"])
            append_csv(csv_path, {
                "pipeline":"P3_LLM_RAG_BERT","modelo":modelo,
                "pergunta_id":row["id"],"pergunta":row["question"],
                "ground_truth":row["answer"],"resposta":resposta,
                "meteor":m,"cosseno":c,
                "tempo_s":round(t,2),"tokens_prompt":tp,
                "tokens_resposta":tr,"tokens_total":tp+tr,
                "trechos_usados":" ||| ".join(trechos),
                "dist_media_rag":round(_, 4) if isinstance(_, float) else 0,
                "score_medio_bert":score
            }, COLUNAS_BERT, lock)
            print(f"    [P3][{modelo}] {row['id']} | {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"    [P3][{modelo}] ERRO {row['id']}: {e}")

def rodar_p4(modelo, df_perguntas):
    csv_path  = PIPELINES["P4_LLM_RAG_CLUS"]
    lock      = csv_locks["P4_LLM_RAG_CLUS"]
    ja_feitos = ids_ja_processados(csv_path, modelo)
    pendentes = df_perguntas[~df_perguntas["id"].astype(str).isin(ja_feitos)]
    print(f"  [P4][{modelo}] {len(pendentes):,} pendentes")
    for _, row in pendentes.iterrows():
        try:
            trechos, dist, cl, sim = rag_cluster(row["question"])
            if not trechos:
                continue
            resposta, t, tp, tr = chamar_ollama(modelo, prompt_contexto(row["question"], trechos))
            m, c = metricas_texto(resposta, row["answer"])
            append_csv(csv_path, {
                "pipeline":"P4_LLM_RAG_CLUS","modelo":modelo,
                "pergunta_id":row["id"],"pergunta":row["question"],
                "ground_truth":row["answer"],"resposta":resposta,
                "meteor":m,"cosseno":c,
                "tempo_s":round(t,2),"tokens_prompt":tp,
                "tokens_resposta":tr,"tokens_total":tp+tr,
                "trechos_usados":" ||| ".join(trechos),
                "dist_media_rag":dist,
                "cluster_escolhido":cl,
                "sim_cluster":sim
            }, COLUNAS_CLUSTER, lock)
            print(f"    [P4][{modelo}] {row['id']} | {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"    [P4][{modelo}] ERRO {row['id']}: {e}")

FUNCS_PIPELINE = {
    "P1_LLM_BASE":     rodar_p1,
    "P2_LLM_RAG":      rodar_p2,
    "P3_LLM_RAG_BERT": rodar_p3,
    "P4_LLM_RAG_CLUS": rodar_p4,
}

# ─── 4. EXECUÇÃO PRINCIPAL ───────────────────────────────────
print("\n" + "=" * 60)
print("INICIANDO EXPERIMENTOS")
print(f"  Perguntas: {len(df_qa):,}")
print(f"  Pipelines: {list(FUNCS_PIPELINE.keys())}")
print(f"  Modelos em paralelo: {MODELOS}")
print("=" * 60)

tempo_inicio_total = time.time()
tempos_pipeline    = {}

for nome_pipeline, func in FUNCS_PIPELINE.items():
    print(f"\n{'='*60}")
    print(f"PIPELINE: {nome_pipeline}")
    print(f"{'='*60}")
    t0_pipe = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(func, modelo, df_qa): modelo
            for modelo in MODELOS
        }
        for future in as_completed(futures):
            modelo = futures[future]
            try:
                future.result()
                print(f"  [OK] {modelo} concluído")
            except Exception as e:
                print(f"  [ERRO] {modelo}: {e}")

    t_pipe = time.time() - t0_pipe
    tempos_pipeline[nome_pipeline] = t_pipe
    print(f"\n  {nome_pipeline} concluído em {t_pipe/3600:.1f}h ({t_pipe/60:.0f} min)")

# ─── 5. RESUMO FINAL ─────────────────────────────────────────
tempo_total = time.time() - tempo_inicio_total

linhas = []
linhas.append("=" * 60)
linhas.append("RESUMO DE TEMPO")
linhas.append("=" * 60)
for pipe, t in tempos_pipeline.items():
    linhas.append(f"  {pipe}: {t/3600:.2f}h ({t/60:.0f} min)")
linhas.append(f"\n  TOTAL: {tempo_total/3600:.2f}h ({tempo_total/60:.0f} min)")
linhas.append("=" * 60)
linhas.append("\nMÉTRICAS MÉDIAS")
for nome, csv_path in PIPELINES.items():
    if os.path.exists(csv_path):
        df_res = pd.read_csv(csv_path)
        linhas.append(f"\n{nome}:")
        g = df_res.groupby("modelo")[["meteor","cosseno","tokens_total","tempo_s"]].mean().round(4)
        linhas.append(g.to_string())

texto = "\n".join(linhas)
print("\n" + texto)
with open(ARQ_TEMPO, "w", encoding="utf-8") as f:
    f.write(texto)
print(f"\nResumo salvo em: {ARQ_TEMPO}")
print("Experimento concluído!")