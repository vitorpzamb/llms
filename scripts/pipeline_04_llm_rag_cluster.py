"""
pipeline_04_llm_rag_cluster.py
Pipeline 4 — LLM + RAG + Clustering
Identifica o cluster mais próximo da pergunta e restringe
a busca RAG aos trechos desse cluster antes de enviar ao LLM.
Centróides são salvos em disco e reutilizados nas próximas execuções.
"""

import time
import os
import numpy as np
import pandas as pd
import requests
import chromadb
import nltk
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
MODELOS            = ["llama3", "mistral", "gemma:2b"]
N_PERGUNTAS        = 10
TOP_K              = 5
OLLAMA_URL         = "http://localhost:11434/api/generate"
PASTA_CHROMA       = "./chroma_db"
COLECAO            = "bioasq_corpus"
CSV_CLUSTERS       = "./outputs/clusters_resultado.csv"
OUTPUT_CSV         = "./outputs/resultados_pipeline4.csv"
ARQUIVO_CENTROIDES = "./outputs/centroides.npy"
SEED               = 42

# ─── 1. CARREGAR PERGUNTAS ───────────────────────────────────
print("Carregando perguntas BioASQ...")
qa = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
df = pd.DataFrame(qa["test"])
df_amostra = df.sample(N_PERGUNTAS, random_state=SEED).reset_index(drop=True)
print(f"  {N_PERGUNTAS} perguntas selecionadas")

# ─── 2. CARREGAR MODELO DE EMBEDDING ─────────────────────────
print("Carregando modelo de embedding...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Pronto!")

# ─── 3. CONECTAR AO CHROMADB ─────────────────────────────────
print("Conectando ao ChromaDB...")
client  = chromadb.PersistentClient(path=PASTA_CHROMA)
colecao = client.get_collection(COLECAO)
print(f"  {colecao.count():,} trechos indexados")

# ─── 4. CARREGAR CLUSTERS ────────────────────────────────────
print(f"Carregando clusters de: {CSV_CLUSTERS}")
df_clusters = pd.read_csv(CSV_CLUSTERS)
df_clusters["id"] = df_clusters["id"].astype(str)

id_to_cluster  = dict(zip(df_clusters["id"], df_clusters["cluster"]))
ids_por_cluster = df_clusters.groupby("cluster")["id"].apply(list).to_dict()
n_clusters     = df_clusters["cluster"].nunique()
print(f"  {n_clusters} clusters | {len(df_clusters):,} trechos mapeados")

# ─── 5. CENTRÓIDES — CARREGAR OU CALCULAR ────────────────────
os.makedirs("./outputs", exist_ok=True)

if os.path.exists(ARQUIVO_CENTROIDES):
    print(f"\nCentróides encontrados em disco: {ARQUIVO_CENTROIDES}")
    centroides_array = np.load(ARQUIVO_CENTROIDES)
    centroides = {k: centroides_array[k] for k in range(n_clusters)}
    print(f"  {n_clusters} centróides carregados!")
else:
    print(f"\nCentróides não encontrados — calculando pela primeira vez...")
    centroides = {}

    for k in sorted(ids_por_cluster.keys()):
        ids_cluster = ids_por_cluster[k]
        BATCH = 1000
        embeddings_cluster = []

        for i in range(0, len(ids_cluster), BATCH):
            batch_ids = ids_cluster[i:i+BATCH]
            resultado = colecao.get(ids=batch_ids, include=["embeddings"])
            embeddings_cluster.extend(resultado["embeddings"])

        centroides[k] = np.mean(embeddings_cluster, axis=0)
        print(f"  Cluster {k}: {len(ids_cluster):,} trechos → centróide calculado")

    # salvar em disco
    centroides_array = np.array([centroides[k] for k in sorted(centroides.keys())])
    np.save(ARQUIVO_CENTROIDES, centroides_array)
    print(f"\n  Centróides salvos em: {ARQUIVO_CENTROIDES}")

# ─── 6. FUNÇÃO: IDENTIFICAR CLUSTER DA PERGUNTA ──────────────
def identificar_cluster(pergunta):
    vetor = embedding_model.encode(pergunta)
    sims = {
        k: cosine_similarity([vetor], [centroides[k]])[0][0]
        for k in centroides
    }
    cluster_escolhido = max(sims, key=sims.get)
    sim_max = sims[cluster_escolhido]
    return cluster_escolhido, round(float(sim_max), 4)

# ─── 7. FUNÇÃO: RAG RESTRITO AO CLUSTER ──────────────────────
def recuperar_no_cluster(pergunta, cluster_id, top_k=TOP_K):
    ids_cluster = ids_por_cluster[cluster_id]
    vetor = embedding_model.encode(pergunta).tolist()

    resultado_geral = colecao.query(
        query_embeddings=[vetor],
        n_results=min(500, len(ids_cluster))
    )

    cluster_set = set(ids_cluster)
    trechos_filtrados  = []
    distancias_filtradas = []

    for doc, id_, dist in zip(
        resultado_geral["documents"][0],
        resultado_geral["ids"][0],
        resultado_geral["distances"][0]
    ):
        if id_ in cluster_set:
            trechos_filtrados.append(doc)
            distancias_filtradas.append(dist)
        if len(trechos_filtrados) == top_k:
            break

    dist_media = round(float(np.mean(distancias_filtradas)), 4) if distancias_filtradas else 1.0
    return trechos_filtrados, dist_media

# ─── 8. FUNÇÃO: CHAMAR OLLAMA ────────────────────────────────
def chamar_ollama(modelo, pergunta, trechos):
    contexto = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(trechos)])

    prompt = f"""You are a biomedical expert. Use the following context passages to answer the question accurately and concisely.

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

# ─── 9. FUNÇÃO: CALCULAR MÉTRICAS ────────────────────────────
def calcular_metricas(resposta_gerada, ground_truth):
    ref_tokens = nltk.word_tokenize(ground_truth.lower())
    hyp_tokens = nltk.word_tokenize(resposta_gerada.lower())
    meteor = meteor_score([ref_tokens], hyp_tokens)

    emb_gerada = embedding_model.encode([resposta_gerada])
    emb_ref    = embedding_model.encode([ground_truth])
    cosseno    = cosine_similarity(emb_gerada, emb_ref)[0][0]

    return round(meteor, 4), round(float(cosseno), 4)

# ─── 10. RODAR OS PIPELINES ──────────────────────────────────
print(f"\nRodando Pipeline 4 — LLM + RAG + Clustering")
print(f"  Top-K dentro do cluster: {TOP_K}")
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
            cluster_id, sim_cluster = identificar_cluster(pergunta)
            trechos, dist_rag       = recuperar_no_cluster(pergunta, cluster_id)

            if not trechos:
                print(f"    AVISO: nenhum trecho no cluster {cluster_id}, pulando.")
                continue

            resposta, tempo, tokens_p, tokens_r = chamar_ollama(
                modelo, pergunta, trechos
            )
            meteor, cosseno = calcular_metricas(resposta, ground_truth)

            resultados.append({
                "pipeline":          "LLM_RAG_CLUSTER",
                "modelo":            modelo,
                "pergunta_id":       row["id"],
                "pergunta":          pergunta,
                "ground_truth":      ground_truth,
                "resposta":          resposta,
                "cluster_escolhido": cluster_id,
                "sim_cluster":       sim_cluster,
                "trechos_usados":    " ||| ".join(trechos),
                "dist_media_rag":    dist_rag,
                "meteor":            meteor,
                "cosseno":           cosseno,
                "tempo_s":           round(tempo, 2),
                "tokens_prompt":     tokens_p,
                "tokens_resposta":   tokens_r,
                "tokens_total":      tokens_p + tokens_r
            })

            print(f"    Cluster: {cluster_id} (sim={sim_cluster:.4f}) | "
                  f"METEOR: {meteor:.4f} | Cosseno: {cosseno:.4f} | "
                  f"Tempo: {tempo:.1f}s | Tokens: {tokens_p + tokens_r}")

        except Exception as e:
            print(f"    ERRO: {e}")
            resultados.append({
                "pipeline": "LLM_RAG_CLUSTER", "modelo": modelo,
                "pergunta_id": row["id"], "pergunta": pergunta,
                "ground_truth": ground_truth, "resposta": "ERRO",
                "cluster_escolhido": None, "sim_cluster": None,
                "trechos_usados": None, "dist_media_rag": None,
                "meteor": None, "cosseno": None, "tempo_s": None,
                "tokens_prompt": None, "tokens_resposta": None,
                "tokens_total": None
            })

    print()

# ─── 11. SALVAR RESULTADOS ────────────────────────────────────
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(OUTPUT_CSV, index=False)
print(f"Resultados salvos em: {OUTPUT_CSV}")

# ─── 12. RESUMO ───────────────────────────────────────────────
print("\n─── Resumo por modelo ───")
resumo = df_resultados.groupby("modelo")[
    ["meteor", "cosseno", "tempo_s", "tokens_total"]
].mean().round(4)
print(resumo.to_string())

print("\n─── Clusters mais utilizados ───")
print(df_resultados["cluster_escolhido"].value_counts().sort_index().to_string())