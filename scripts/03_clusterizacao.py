"""
03_clusterizacao.py
Clusterização exploratória dos embeddings do corpus BioASQ.
- Extrai embeddings do ChromaDB
- Aplica elbow method para escolher número de clusters
- Clusteriza com K-Means
- Visualiza com UMAP em 2D
- Salva resultados em CSV para análise no R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
import chromadb
import umap
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
PASTA_CHROMA  = "./chroma_db"
COLECAO       = "bioasq_corpus"
AMOSTRA       = 5000   # usa amostra para UMAP e elbow (mais rápido)
K_VALORES     = [5, 8, 10, 15, 20, 25, 30]  # valores de K a testar
K_FINAL       = None   # será definido após ver o gráfico do cotovelo
SEED          = 42

# ─── 1. CARREGAR EMBEDDINGS DO CHROMADB ──────────────────────
print("Carregando embeddings do ChromaDB...")
client  = chromadb.PersistentClient(path=PASTA_CHROMA)
colecao = client.get_collection(COLECAO)

total = colecao.count()
print(f"  {total:,} trechos no índice")

# busca todos os embeddings em batches
print("  Extraindo embeddings (pode demorar ~1 min)...")
todos_embeddings = []
todos_ids        = []
todos_textos     = []
BATCH            = 5000

for offset in range(0, total, BATCH):
    resultado = colecao.get(
        limit=BATCH,
        offset=offset,
        include=["embeddings", "documents"]
    )
    todos_embeddings.extend(resultado["embeddings"])
    todos_ids.extend(resultado["ids"])
    todos_textos.extend(resultado["documents"])
    print(f"    {min(offset + BATCH, total):,} / {total:,} extraídos")

embeddings = np.array(todos_embeddings)
embeddings = normalize(embeddings)  # normaliza para cosseno
print(f"  Embeddings extraídos: {embeddings.shape}")

# ─── 2. AMOSTRA PARA ELBOW E UMAP ────────────────────────────
print(f"\nUsando amostra de {AMOSTRA:,} trechos para análise...")
np.random.seed(SEED)
idx_amostra  = np.random.choice(len(embeddings), AMOSTRA, replace=False)
emb_amostra  = embeddings[idx_amostra]
txt_amostra  = [todos_textos[i] for i in idx_amostra]

# ─── 3. ELBOW METHOD ─────────────────────────────────────────
print("\nCalculando elbow method...")
inercias = []

for k in K_VALORES:
    km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=3)
    km.fit(emb_amostra)
    inercias.append(km.inertia_)
    print(f"  K={k:2d} — inércia: {km.inertia_:.2f}")

# gráfico do cotovelo
plt.figure(figsize=(8, 4))
plt.plot(K_VALORES, inercias, "o-", color="#1a5cff", linewidth=2, markersize=7)
plt.xlabel("Número de clusters (K)", fontsize=12)
plt.ylabel("Inércia", fontsize=12)
plt.title("Elbow Method — escolha do número de clusters", fontsize=13)
plt.xticks(K_VALORES)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150)
plt.show()
print("\nGráfico salvo em: elbow_method.png")
print("Analise o gráfico e escolha o K onde a curva 'dobra' (cotovelo).")

# ─── 4. PEDIR K AO USUÁRIO ───────────────────────────────────
while True:
    try:
        K_FINAL = int(input("\nDigite o número de clusters escolhido: "))
        if K_FINAL in range(2, 51):
            break
        print("Digite um valor entre 2 e 50.")
    except ValueError:
        print("Digite um número inteiro.")

# ─── 5. CLUSTERIZAÇÃO FINAL ──────────────────────────────────
print(f"\nClusterizando com K={K_FINAL} no corpus completo...")
km_final = MiniBatchKMeans(n_clusters=K_FINAL, random_state=SEED, n_init=5)
clusters = km_final.fit_predict(embeddings)
print("  Clusterização concluída!")

# contagem por cluster
contagem = pd.Series(clusters).value_counts().sort_index()
print("\nTrechos por cluster:")
for k, n in contagem.items():
    print(f"  Cluster {k:2d}: {n:,} trechos")

# ─── 6. UMAP ─────────────────────────────────────────────────
print(f"\nReduzindo para 2D com UMAP (amostra de {AMOSTRA:,})...")
print("  Isso pode demorar alguns minutos...")

reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
emb_2d  = reducer.fit_transform(emb_amostra)

clusters_amostra = km_final.predict(emb_amostra)

# gráfico UMAP
cores = plt.cm.tab20(np.linspace(0, 1, K_FINAL))

plt.figure(figsize=(12, 8))
for k in range(K_FINAL):
    mask = clusters_amostra == k
    plt.scatter(
        emb_2d[mask, 0], emb_2d[mask, 1],
        c=[cores[k]], label=f"C{k}", s=5, alpha=0.6
    )

plt.title(f"UMAP — {K_FINAL} clusters (amostra de {AMOSTRA:,} trechos)", fontsize=13)
plt.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("umap_clusters.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico salvo em: umap_clusters.png")

# ─── 7. PALAVRAS MAIS FREQUENTES POR CLUSTER ─────────────────
print("\nPalavras mais frequentes por cluster (top 10):")
from collections import Counter
import re

stopwords = {"the","a","an","of","in","to","and","is","are","was","were",
             "for","with","that","this","it","as","by","from","at","or",
             "be","have","has","had","not","on","but","its","their","they",
             "which","we","our","these","been","were","also","can","may",
             "after","before","between","more","than","however","each"}

for k in range(K_FINAL):
    idx_cluster = [i for i, c in enumerate(clusters) if c == k]
    textos_cluster = [todos_textos[i] for i in idx_cluster[:200]]  # max 200
    palavras = []
    for t in textos_cluster:
        palavras += [w.lower() for w in re.findall(r'\b[a-z]{4,}\b', t)
                     if w.lower() not in stopwords]
    top = Counter(palavras).most_common(10)
    top_str = ", ".join([f"{w}({n})" for w, n in top])
    print(f"  Cluster {k:2d}: {top_str}")

# ─── 8. SALVAR CSV ───────────────────────────────────────────
print("\nSalvando resultados em CSV...")
df_resultado = pd.DataFrame({
    "id":      todos_ids,
    "cluster": clusters,
    "texto":   [t[:300] for t in todos_textos]
})
df_resultado.to_csv("clusters_resultado.csv", index=False)
print("  Salvo em: clusters_resultado.csv")
print("\nScript finalizado!")