"""
02_embeddings.py
Gera embeddings dos trechos do BioASQ e indexa no ChromaDB.
Retoma de onde parou se interrompido.
"""

import time
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
MODELO_EMBEDDING = "all-MiniLM-L6-v2"
PASTA_CHROMA     = "./chroma_db"
COLECAO          = "bioasq_corpus"
BATCH_SIZE       = 256

# ─── 1. CARREGAR O CORPUS ────────────────────────────────────
print("Carregando corpus BioASQ...")
corpus = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")
df = corpus["test"].to_pandas()

textos = df["passage"].tolist()
ids    = [str(i) for i in df["id"].tolist()]
print(f"  {len(textos):,} trechos carregados")

# ─── 2. CARREGAR MODELO DE EMBEDDING ─────────────────────────
print(f"\nCarregando modelo de embedding: {MODELO_EMBEDDING}")
modelo = SentenceTransformer(MODELO_EMBEDDING)
print("  Modelo carregado!")

# ─── 3. CONECTAR AO CHROMADB ─────────────────────────────────
print(f"\nConectando ao ChromaDB em: {PASTA_CHROMA}")
client = chromadb.PersistentClient(path=PASTA_CHROMA)

try:
    colecao = client.get_collection(COLECAO)
    ja_indexados = colecao.count()
    print(f"  Coleção existente: {ja_indexados:,} trechos já indexados.")
except:
    colecao = client.create_collection(
        name=COLECAO,
        metadata={"hnsw:space": "cosine"}
    )
    ja_indexados = 0
    print(f"  Coleção '{COLECAO}' criada.")

# ─── 4. GERAR EMBEDDINGS E INDEXAR ───────────────────────────
print(f"\nGerando embeddings e indexando ({len(textos):,} trechos)...")
print(f"  Batch size: {BATCH_SIZE}\n")

inicio = time.time()
total  = len(textos)

for i in range(0, total, BATCH_SIZE):
    if i + BATCH_SIZE <= ja_indexados:
        continue  # já indexado, pula

    batch_textos = textos[i : i + BATCH_SIZE]
    batch_ids    = ids[i : i + BATCH_SIZE]

    embeddings = modelo.encode(
        batch_textos,
        show_progress_bar=False
    ).tolist()

    colecao.add(
        embeddings=embeddings,
        documents=batch_textos,
        ids=batch_ids,
        metadatas=[{"texto": t[:500]} for t in batch_textos]
    )

    processados = min(i + BATCH_SIZE, total)
    pct = processados / total * 100
    elapsed = time.time() - inicio
    eta = (elapsed / max(processados - ja_indexados, 1)) * (total - processados)
    print(f"  {processados:>6,} / {total:,} ({pct:5.1f}%) — "
          f"tempo: {elapsed:.0f}s — restante estimado: {eta:.0f}s")

duracao = time.time() - inicio
print(f"\nIndexação concluída em {duracao:.0f}s ({duracao/60:.1f} min)")
print(f"Total indexado: {colecao.count():,} trechos")

# ─── 5. TESTE DE BUSCA ───────────────────────────────────────
print("\n─── Teste de busca ───")
pergunta_teste = "What is the role of BRCA1 in DNA repair?"

vetor_pergunta = modelo.encode(pergunta_teste).tolist()

resultados = colecao.query(
    query_embeddings=[vetor_pergunta],
    n_results=3
)

print(f"Pergunta: {pergunta_teste}\n")
print("Top 3 trechos recuperados:")
for i, (doc, dist) in enumerate(zip(
    resultados["documents"][0],
    resultados["distances"][0]
)):
    print(f"\n[{i+1}] Distância: {dist:.4f}")
    print(f"     {doc[:300]}...")

print("\nScript finalizado. Índice salvo em ./chroma_db/")