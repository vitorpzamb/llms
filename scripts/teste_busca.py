import chromadb
from sentence_transformers import SentenceTransformer

MODELO_EMBEDDING = "all-MiniLM-L6-v2"
PASTA_CHROMA     = "./chroma_db"
COLECAO          = "bioasq_corpus"

print("Carregando modelo e conectando ao ChromaDB...")
modelo  = SentenceTransformer(MODELO_EMBEDDING)
client  = chromadb.PersistentClient(path=PASTA_CHROMA)
colecao = client.get_collection(COLECAO)

print(f"Coleção carregada: {colecao.count():,} trechos indexados\n")

pergunta = "What is the role of BRCA1 in DNA repair?"
vetor    = modelo.encode(pergunta).tolist()

resultados = colecao.query(
    query_embeddings=[vetor],
    n_results=3
)

print(f"Pergunta: {pergunta}\n")
print("Top 3 trechos recuperados:")
for i, (doc, dist) in enumerate(zip(
    resultados["documents"][0],
    resultados["distances"][0]
)):
    print(f"\n[{i+1}] Distância: {dist:.4f}")
    print(f"     {doc[:300]}...")