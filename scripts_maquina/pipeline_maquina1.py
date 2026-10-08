"""
pipeline_maquina1.py — Máquina 1
Pipelines: P1 (LLM Base) + P3 (RAG + BERT)
Lógica: 1 modelo por vez, sequencial, com GPU (CUDA)
Retomada automática via CSVs incrementais
Resumo a cada 50 perguntas com ETA
"""

import time
import os
import csv
import numpy as np
import pandas as pd
import requests
import chromadb
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
OLLAMA_URL     = "http://localhost:11434/api/generate"
PASTA_CHROMA   = "./chroma_db"
COLECAO        = "bioasq_corpus"
PASTA_OUTPUTS  = "./outputs_maquina"
BERT_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_BERT_RAG = 10
TOP_K_BERT_FIN = 3
TIMEOUT_OLLAMA = 120
DEVICE         = "cuda"   # usar GPU — trocar para "cpu" se não tiver CUDA
RESUMO_CADA    = 50       # imprimir resumo a cada N perguntas

os.makedirs(PASTA_OUTPUTS, exist_ok=True)

CSV_P1 = f"{PASTA_OUTPUTS}/resultados_P1_LLM_BASE.csv"
CSV_P3 = f"{PASTA_OUTPUTS}/resultados_P3_LLM_RAG_BERT.csv"

COLUNAS_P1 = [
    "pipeline", "modelo", "pergunta_id", "pergunta", "ground_truth",
    "resposta", "meteor", "cosseno",
    "tempo_ollama_s", "tempo_total_s", "tokens_prompt", "tokens_resposta", "tokens_total",
    "timestamp"
]
COLUNAS_P3 = COLUNAS_P1 + ["trechos_usados", "dist_media_rag", "score_medio_bert"]

# ─── 1. CARREGAR RECURSOS ────────────────────────────────────
print("=" * 60)
print("MÁQUINA 1 — P1 (LLM Base) + P3 (RAG + BERT)")
print("=" * 60)

print("\n[1/5] Perguntas BioASQ...")
qa    = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
df_qa = pd.concat([
    pd.DataFrame(qa["train"]),
    pd.DataFrame(qa["test"])
], ignore_index=True)
print(f"  {len(df_qa):,} perguntas carregadas")

print(f"\n[2/5] Modelo de embedding (device={DEVICE})...")
emb_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
print("  Pronto!")

print(f"\n[3/5] BERT cross-encoder (device={DEVICE})...")
cross_encoder = CrossEncoder(BERT_MODEL, device=DEVICE)
print("  Pronto!")

print("\n[4/5] ChromaDB...")
client  = chromadb.PersistentClient(path=PASTA_CHROMA)
colecao = client.get_collection(COLECAO)
print(f"  {colecao.count():,} trechos indexados")

print("\n[5/5] Recursos prontos!\n")

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
    t  = time.time() - t0
    d  = r.json()
    return (
        d.get("response", "").strip(),
        t,
        d.get("prompt_eval_count", 0),
        d.get("eval_count", 0)
    )

def metricas(resposta, ground_truth):
    ref = nltk.word_tokenize(ground_truth.lower())
    hyp = nltk.word_tokenize(resposta.lower())
    m   = meteor_score([ref], hyp)
    eg  = emb_model.encode([resposta])
    er  = emb_model.encode([ground_truth])
    c   = cosine_similarity(eg, er)[0][0]
    return round(m, 4), round(float(c), 4)

def rag_busca(pergunta, top_k):
    vetor = emb_model.encode(pergunta).tolist()
    res   = colecao.query(query_embeddings=[vetor], n_results=top_k)
    return res["documents"][0], float(np.mean(res["distances"][0]))

def bert_rerank(pergunta, candidatos, top_k=TOP_K_BERT_FIN):
    pares  = [(pergunta, t) for t in candidatos]
    scores = cross_encoder.predict(pares)
    ranked = sorted(zip(candidatos, scores), key=lambda x: x[1], reverse=True)
    top    = ranked[:top_k]
    return [r[0] for r in top], round(float(np.mean([r[1] for r in top])), 4)

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

def append_csv(csv_path, row, colunas):
    novo = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        if novo:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in colunas})

def imprimir_resumo(pipeline, modelo, processadas, total, tempos_janela):
    media     = sum(tempos_janela) / len(tempos_janela)
    restantes = total - processadas
    eta_s     = media * restantes
    eta_h     = eta_s / 3600
    print(f"  [{pipeline}][{modelo}] {processadas}/{total} | "
          f"Média últimas {len(tempos_janela)}: {media:.1f}s/perg | "
          f"ETA: {eta_h:.1f}h | "
          f"{time.strftime('%H:%M:%S')}")

# ─── 3. PIPELINE 1 — LLM BASE ────────────────────────────────

def rodar_p1(modelo, df_perguntas):
    ja_feitos   = ids_ja_processados(CSV_P1, modelo)
    pendentes   = df_perguntas[~df_perguntas["id"].astype(str).isin(ja_feitos)]
    total       = len(df_perguntas)
    print(f"\n[P1][{modelo}] {len(pendentes):,} pendentes de {total:,}")

    tempos      = []
    processadas = total - len(pendentes)

    for _, row in pendentes.iterrows():
        t_inicio = time.time()
        try:
            resposta, t_ollama, tp, tr = chamar_ollama(modelo, prompt_base(row["question"]))
            m, c    = metricas(resposta, row["answer"])
            t_total = time.time() - t_inicio

            append_csv(CSV_P1, {
                "pipeline": "P1_LLM_BASE", "modelo": modelo,
                "pergunta_id": row["id"], "pergunta": row["question"],
                "ground_truth": row["answer"], "resposta": resposta,
                "meteor": m, "cosseno": c,
                "tempo_ollama_s": round(t_ollama, 2),
                "tempo_total_s": round(t_total, 2),
                "tokens_prompt": tp, "tokens_resposta": tr,
                "tokens_total": tp + tr,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, COLUNAS_P1)

            tempos.append(t_total)
            processadas += 1

            if processadas % RESUMO_CADA == 0:
                imprimir_resumo("P1", modelo, processadas, total, tempos[-RESUMO_CADA:])

        except Exception as e:
            print(f"    [P1][{modelo}] ERRO {row['id']}: {e}")

    print(f"  [P1][{modelo}] CONCLUÍDO — {processadas}/{total}")

# ─── 4. PIPELINE 3 — RAG + BERT ──────────────────────────────

def rodar_p3(modelo, df_perguntas):
    ja_feitos   = ids_ja_processados(CSV_P3, modelo)
    pendentes   = df_perguntas[~df_perguntas["id"].astype(str).isin(ja_feitos)]
    total       = len(df_perguntas)
    print(f"\n[P3][{modelo}] {len(pendentes):,} pendentes de {total:,}")

    tempos      = []
    processadas = total - len(pendentes)

    for _, row in pendentes.iterrows():
        t_inicio = time.time()
        try:
            candidatos, dist = rag_busca(row["question"], TOP_K_BERT_RAG)
            trechos, score   = bert_rerank(row["question"], candidatos)
            resposta, t_ollama, tp, tr = chamar_ollama(
                modelo, prompt_contexto(row["question"], trechos)
            )
            m, c    = metricas(resposta, row["answer"])
            t_total = time.time() - t_inicio

            append_csv(CSV_P3, {
                "pipeline": "P3_LLM_RAG_BERT", "modelo": modelo,
                "pergunta_id": row["id"], "pergunta": row["question"],
                "ground_truth": row["answer"], "resposta": resposta,
                "meteor": m, "cosseno": c,
                "tempo_ollama_s": round(t_ollama, 2),
                "tempo_total_s": round(t_total, 2),
                "tokens_prompt": tp, "tokens_resposta": tr,
                "tokens_total": tp + tr,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "trechos_usados": " ||| ".join(trechos),
                "dist_media_rag": round(dist, 4),
                "score_medio_bert": score
            }, COLUNAS_P3)

            tempos.append(t_total)
            processadas += 1

            if processadas % RESUMO_CADA == 0:
                imprimir_resumo("P3", modelo, processadas, total, tempos[-RESUMO_CADA:])

        except Exception as e:
            print(f"    [P3][{modelo}] ERRO {row['id']}: {e}")

    print(f"  [P3][{modelo}] CONCLUÍDO — {processadas}/{total}")

# ─── 5. EXECUÇÃO PRINCIPAL ───────────────────────────────────

print("=" * 60)
print("INICIANDO EXPERIMENTOS")
print(f"  Pipelines: P1 (LLM Base) + P3 (RAG + BERT)")
print(f"  Modelos: {MODELOS} (sequencial)")
print(f"  Device: {DEVICE}")
print(f"  Total de perguntas: {len(df_qa):,}")
print(f"  Resumo a cada {RESUMO_CADA} perguntas")
print("=" * 60)

tempo_inicio_total = time.time()

for modelo in MODELOS:
    print(f"\n{'='*60}")
    print(f"MODELO: {modelo}  |  {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    t0 = time.time()
    rodar_p1(modelo, df_qa)
    rodar_p3(modelo, df_qa)
    t_modelo = time.time() - t0

    print(f"\n  [{modelo}] modelo concluído em {t_modelo/3600:.1f}h ({t_modelo/60:.0f} min)")

# ─── 6. RESUMO FINAL ─────────────────────────────────────────
tempo_total = time.time() - tempo_inicio_total

print("\n" + "=" * 60)
print("EXPERIMENTO CONCLUÍDO")
print(f"  Tempo total: {tempo_total/3600:.1f}h ({tempo_total/60:.0f} min)")
print("=" * 60)

for nome, csv_path in [("P1_LLM_BASE", CSV_P1), ("P3_LLM_RAG_BERT", CSV_P3)]:
    if os.path.exists(csv_path):
        df_res = pd.read_csv(csv_path)
        print(f"\n{nome}:")
        g = df_res.groupby("modelo")[["meteor", "cosseno", "tempo_total_s", "tokens_total"]].mean().round(4)
        print(g.to_string())

# salvar resumo em txt
with open(f"{PASTA_OUTPUTS}/tempo_maquina1.txt", "w") as f:
    f.write(f"Tempo total: {tempo_total/3600:.2f}h\n")
    for nome, csv_path in [("P1_LLM_BASE", CSV_P1), ("P3_LLM_RAG_BERT", CSV_P3)]:
        if os.path.exists(csv_path):
            df_res = pd.read_csv(csv_path)
            f.write(f"\n{nome}:\n")
            g = df_res.groupby("modelo")[["meteor", "cosseno", "tempo_total_s", "tokens_total"]].mean().round(4)
            f.write(g.to_string() + "\n")

print(f"\nResumo salvo em: {PASTA_OUTPUTS}/tempo_maquina1.txt")