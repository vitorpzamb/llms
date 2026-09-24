"""
inspecionar_cluster.py
Mostra amostras de trechos de um cluster específico para entender seu conteúdo.
"""

import pandas as pd
import numpy as np

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
CSV_PATH      = "./clusters_resultado.csv"
CLUSTER_ALVO  = 1    # cluster a inspecionar
N_AMOSTRAS    = 20   # quantos trechos mostrar
SEED          = 42

# ─── CARREGAR CSV ────────────────────────────────────────────
print(f"Carregando {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

print(f"\nDistribuição de clusters:")
print(df["cluster"].value_counts().sort_index().to_string())

# ─── INSPECIONAR CLUSTER ALVO ────────────────────────────────
df_cluster = df[df["cluster"] == CLUSTER_ALVO].copy()
print(f"\n{'─'*60}")
print(f"Cluster {CLUSTER_ALVO}: {len(df_cluster):,} trechos")
print(f"{'─'*60}")

# tamanho dos trechos
df_cluster["texto"] = df_cluster["texto"].fillna("").astype(str)
df_cluster["n_palavras"] = df_cluster["texto"].str.split().str.len()
print(f"\nTamanho dos trechos (palavras):")
print(f"  Média:   {df_cluster['n_palavras'].mean():.0f}")
print(f"  Mediana: {df_cluster['n_palavras'].median():.0f}")
print(f"  Mínimo:  {df_cluster['n_palavras'].min()}")
print(f"  Máximo:  {df_cluster['n_palavras'].max()}")

# amostras aleatórias
print(f"\n{'─'*60}")
print(f"Amostras aleatórias do Cluster {CLUSTER_ALVO}:")
print(f"{'─'*60}")

amostra = df_cluster.sample(N_AMOSTRAS, random_state=SEED)
for i, (_, row) in enumerate(amostra.iterrows()):
    print(f"\n[{i+1}] ID: {row['id']} | Palavras: {row['n_palavras']}")
    print(f"     {row['texto']}")

# comparação com outro cluster para referência
print(f"\n{'─'*60}")
print(f"Comparação — 5 amostras do Cluster 0 (referência):")
print(f"{'─'*60}")
df_c0 = df[df["cluster"] == 0].sample(5, random_state=SEED)
for i, (_, row) in enumerate(df_c0.iterrows()):
    n = len(row["texto"].split())
    print(f"\n[{i+1}] ID: {row['id']} | Palavras: {n}")
    print(f"     {row['texto']}")