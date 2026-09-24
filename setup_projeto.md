# Setup do Projeto — RAG Evaluation Pipeline

> **TCC:** Avaliação de Pipelines RAG em Domínio Biomédico: Qualidade e Eficiência entre Modelos de Linguagem Open Source  
> **Autor:** Vitor Prado Zambaldi · UFPR · 2026

---

## Análise inicial da base de dados

A base utilizada é a **rag-mini-bioasq**, disponível publicamente no HuggingFace, derivada do BioASQ challenge. O acesso é feito via biblioteca `datasets` com autenticação pelo token HuggingFace.

### Estrutura da base

| Subset | Linhas | Colunas |
|---|---|---|
| `question-answer-passages` | 4.719 | `question`, `answer`, `relevant_passage_ids`, `id` |
| `text-corpus` | 40.221 | `passage`, `id` |

### Trechos do corpus (palavras por trecho)

| Estatística | Valor |
|---|---|
| Média | 147 palavras |
| Mediana | 163 palavras |
| Mínimo | 1 palavra |
| Máximo | 4.215 palavras |

> Os trechos com menos de 10 palavras são outliers e podem ser filtrados antes de indexar.

### Trechos relevantes por pergunta

| Estatística | Valor |
|---|---|
| Média | 89 trechos |
| Mediana | 60 trechos |
| Mínimo | 8 trechos |
| Máximo | 1.567 trechos |

A distribuição assimétrica indica que algumas perguntas são muito mais complexas que outras — o que permite analisar se os pipelines se comportam diferente em perguntas simples vs. complexas.

---

## Ambiente e dependências

### Modelos LLM (via Ollama — instalação separada)

| Modelo | Criador | Parâmetros | Uso no projeto |
|---|---|---|---|
| LLaMA 3 | Meta | 8B | Modelo de porte médio |
| Mistral | Mistral AI | 7B | Modelo eficiente |
| Gemma | Google | 2B | Modelo leve |

```bash
# baixar os modelos após instalar o Ollama (ollama.com)
ollama pull llama3
ollama pull mistral
ollama pull gemma:2b
```

---

### Bibliotecas Python

#### Dados e base
| Função | Biblioteca |
|---|---|
| Carregamento da base BioASQ | `datasets` |
| Manipulação de dados | `pandas`, `numpy` |

#### Embeddings e busca vetorial
| Função | Biblioteca |
|---|---|
| Geração de embeddings | `sentence-transformers` |
| Banco vetorial e busca | `chromadb` |

#### Modelos de linguagem e pipeline
| Função | Biblioteca |
|---|---|
| Modelos BERT (filtragem) | `transformers`, `torch` |
| Orquestração do pipeline RAG | `langchain`, `langchain-community` |

#### Clusterização
| Função | Biblioteca |
|---|---|
| Algoritmos de clustering (K-Means etc.) | `scikit-learn` |
| Redução de dimensionalidade (visualização) | `umap-learn` |

#### Métricas de avaliação
| Função | Biblioteca |
|---|---|
| METEOR | `nltk` |
| Faithfulness / RAGAS | `ragas` |

#### Visualizações
| Função | Biblioteca |
|---|---|
| Gráficos estáticos | `matplotlib`, `seaborn` |
| Gráficos interativos | `plotly` |

---

### Instalação completa (pip)

```bash
pip install sentence-transformers chromadb transformers torch scikit-learn umap-learn nltk ragas langchain langchain-community matplotlib seaborn plotly
```

---

## Estrutura de pastas do projeto

```
projeto/
│
├── data/                        # base BioASQ (cache HuggingFace)
├── scripts/
│   ├── 01_exploracao.py         # análise inicial da base
│   ├── 02_embeddings.py         # geração de embeddings e indexação
│   ├── 03_clusterizacao.py      # clusterização dos trechos
│   ├── 04_pipeline_base.py      # Pipeline 1 — LLM sem RAG
│   ├── 05_pipeline_rag.py       # Pipeline 2 — LLM + RAG
│   ├── 06_pipeline_bert.py      # Pipeline 3 — LLM + RAG + BERT
│   └── 07_pipeline_cluster.py   # Pipeline 4 — LLM + RAG + Clustering
├── evaluation/
│   ├── metrics.py               # METEOR, cosseno, Faithfulness
│   └── efficiency.py            # tokens e tempo de inferência
├── results/                     # outputs dos experimentos (CSV, JSON)
├── notebooks/                   # análises exploratórias
├── config.yaml                  # configuração do modelo (troca em uma linha)
└── setup_projeto.md             # este arquivo
```

---

## Configuração do modelo (config.yaml)

```yaml
modelo:
  nome: "llama3"        # trocar por: mistral | gemma:2b
  provedor: "ollama"
  url: "http://localhost:11434"

rag:
  top_k: 5              # número de trechos recuperados
  chunk_size: 256       # tamanho dos chunks em tokens

embedding:
  modelo: "all-MiniLM-L6-v2"   # modelo de embedding

avaliacao:
  n_perguntas: 100      # número de perguntas por experimento
```

> Trocar o modelo testado é apenas uma linha no `config.yaml` — o pipeline inteiro roda sem nenhuma outra alteração.

chat atlas