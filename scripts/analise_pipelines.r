# ═══════════════════════════════════════════════════════════════
# analise_pipelines.R
# Análise comparativa dos 3 pipelines — LLM base, RAG, RAG+BERT
# ═══════════════════════════════════════════════════════════════

library(tidyverse)
library(knitr)

# ─── 0. CONFIGURAÇÃO ────────────────────────────────────────
# Ajuste o caminho se necessário
PASTA_OUTPUTS <- "./outputs"

# ─── 1. CARREGAR OS CSVs ────────────────────────────────────
p1 <- read.csv(file.path(PASTA_OUTPUTS, "resultados_pipeline1.csv"))
p2 <- read.csv(file.path(PASTA_OUTPUTS, "resultados_pipeline2.csv"))
p3 <- read.csv(file.path(PASTA_OUTPUTS, "resultados_pipeline3.csv"))

# unir tudo num único dataframe
df <- bind_rows(p1, p2, p3)

# rótulos mais legíveis
df <- df %>%
  mutate(
    pipeline = case_when(
      pipeline == "LLM_base"     ~ "1. LLM Base",
      pipeline == "LLM_RAG"      ~ "2. LLM + RAG",
      pipeline == "LLM_RAG_BERT" ~ "3. LLM + RAG + BERT"
    ),
    modelo = case_when(
      modelo == "llama3"   ~ "LLaMA 3 (8B)",
      modelo == "mistral"  ~ "Mistral (7B)",
      modelo == "gemma:2b" ~ "Gemma (2B)"
    )
  )

cat("✓ Dados carregados:", nrow(df), "linhas\n")
cat("  Pipelines:", unique(df$pipeline), "\n")
cat("  Modelos:", unique(df$modelo), "\n\n")

# ─── 2. TABELA RESUMO ────────────────────────────────────────
cat("═══════════════════════════════════════\n")
cat("TABELA RESUMO — Médias por Pipeline e Modelo\n")
cat("═══════════════════════════════════════\n")

resumo <- df %>%
  group_by(pipeline, modelo) %>%
  summarise(
    METEOR   = round(mean(meteor, na.rm = TRUE), 4),
    Cosseno  = round(mean(cosseno, na.rm = TRUE), 4),
    Tokens   = round(mean(tokens_total, na.rm = TRUE), 0),
    Tempo_s  = round(mean(tempo_s, na.rm = TRUE), 1),
    .groups  = "drop"
  ) %>%
  arrange(pipeline, modelo)

print(resumo)
cat("\n")

# ─── 3. GRÁFICO — METEOR POR PIPELINE ────────────────────────
p_meteor <- ggplot(resumo, aes(x = pipeline, y = METEOR, fill = modelo)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(METEOR, 3)),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c(
    "LLaMA 3 (8B)"  = "#1a5cff",
    "Mistral (7B)"  = "#e84c1e",
    "Gemma (2B)"    = "#2db36f"
  )) +
  labs(
    title    = "Qualidade das Respostas por Pipeline (METEOR)",
    subtitle = "Média das 10 perguntas — quanto maior, melhor",
    x        = "Pipeline",
    y        = "METEOR médio",
    fill     = "Modelo"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    axis.text.x   = element_text(angle = 10, hjust = 1),
    legend.position = "top"
  ) +
  ylim(0, max(resumo$METEOR) * 1.2)

ggsave(file.path(PASTA_OUTPUTS, "grafico_meteor.png"),
       p_meteor, width = 9, height = 6, dpi = 150)
cat("✓ Salvo: grafico_meteor.png\n")

# ─── 4. GRÁFICO — COSSENO POR PIPELINE ───────────────────────
p_cosseno <- ggplot(resumo, aes(x = pipeline, y = Cosseno, fill = modelo)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(Cosseno, 3)),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c(
    "LLaMA 3 (8B)"  = "#1a5cff",
    "Mistral (7B)"  = "#e84c1e",
    "Gemma (2B)"    = "#2db36f"
  )) +
  labs(
    title    = "Similaridade Semântica por Pipeline (Cosseno)",
    subtitle = "Média das 10 perguntas — quanto maior, melhor",
    x        = "Pipeline",
    y        = "Cosseno médio",
    fill     = "Modelo"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    axis.text.x   = element_text(angle = 10, hjust = 1),
    legend.position = "top"
  ) +
  ylim(0, 1)

ggsave(file.path(PASTA_OUTPUTS, "grafico_cosseno.png"),
       p_cosseno, width = 9, height = 6, dpi = 150)
cat("✓ Salvo: grafico_cosseno.png\n")

# ─── 5. GRÁFICO — TOKENS POR PIPELINE ────────────────────────
p_tokens <- ggplot(resumo, aes(x = pipeline, y = Tokens, fill = modelo)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(Tokens, 0)),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c(
    "LLaMA 3 (8B)"  = "#1a5cff",
    "Mistral (7B)"  = "#e84c1e",
    "Gemma (2B)"    = "#2db36f"
  )) +
  labs(
    title    = "Tokens Consumidos por Pipeline",
    subtitle = "Média das 10 perguntas — quanto menor, mais eficiente",
    x        = "Pipeline",
    y        = "Tokens médios",
    fill     = "Modelo"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    axis.text.x   = element_text(angle = 10, hjust = 1),
    legend.position = "top"
  )

ggsave(file.path(PASTA_OUTPUTS, "grafico_tokens.png"),
       p_tokens, width = 9, height = 6, dpi = 150)
cat("✓ Salvo: grafico_tokens.png\n")

# ─── 6. SCATTER — QUALIDADE × TOKENS ─────────────────────────
cores_modelo <- c(
  "LLaMA 3 (8B)"  = "#1a5cff",
  "Mistral (7B)"  = "#e84c1e",
  "Gemma (2B)"    = "#2db36f"
)
formas_pipeline <- c(
  "1. LLM Base"        = 16,
  "2. LLM + RAG"       = 17,
  "3. LLM + RAG + BERT"= 15
)

p_scatter_tokens <- ggplot(resumo,
  aes(x = Tokens, y = METEOR, color = modelo, shape = pipeline)) +
  # quadrante ideal
  annotate("rect",
    xmin = -Inf, xmax = median(resumo$Tokens),
    ymin = median(resumo$METEOR), ymax = Inf,
    fill = "#1a5cff", alpha = 0.05) +
  annotate("text",
    x = min(resumo$Tokens) + 50,
    y = max(resumo$METEOR) * 0.98,
    label = "Quadrante ideal\n(alta qualidade, baixo custo)",
    hjust = 0, size = 3, color = "#1a5cff") +
  geom_point(size = 5, alpha = 0.9) +
  geom_text(aes(label = paste0(modelo, "\n", pipeline)),
            vjust = -0.8, size = 2.8, show.legend = FALSE) +
  scale_color_manual(values = cores_modelo) +
  scale_shape_manual(values = formas_pipeline) +
  labs(
    title    = "Qualidade × Tokens Consumidos",
    subtitle = "Cada ponto = combinação modelo × pipeline",
    x        = "Tokens médios por pergunta",
    y        = "METEOR médio",
    color    = "Modelo",
    shape    = "Pipeline"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title     = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(PASTA_OUTPUTS, "scatter_qualidade_tokens.png"),
       p_scatter_tokens, width = 10, height = 7, dpi = 150)
cat("✓ Salvo: scatter_qualidade_tokens.png\n")

# ─── 7. SCATTER — QUALIDADE × TEMPO ──────────────────────────
p_scatter_tempo <- ggplot(resumo,
  aes(x = Tempo_s, y = METEOR, color = modelo, shape = pipeline)) +
  annotate("rect",
    xmin = -Inf, xmax = median(resumo$Tempo_s),
    ymin = median(resumo$METEOR), ymax = Inf,
    fill = "#1a5cff", alpha = 0.05) +
  annotate("text",
    x = min(resumo$Tempo_s) + 0.5,
    y = max(resumo$METEOR) * 0.98,
    label = "Quadrante ideal\n(alta qualidade, baixo tempo)",
    hjust = 0, size = 3, color = "#1a5cff") +
  geom_point(size = 5, alpha = 0.9) +
  geom_text(aes(label = paste0(modelo, "\n", pipeline)),
            vjust = -0.8, size = 2.8, show.legend = FALSE) +
  scale_color_manual(values = cores_modelo) +
  scale_shape_manual(values = formas_pipeline) +
  labs(
    title    = "Qualidade × Tempo de Inferência",
    subtitle = "Cada ponto = combinação modelo × pipeline",
    x        = "Tempo médio por pergunta (segundos)",
    y        = "METEOR médio",
    color    = "Modelo",
    shape    = "Pipeline"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title     = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(PASTA_OUTPUTS, "scatter_qualidade_tempo.png"),
       p_scatter_tempo, width = 10, height = 7, dpi = 150)
cat("✓ Salvo: scatter_qualidade_tempo.png\n")

# ─── 8. TABELA POR PERGUNTA ───────────────────────────────────
cat("\n═══════════════════════════════════════\n")
cat("RESPOSTAS POR PERGUNTA (amostra — LLaMA 3)\n")
cat("═══════════════════════════════════════\n")

tabela_perguntas <- df %>%
  filter(modelo == "LLaMA 3 (8B)") %>%
  select(pipeline, pergunta, ground_truth, resposta, meteor, cosseno) %>%
  mutate(
    pergunta     = str_trunc(pergunta, 60),
    ground_truth = str_trunc(ground_truth, 80),
    resposta     = str_trunc(resposta, 80)
  )

for (pipe in unique(tabela_perguntas$pipeline)) {
  cat("\n---", pipe, "---\n")
  subset <- tabela_perguntas %>% filter(pipeline == pipe)
  for (i in 1:nrow(subset)) {
    cat(sprintf("\n[%d] %s\n", i, subset$pergunta[i]))
    cat(sprintf("    GT:       %s\n", subset$ground_truth[i]))
    cat(sprintf("    Resposta: %s\n", subset$resposta[i]))
    cat(sprintf("    METEOR: %.4f | Cosseno: %.4f\n",
                subset$meteor[i], subset$cosseno[i]))
  }
}

cat("\n\n═══════════════════════════════════════\n")
cat("ANÁLISE CONCLUÍDA\n")
cat("Gráficos salvos em:", PASTA_OUTPUTS, "\n")
cat("═══════════════════════════════════════\n")


# ─── EXIBIR GRÁFICOS NO R ────────────────────────────────────
print(p_meteor)
print(p_cosseno)
print(p_tokens)
print(p_scatter_tokens)
print(p_scatter_tempo)

# ─── TABELAS COMPARATIVAS ────────────────────────────────────
library(tidyr)

# função auxiliar para montar tabela wide
tabela_wide <- function(df, coluna) {
  resumo %>%
    select(pipeline, modelo, valor = all_of(coluna)) %>%
    pivot_wider(names_from = pipeline, values_from = valor) %>%
    rename(Modelo = modelo)
}

cat("\n═══════════════════════════════════════\n")
cat("TABELA — METEOR por Pipeline\n")
cat("═══════════════════════════════════════\n")
print(tabela_wide(resumo, "METEOR"))

cat("\n═══════════════════════════════════════\n")
cat("TABELA — Cosseno por Pipeline\n")
cat("═══════════════════════════════════════\n")
print(tabela_wide(resumo, "Cosseno"))

cat("\n═══════════════════════════════════════\n")
cat("TABELA — Tokens por Pipeline\n")
cat("═══════════════════════════════════════\n")
print(tabela_wide(resumo, "Tokens"))