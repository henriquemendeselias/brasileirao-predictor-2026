"""
Motor de Probabilidades de Partida Única baseado na Distribuição de Poisson.

Fórmulas fundamentais:
- Expectativa de gols (Lambda):
  lambda_mandante = ForcaAtaqueMandante_i * ForcaDefesaVisitante_j * mu_mandante
  lambda_visitante = ForcaAtaqueVisitante_j * ForcaDefesaMandante_i * mu_visitante

- Função Massa de Probabilidade (PMF) de Poisson:
  P(X = k) = (lambda^k * e^(-lambda)) / k!

- Matriz de Probabilidades Conjuntas (assumindo independência condicional):
  P(Mandante = x, Visitante = y) = P(X = x) * P(Y = y) para x, y in {0, 1, ..., k_max}
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson


K_MAX_DEFAULT = 8


def calcular_lambdas_confronto(
    mandante: str,
    visitante: str,
    df_tabela: pd.DataFrame,
    metricas_liga: Dict[str, float],
) -> Tuple[float, float]:
    """
    Calcula a expectativa de gols (lambda_mandante, lambda_visitante) para um confronto
    específico usando as forças relativas de ataque e defesa e as médias da liga.
    """
    dados_mandante = df_tabela.loc[df_tabela["clube"] == mandante].iloc[0]
    dados_visitante = df_tabela.loc[df_tabela["clube"] == visitante].iloc[0]

    forca_ataque_casa = float(dados_mandante["forca_ataque_mandante"])
    forca_defesa_casa = float(dados_mandante["forca_defesa_mandante"])

    forca_ataque_fora = float(dados_visitante["forca_ataque_visitante"])
    forca_defesa_fora = float(dados_visitante["forca_defesa_visitante"])

    mu_mandante = float(metricas_liga["media_gols_mandante"])
    mu_visitante = float(metricas_liga["media_gols_visitante"])

    lambda_mandante = forca_ataque_casa * forca_defesa_fora * mu_mandante
    lambda_visitante = forca_ataque_fora * forca_defesa_casa * mu_visitante

    return lambda_mandante, lambda_visitante


def construir_matriz_poisson(
    lambda_mandante: float,
    lambda_visitante: float,
    k_max: int = K_MAX_DEFAULT,
) -> np.ndarray:
    """
    Constrói a matriz (k_max + 1) x (k_max + 1) de probabilidades de placares exatos.
    Linha i = gols do mandante (0 a k_max)
    Coluna j = gols do visitante (0 a k_max)
    """
    gols = np.arange(0, k_max + 1)
    prob_mandante = poisson.pmf(gols, mu=lambda_mandante)
    prob_visitante = poisson.pmf(gols, mu=lambda_visitante)

    # Produto externo: M[i, j] = P(Mandante = i) * P(Visitante = j)
    matriz = np.outer(prob_mandante, prob_visitante)
    return matriz


def prever_confronto_poisson(
    mandante: str,
    visitante: str,
    df_tabela: pd.DataFrame,
    metricas_liga: Dict[str, float],
    k_max: int = K_MAX_DEFAULT,
) -> Dict[str, object]:
    """
    Calcula todas as métricas analíticas de Poisson para um confronto:
    - lambda_mandante e lambda_visitante
    - Placar mais provável (moda da matriz conjunta) e sua probabilidade
    - Probabilidades agregadas de Vitória do Mandante, Empate e Vitória do Visitante
    - Matriz de probabilidades (k_max+1 x k_max+1)
    """
    lam_m, lam_v = calcular_lambdas_confronto(mandante, visitante, df_tabela, metricas_liga)
    matriz = construir_matriz_poisson(lam_m, lam_v, k_max=k_max)

    # Índice (x, y) de maior probabilidade conjunta
    idx_flat = int(np.argmax(matriz))
    gols_m_provavel, gols_v_provavel = np.unravel_index(idx_flat, matriz.shape)
    prob_placar_provavel = float(matriz[gols_m_provavel, gols_v_provavel])

    # Probabilidades agregadas:
    # - Triângulo inferior estrito (i > j): Vitória do Mandante
    # - Diagonal principal (i == j): Empate
    # - Triângulo superior estrito (i < j): Vitória do Visitante
    prob_vitoria_mandante = float(np.sum(np.tril(matriz, k=-1)))
    prob_empate = float(np.sum(np.diag(matriz)))
    prob_vitoria_visitante = float(np.sum(np.triu(matriz, k=1)))

    # Normalização leve caso o truncamento em k_max deixe fração residual (> 0.9999)
    soma_total = prob_vitoria_mandante + prob_empate + prob_vitoria_visitante

    return {
        "mandante": mandante,
        "visitante": visitante,
        "lambda_mandante": float(lam_m),
        "lambda_visitante": float(lam_v),
        "gols_mandante_provavel": int(gols_m_provavel),
        "gols_visitante_provavel": int(gols_v_provavel),
        "placar_provavel": f"{int(gols_m_provavel)}x{int(gols_v_provavel)}",
        "prob_placar_provavel": float((prob_placar_provavel / soma_total) * 100.0),
        "prob_vitoria_mandante": float((prob_vitoria_mandante / soma_total) * 100.0),
        "prob_empate": float((prob_empate / soma_total) * 100.0),
        "prob_vitoria_visitante": float((prob_vitoria_visitante / soma_total) * 100.0),
        "soma_prob_matriz": float(soma_total),
        "matriz_placares": matriz,
    }


def prever_todos_jogos_restantes(
    df_restantes: pd.DataFrame,
    df_tabela: pd.DataFrame,
    metricas_liga: Dict[str, float],
    k_max: int = K_MAX_DEFAULT,
) -> pd.DataFrame:
    """
    Enriquece o DataFrame de jogos restantes com as previsões de Poisson
    (lambdas, placar mais provável e probabilidades de 1X2).
    """
    linhas = []
    for idx, row in df_restantes.iterrows():
        prev = prever_confronto_poisson(
            mandante=row["mandante"],
            visitante=row["visitante"],
            df_tabela=df_tabela,
            metricas_liga=metricas_liga,
            k_max=k_max,
        )
        linhas.append(
            {
                "id_jogo": idx,
                "rodada": int(row["rodada"]),
                "data": row["data"],
                "horario": row["horario"],
                "estadio": row["estadio"],
                "mandante": row["mandante"],
                "sigla_mandante": row["sigla_mandante"],
                "visitante": row["visitante"],
                "sigla_visitante": row["sigla_visitante"],
                "lambda_mandante": prev["lambda_mandante"],
                "lambda_visitante": prev["lambda_visitante"],
                "gols_mandante_provavel": prev["gols_mandante_provavel"],
                "gols_visitante_provavel": prev["gols_visitante_provavel"],
                "placar_provavel": prev["placar_provavel"],
                "prob_placar_provavel": prev["prob_placar_provavel"],
                "prob_vitoria_mandante": prev["prob_vitoria_mandante"],
                "prob_empate": prev["prob_empate"],
                "prob_vitoria_visitante": prev["prob_vitoria_visitante"],
            }
        )

    return pd.DataFrame(linhas)
