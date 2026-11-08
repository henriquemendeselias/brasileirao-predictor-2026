"""
Motor de Simulação de Monte Carlo (10.000 iterações) para a Reta Final do Brasileirão.

Arquitetura Vetorizada (NumPy):
1. Recebe o estado atual da tabela (338 jogos realizados) e a lista dos 42 jogos restantes
   já enriquecidos com `lambda_mandante` e `lambda_visitante`.
2. Permite receber um dicionário opcional de `placares_manuais` (ex: {id_jogo: (gols_m, gols_v)}):
   - Jogos com placar manual têm seus resultados fixados nas 10.000 simulações.
   - Jogos em aberto têm seus gols sorteados estocasticamente via Distribuição de Poisson:
     `rng.poisson(lam=lambda, size=(n_simulacoes, n_jogos))`
3. Consolida Pontos, Vitórias, Saldo de Gols e Gols Pró para as 20 equipes simultaneamente
   em matrizes `(n_simulacoes, 20)`.
4. Ordena cada uma das 10.000 simulações pelos 4 critérios de desempate oficiais da CBF:
   Pontos > Vitórias > Saldo de Gols > Gols Pró.
5. Calcula as frequências relativas de cada posição (1º a 20º) e das zonas:
   - Título (1º)
   - Libertadores Direta / G4 (1º ao 4º)
   - Pré-Libertadores (5º ao 6º) e Libertadores Total / G6 (1º ao 6º)
   - Sul-Americana (7º ao 12º)
   - Zona Neutro / Permanência sem vaga (13º ao 16º)
   - Rebaixamento / Z4 (17º ao 20º)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


N_SIMULACOES_DEFAULT = 10_000


def projetar_tabela_deterministica(
    df_tabela_atual: pd.DataFrame,
    df_jogos_previstos: pd.DataFrame,
    placares_manuais: Optional[Dict[int, Tuple[int, int]]] = None,
    preencher_restantes_com_provavel: bool = True,
) -> pd.DataFrame:
    """
    Calcula a Tabela de Classificação projetada a partir dos placares manuais
    do usuário e (opcionalmente) dos placares mais prováveis de Poisson para os
    jogos deixados em branco (fechando os 38 jogos da tabela).
    """
    placares_manuais = placares_manuais or {}

    # Copia estado base dos clubes
    estado = {}
    for _, row in df_tabela_atual.iterrows():
        clube = row["clube"]
        estado[clube] = {
            "clube": clube,
            "sigla": row["sigla"],
            "pontos": int(row["pontos"]),
            "jogos": int(row["jogos"]),
            "vitorias": int(row["vitorias"]),
            "empates": int(row["empates"]),
            "derrotas": int(row["derrotas"]),
            "gols_pro": int(row["gols_pro"]),
            "gols_contra": int(row["gols_contra"]),
            "saldo_gols": int(row["saldo_gols"]),
            "forca_ataque_mandante": float(row["forca_ataque_mandante"]),
            "forca_defesa_mandante": float(row["forca_defesa_mandante"]),
            "forca_ataque_visitante": float(row["forca_ataque_visitante"]),
            "forca_defesa_visitante": float(row["forca_defesa_visitante"]),
            "forca_ataque_geral": float(row["forca_ataque_geral"]),
            "forca_defesa_geral": float(row["forca_defesa_geral"]),
        }

    for _, jogo in df_jogos_previstos.iterrows():
        id_jogo = int(jogo["id_jogo"])
        mandante = jogo["mandante"]
        visitante = jogo["visitante"]

        if id_jogo in placares_manuais:
            gm, gv = placares_manuais[id_jogo]
        elif preencher_restantes_com_provavel:
            gm = int(jogo["gols_mandante_provavel"])
            gv = int(jogo["gols_visitante_provavel"])
        else:
            continue

        # Atualiza estatísticas do mandante e visitante
        estado[mandante]["jogos"] += 1
        estado[visitante]["jogos"] += 1

        estado[mandante]["gols_pro"] += gm
        estado[mandante]["gols_contra"] += gv
        estado[mandante]["saldo_gols"] = estado[mandante]["gols_pro"] - estado[mandante]["gols_contra"]

        estado[visitante]["gols_pro"] += gv
        estado[visitante]["gols_contra"] += gm
        estado[visitante]["saldo_gols"] = estado[visitante]["gols_pro"] - estado[visitante]["gols_contra"]

        if gm > gv:
            estado[mandante]["vitorias"] += 1
            estado[mandante]["pontos"] += 3
            estado[visitante]["derrotas"] += 1
        elif gm < gv:
            estado[visitante]["vitorias"] += 1
            estado[visitante]["pontos"] += 3
            estado[mandante]["derrotas"] += 1
        else:
            estado[mandante]["empates"] += 1
            estado[visitante]["empates"] += 1
            estado[mandante]["pontos"] += 1
            estado[visitante]["pontos"] += 1

    registros = []
    for info in estado.values():
        j = info["jogos"]
        p = info["pontos"]
        info["aproveitamento"] = round((p / (j * 3)) * 100, 1) if j > 0 else 0.0
        registros.append(info)

    df_proj = pd.DataFrame(registros)
    df_proj = df_proj.sort_values(
        by=["pontos", "vitorias", "saldo_gols", "gols_pro", "clube"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    df_proj.insert(0, "posicao", range(1, len(df_proj) + 1))

    return df_proj


def definir_coluna_inteligente(
    posicao: int,
    prob_titulo: float,
    prob_libertadores: float,
    prob_sulamericana: float,
    prob_rebaixamento: float,
    prob_posicao_exata: float,
) -> Dict[str, object]:
    """
    Determina o indicador principal da 'Coluna Inteligente' à direita da tabela
    de acordo com a faixa de classificação e as probabilidades do clube.
    """
    if posicao == 1 or prob_titulo >= 5.0:
        return {
            "rotulo_objetivo": "Título",
            "prob_objetivo": prob_titulo,
            "cor_badge": "amber",
        }
    elif posicao <= 6:
        return {
            "rotulo_objetivo": "Libertadores",
            "prob_objetivo": prob_libertadores,
            "cor_badge": "emerald",
        }
    elif posicao <= 12:
        return {
            "rotulo_objetivo": "Sul-Americana",
            "prob_objetivo": prob_sulamericana,
            "cor_badge": "sky",
        }
    elif posicao >= 17 or prob_rebaixamento >= 10.0:
        return {
            "rotulo_objetivo": "Risco Z4",
            "prob_objetivo": prob_rebaixamento,
            "cor_badge": "rose",
        }
    else:
        # 13º ao 16º com baixo risco de Z4: destaca a maior probabilidade ativa (Sula ou Permanência)
        if prob_sulamericana >= prob_rebaixamento and prob_sulamericana > 0.0:
            return {
                "rotulo_objetivo": "Sul-Americana",
                "prob_objetivo": prob_sulamericana,
                "cor_badge": "sky",
            }
        return {
            "rotulo_objetivo": f"{posicao}º Lugar",
            "prob_objetivo": prob_posicao_exata,
            "cor_badge": "slate",
        }


def executar_simulacao_monte_carlo(
    df_tabela_atual: pd.DataFrame,
    df_jogos_previstos: pd.DataFrame,
    placares_manuais: Optional[Dict[int, Tuple[int, int]]] = None,
    n_simulacoes: int = N_SIMULACOES_DEFAULT,
    seed: Optional[int] = 42,
) -> Dict[str, object]:
    """
    Executa de forma 100% vetorizada as `n_simulacoes` de Monte Carlo para os jogos restantes.

    Retorna:
    - `df_resumo`: Tabela combinando a projeção de 38 jogos + probabilidades de Monte Carlo
      (Título, G4, G6/Libertadores, Sul-Americana, Z4, Posição Exata e Coluna Inteligente).
    - `matriz_pontos_simulados`: Array (n_simulacoes, n_clubes) com a pontuação final
      de cada clube em cada simulação (para gráficos de densidade/boxplot no BI).
    - `matriz_posicoes_simuladas`: Array (n_clubes, 20) com a probabilidade de cada clube
      terminar em cada posição de 1º a 20º.
    - `clubes_ordem`: Lista ordenada de nomes dos clubes correspondente às matrizes.
    """
    placares_manuais = placares_manuais or {}
    rng = np.random.default_rng(seed=seed)

    # Ordem alfabética fixa para indexação matricial
    df_base = df_tabela_atual.sort_values("clube").reset_index(drop=True)
    clubes = df_base["clube"].tolist()
    n_clubes = len(clubes)
    mapa_idx = {clube: i for i, clube in enumerate(clubes)}

    # Vetores base (1, n_clubes)
    pontos_base = df_base["pontos"].to_numpy(dtype=np.int32)[None, :]
    vitorias_base = df_base["vitorias"].to_numpy(dtype=np.int32)[None, :]
    gp_base = df_base["gols_pro"].to_numpy(dtype=np.int32)[None, :]
    gc_base = df_base["gols_contra"].to_numpy(dtype=np.int32)[None, :]

    # Dados dos jogos restantes (n_jogos,)
    n_jogos = len(df_jogos_previstos)
    idx_mandantes = np.array([mapa_idx[m] for m in df_jogos_previstos["mandante"]], dtype=np.int32)
    idx_visitantes = np.array([mapa_idx[v] for v in df_jogos_previstos["visitante"]], dtype=np.int32)
    lambdas_m = df_jogos_previstos["lambda_mandante"].to_numpy(dtype=np.float64)
    lambdas_v = df_jogos_previstos["lambda_visitante"].to_numpy(dtype=np.float64)
    ids_jogos = df_jogos_previstos["id_jogo"].to_numpy(dtype=np.int32)

    # Sorteio vetorizado de gols: matrizes (n_simulacoes, n_jogos)
    gols_m_sim = rng.poisson(lam=lambdas_m, size=(n_simulacoes, n_jogos)).astype(np.int32)
    gols_v_sim = rng.poisson(lam=lambdas_v, size=(n_simulacoes, n_jogos)).astype(np.int32)

    # Aplica trava nos jogos com placar manual definido pelo usuário
    for j_col, id_jogo in enumerate(ids_jogos):
        if int(id_jogo) in placares_manuais:
            gm_fixo, gv_fixo = placares_manuais[int(id_jogo)]
            gols_m_sim[:, j_col] = int(gm_fixo)
            gols_v_sim[:, j_col] = int(gv_fixo)

    # Resultados por jogo nas simulações
    vitoria_m = (gols_m_sim > gols_v_sim).astype(np.int32)
    empate = (gols_m_sim == gols_v_sim).astype(np.int32)
    vitoria_v = (gols_m_sim < gols_v_sim).astype(np.int32)

    pontos_jogo_m = (vitoria_m * 3) + empate
    pontos_jogo_v = (vitoria_v * 3) + empate

    # Acumulação rápida por clube usando matrizes de incidência (n_jogos, n_clubes)
    incidencia_m = np.zeros((n_jogos, n_clubes), dtype=np.int32)
    incidencia_v = np.zeros((n_jogos, n_clubes), dtype=np.int32)
    incidencia_m[np.arange(n_jogos), idx_mandantes] = 1
    incidencia_v[np.arange(n_jogos), idx_visitantes] = 1

    pontos_finais = pontos_base + (pontos_jogo_m @ incidencia_m) + (pontos_jogo_v @ incidencia_v)
    vitorias_finais = vitorias_base + (vitoria_m @ incidencia_m) + (vitoria_v @ incidencia_v)
    gp_finais = gp_base + (gols_m_sim @ incidencia_m) + (gols_v_sim @ incidencia_v)
    gc_finais = gc_base + (gols_v_sim @ incidencia_m) + (gols_m_sim @ incidencia_v)
    sg_finais = gp_finais - gc_finais

    # Chave composta inteira de 64 bits para ordenação simultânea pelos 4 critérios da CBF:
    # 1º Pontos (0..114) | 2º Vitórias (0..38) | 3º Saldo + 500 (0..1000) | 4º Gols Pró (0..250)
    # Adicionamos pequeno desempate determinístico por índice para evitar colisão residual
    chave_desempate = (
        (pontos_finais.astype(np.int64) * 1_000_000_000)
        + (vitorias_finais.astype(np.int64) * 10_000_000)
        + ((sg_finais.astype(np.int64) + 500) * 10_000)
        + (gp_finais.astype(np.int64) * 25)
        + (n_clubes - np.arange(n_clubes, dtype=np.int64))[None, :]
    )

    # Ordena decrescentemente cada linha (simulação): `ordem_clubes[s, p]` = clube na posição p+1
    ordem_clubes = np.argsort(-chave_desempate, axis=1)

    # Matriz de contagem de posições: contagens[clube_idx, posicao_0_a_19]
    contagens_posicoes = np.zeros((n_clubes, n_clubes), dtype=np.int32)
    for pos in range(n_clubes):
        clubes_na_pos = ordem_clubes[:, pos]
        contagem = np.bincount(clubes_na_pos, minlength=n_clubes)
        contagens_posicoes[:, pos] = contagem

    probs_posicoes = (contagens_posicoes / n_simulacoes) * 100.0

    # Tabela determinística projetada de 38 jogos (palpites manuais + mais prováveis de Poisson)
    df_proj = projetar_tabela_deterministica(
        df_tabela_atual=df_tabela_atual,
        df_jogos_previstos=df_jogos_previstos,
        placares_manuais=placares_manuais,
        preencher_restantes_com_provavel=True,
    )

    # Enriquecimento da tabela projetada com as métricas de Monte Carlo
    linhas_enriquecidas = []
    for _, row in df_proj.iterrows():
        clube = row["clube"]
        c_idx = mapa_idx[clube]
        pos_projetada = int(row["posicao"])

        p_titulo = float(probs_posicoes[c_idx, 0])
        p_g4 = float(np.sum(probs_posicoes[c_idx, 0:4]))
        p_pre_liberta = float(np.sum(probs_posicoes[c_idx, 4:6]))
        p_libertadores = float(np.sum(probs_posicoes[c_idx, 0:6]))
        p_sulamericana = float(np.sum(probs_posicoes[c_idx, 6:12]))
        p_rebaixamento = float(np.sum(probs_posicoes[c_idx, 16:20]))
        p_pos_exata = float(probs_posicoes[c_idx, pos_projetada - 1])

        pts_clube = pontos_finais[:, c_idx]
        media_pts_mc = float(np.mean(pts_clube))
        p10_pts = float(np.percentile(pts_clube, 10))
        p90_pts = float(np.percentile(pts_clube, 90))

        col_inteligente = definir_coluna_inteligente(
            posicao=pos_projetada,
            prob_titulo=round(p_titulo, 2),
            prob_libertadores=round(p_libertadores, 2),
            prob_sulamericana=round(p_sulamericana, 2),
            prob_rebaixamento=round(p_rebaixamento, 2),
            prob_posicao_exata=round(p_pos_exata, 2),
        )

        registro = row.to_dict()
        registro.update(
            {
                "prob_titulo": round(p_titulo, 2),
                "prob_g4": round(p_g4, 2),
                "prob_pre_libertadores": round(p_pre_liberta, 2),
                "prob_libertadores": round(p_libertadores, 2),
                "prob_sulamericana": round(p_sulamericana, 2),
                "prob_rebaixamento": round(p_rebaixamento, 2),
                "prob_posicao_projetada": round(p_pos_exata, 2),
                "media_pontos_mc": round(media_pts_mc, 1),
                "pontos_p10_mc": int(round(p10_pts)),
                "pontos_p90_mc": int(round(p90_pts)),
                "rotulo_coluna_inteligente": col_inteligente["rotulo_objetivo"],
                "valor_coluna_inteligente": col_inteligente["prob_objetivo"],
                "cor_coluna_inteligente": col_inteligente["cor_badge"],
            }
        )
        linhas_enriquecidas.append(registro)

    df_resumo = pd.DataFrame(linhas_enriquecidas)

    return {
        "df_resumo": df_resumo,
        "matriz_pontos_simulados": pontos_finais,
        "matriz_posicoes_prob": probs_posicoes,
        "clubes_ordem": clubes,
        "n_simulacoes": n_simulacoes,
    }


if __name__ == "__main__":
    from src.data_engineering.etl import executar_pipeline_etl
    from src.data_science.poisson import prever_todos_jogos_restantes

    _, df_restantes, df_tabela, metricas = executar_pipeline_etl()
    df_previstos = prever_todos_jogos_restantes(df_restantes, df_tabela, metricas)
    resultado_mc = executar_simulacao_monte_carlo(df_tabela, df_previstos, seed=42)

    df_res = resultado_mc["df_resumo"]
    print("=== SIMULAÇÃO DE MONTE CARLO (10.000 ITERAÇÕES) + PROJEÇÃO 38 JOGOS ===")
    cols = [
        "posicao",
        "clube",
        "pontos",
        "jogos",
        "vitorias",
        "saldo_gols",
        "prob_titulo",
        "prob_libertadores",
        "prob_sulamericana",
        "prob_rebaixamento",
        "rotulo_coluna_inteligente",
        "valor_coluna_inteligente",
    ]
    print(df_res[cols].to_string(index=False))
