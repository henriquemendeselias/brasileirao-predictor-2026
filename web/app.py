"""
Aplicação Web Interativa (Flask) — Simulador da Reta Final do Brasileirão Série A 2026.

Endpoints:
- GET `/`             : Renderiza a interface principal inspirada no Simulador do ge.globo.
- GET `/api/estado`   : Retorna todas as 38 rodadas (338 jogos realizados + 42 jogos futuros
                        com previsão de Poisson), tabela atual (338 jogos) e simulação base.
- POST `/api/simular` : Recebe placares manuais do usuário (`placares_manuais`) e modo de
                        preenchimento, recalculando a Tabela de Classificação e executando
                        as 10.000 simulações de Monte Carlo com a Coluna Inteligente.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request

# Garante que a raiz do projeto esteja no sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_engineering.etl import (
    calcular_metricas_liga,
    carregar_jogos_restantes,
    carregar_partidas_realizadas,
    construir_tabela_e_forcas,
)
from src.data_science.monte_carlo import (
    definir_coluna_inteligente,
    executar_simulacao_monte_carlo,
    projetar_tabela_deterministica,
)
from src.data_science.poisson import prever_todos_jogos_restantes

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# Cache em memória dos dados base do campeonato
DF_PARTIDAS = carregar_partidas_realizadas()
DF_RESTANTES = carregar_jogos_restantes()
METRICAS_LIGA = calcular_metricas_liga(DF_PARTIDAS)
DF_TABELA_ATUAL = construir_tabela_e_forcas(DF_PARTIDAS, METRICAS_LIGA)
DF_JOGOS_PREVISTOS = prever_todos_jogos_restantes(DF_RESTANTES, DF_TABELA_ATUAL, METRICAS_LIGA)


def montar_calendario_38_rodadas() -> Dict[int, list]:
    """
    Organiza todas as 380 partidas do campeonato (338 realizadas + 42 futuras)
    agrupadas por rodada (1 a 38).
    """
    rodadas: Dict[int, list] = {r: [] for r in range(1, 39)}

    # 1. Partidas já realizadas (travadas/oficiais)
    for idx, row in DF_PARTIDAS.iterrows():
        r = int(row["rodada"])
        rodadas[r].append(
            {
                "id_unico": f"realizado_{idx}",
                "id_jogo_futuro": None,
                "rodada": r,
                "data": row["data"],
                "horario": row["horario"],
                "estadio": row["estadio"],
                "mandante": row["mandante"],
                "sigla_mandante": row["sigla_mandante"],
                "visitante": row["visitante"],
                "sigla_visitante": row["sigla_visitante"],
                "gols_mandante": int(row["gols_mandante"]),
                "gols_visitante": int(row["gols_visitante"]),
                "realizado": True,
                "placar_provavel": None,
                "gols_mandante_provavel": None,
                "gols_visitante_provavel": None,
                "lambda_mandante": None,
                "lambda_visitante": None,
                "prob_vitoria_mandante": None,
                "prob_empate": None,
                "prob_vitoria_visitante": None,
            }
        )

    # 2. Partidas restantes (editáveis pelo usuário + sugestão de Poisson)
    for _, row in DF_JOGOS_PREVISTOS.iterrows():
        r = int(row["rodada"])
        id_fut = int(row["id_jogo"])
        rodadas[r].append(
            {
                "id_unico": f"futuro_{id_fut}",
                "id_jogo_futuro": id_fut,
                "rodada": r,
                "data": row["data"],
                "horario": row["horario"],
                "estadio": row["estadio"],
                "mandante": row["mandante"],
                "sigla_mandante": row["sigla_mandante"],
                "visitante": row["visitante"],
                "sigla_visitante": row["sigla_visitante"],
                "gols_mandante": None,
                "gols_visitante": None,
                "realizado": False,
                "placar_provavel": row["placar_provavel"],
                "gols_mandante_provavel": int(row["gols_mandante_provavel"]),
                "gols_visitante_provavel": int(row["gols_visitante_provavel"]),
                "lambda_mandante": float(row["lambda_mandante"]),
                "lambda_visitante": float(row["lambda_visitante"]),
                "prob_vitoria_mandante": float(row["prob_vitoria_mandante"]),
                "prob_empate": float(row["prob_empate"]),
                "prob_vitoria_visitante": float(row["prob_vitoria_visitante"]),
            }
        )

    return rodadas


CALENDARIO_38_RODADAS = montar_calendario_38_rodadas()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/estado", methods=["GET"])
def obter_estado_inicial():
    """
    Retorna o estado inicial do simulador:
    - Tabela atual (com os 338 jogos concluídos + probabilidades iniciais de Monte Carlo)
    - Calendário completo da 1ª à 38ª rodada
    - Rodada inicial padrão (34ª Rodada)
    """
    resultado_mc = executar_simulacao_monte_carlo(
        df_tabela_atual=DF_TABELA_ATUAL,
        df_jogos_previstos=DF_JOGOS_PREVISTOS,
        placares_manuais={},
        n_simulacoes=10_000,
        seed=None,
    )
    df_mc = resultado_mc["df_resumo"].set_index("clube")
    probs_posicoes = resultado_mc["matriz_posicoes_prob"]
    mapa_idx_mc = {c: i for i, c in enumerate(resultado_mc["clubes_ordem"])}

    # Monta a tabela inicial exibindo os 339 jogos reais, mas já acompanhada das probabilidades base
    tabela_inicial = []
    for _, row in DF_TABELA_ATUAL.iterrows():
        clube = row["clube"]
        pos = int(row["posicao"])
        dados_mc = df_mc.loc[clube]
        prob_pos_atual = float(probs_posicoes[mapa_idx_mc[clube], pos - 1])

        col_int = definir_coluna_inteligente(
            posicao=pos,
            prob_titulo=float(dados_mc["prob_titulo"]),
            prob_libertadores=float(dados_mc["prob_libertadores"]),
            prob_sulamericana=float(dados_mc["prob_sulamericana"]),
            prob_rebaixamento=float(dados_mc["prob_rebaixamento"]),
            prob_posicao_exata=round(prob_pos_atual, 2),
        )

        tabela_inicial.append(
            {
                "posicao": pos,
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
                "aproveitamento": float(row["aproveitamento"]),
                "prob_titulo": float(dados_mc["prob_titulo"]),
                "prob_libertadores": float(dados_mc["prob_libertadores"]),
                "prob_sulamericana": float(dados_mc["prob_sulamericana"]),
                "prob_rebaixamento": float(dados_mc["prob_rebaixamento"]),
                "rotulo_coluna_inteligente": col_int["rotulo_objetivo"],
                "valor_coluna_inteligente": col_int["prob_objetivo"],
                "cor_coluna_inteligente": col_int["cor_badge"],
            }
        )

    return jsonify(
        {
            "rodada_inicial": 35,
            "metricas_liga": METRICAS_LIGA,
            "tabela": tabela_inicial,
            "rodadas": CALENDARIO_38_RODADAS,
        }
    )


@app.route("/api/simular", methods=["POST"])
def simular_campeonato():
    """
    Recebe:
    {
        "placares_manuais": {"0": [2, 1], "5": [0, 0], ...},
        "modo": "manual" | "sortear_dinamico" | "preencher_provaveis"
    }
    Regras:
    1. `placares_manuais` (digitados pelo usuário) são SEMPRE tomados como verdade absoluta:
       - São travados nas 10.000 simulações de Monte Carlo (atualizando as probabilidades
         da coluna da direita com base nessa verdade).
       - São travados no cálculo da tabela de classificação.
    2. Se `modo == "sortear_dinamico"` (botão Simular Campeonato):
       - Para cada jogo que NÃO está em `placares_manuais`, sorteia estocasticamente um
         placar via Distribuição de Poisson (`rng.poisson(lambda_m)`, `rng.poisson(lambda_v)`).
       - Retorna esses `placares_exibidos` para preencher todas as caixinhas da direita na interface.
       - Calcula a tabela de 38 jogos somando exatamente os `placares_manuais` + `placares_sorteados`.
    3. Se `modo == "preencher_provaveis"` (botão Preencher Mais Prováveis):
       - Para cada jogo que NÃO está em `placares_manuais`, preenche com a moda de Poisson.
       - Retorna esses `placares_exibidos` e a tabela de 38 jogos correspondente.
    4. Se `modo == "manual"` (usuário digitando um placar no teclado):
       - Calcula a tabela parcial somando apenas os `placares_manuais` e recalcula as 10.000
         simulações de Monte Carlo tomando esses placares manuais como verdade.
    """
    payload = request.get_json(silent=True) or {}
    manuais_raw = payload.get("placares_manuais", {})
    simulados_raw = payload.get("placares_simulados", {})
    modo = payload.get("modo", "manual")

    placares_manuais: Dict[int, Tuple[int, int]] = {}
    for k, v in manuais_raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            if v[0] is not None and v[1] is not None and str(v[0]).strip() != "" and str(v[1]).strip() != "":
                gm = max(0, min(20, int(v[0])))
                gv = max(0, min(20, int(v[1])))
                placares_manuais[int(k)] = (gm, gv)

    placares_simulados_anteriores: Dict[int, Tuple[int, int]] = {}
    for k, v in simulados_raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            if v[0] is not None and v[1] is not None and str(v[0]).strip() != "" and str(v[1]).strip() != "":
                gm = max(0, min(20, int(v[0])))
                gv = max(0, min(20, int(v[1])))
                placares_simulados_anteriores[int(k)] = (gm, gv)

    # 1. Roda as 10.000 simulações de Monte Carlo tomando EXCLUSIVAMENTE `placares_manuais` como verdade
    resultado_mc = executar_simulacao_monte_carlo(
        df_tabela_atual=DF_TABELA_ATUAL,
        df_jogos_previstos=DF_JOGOS_PREVISTOS,
        placares_manuais=placares_manuais,
        n_simulacoes=10_000,
        seed=None,
    )
    df_mc = resultado_mc["df_resumo"].set_index("clube")
    probs_posicoes = resultado_mc["matriz_posicoes_prob"]
    mapa_idx_mc = {c: i for i, c in enumerate(resultado_mc["clubes_ordem"])}

    # 2. Define os placares ativos para a Tabela e para as caixinhas da interface
    placares_para_tabela: Dict[int, Tuple[int, int]] = {}
    placares_sorteados_ou_provaveis: Dict[str, list] = {}

    if modo == "sortear_dinamico":
        rng = np.random.default_rng()
        for _, jogo in DF_JOGOS_PREVISTOS.iterrows():
            id_j = int(jogo["id_jogo"])
            if id_j in placares_manuais:
                gm, gv = placares_manuais[id_j]
            else:
                gm = int(rng.poisson(lam=float(jogo["lambda_mandante"])))
                gv = int(rng.poisson(lam=float(jogo["lambda_visitante"])))
            placares_para_tabela[id_j] = (gm, gv)
            placares_sorteados_ou_provaveis[str(id_j)] = [gm, gv]

    elif modo == "preencher_provaveis":
        for _, jogo in DF_JOGOS_PREVISTOS.iterrows():
            id_j = int(jogo["id_jogo"])
            if id_j in placares_manuais:
                gm, gv = placares_manuais[id_j]
            else:
                gm = int(jogo["gols_mandante_provavel"])
                gv = int(jogo["gols_visitante_provavel"])
            placares_para_tabela[id_j] = (gm, gv)
            placares_sorteados_ou_provaveis[str(id_j)] = [gm, gv]

    else:
        # Modo "manual": se já existiam placares simulados visíveis nas caixinhas (azuis),
        # mantemos eles na soma da tabela e sobrepomos com os `placares_manuais` (verdes)
        placares_para_tabela.update(placares_simulados_anteriores)
        placares_para_tabela.update(placares_manuais)

    # 3. Constrói a Tabela de Classificação 100% fiel aos placares visíveis nas caixinhas (`placares_para_tabela`)
    df_tabela_calculada = projetar_tabela_deterministica(
        df_tabela_atual=DF_TABELA_ATUAL,
        df_jogos_previstos=DF_JOGOS_PREVISTOS,
        placares_manuais=placares_para_tabela,
        preencher_restantes_com_provavel=False,
    )

    tabela_saida = []
    for _, row in df_tabela_calculada.iterrows():
        clube = row["clube"]
        pos = int(row["posicao"])
        dados_mc = df_mc.loc[clube]
        prob_pos_atual = float(probs_posicoes[mapa_idx_mc[clube], pos - 1])

        col_int = definir_coluna_inteligente(
            posicao=pos,
            prob_titulo=float(dados_mc["prob_titulo"]),
            prob_libertadores=float(dados_mc["prob_libertadores"]),
            prob_sulamericana=float(dados_mc["prob_sulamericana"]),
            prob_rebaixamento=float(dados_mc["prob_rebaixamento"]),
            prob_posicao_exata=round(prob_pos_atual, 2),
        )
        tabela_saida.append(
            {
                "posicao": pos,
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
                "aproveitamento": float(row["aproveitamento"]),
                "prob_titulo": float(dados_mc["prob_titulo"]),
                "prob_libertadores": float(dados_mc["prob_libertadores"]),
                "prob_sulamericana": float(dados_mc["prob_sulamericana"]),
                "prob_rebaixamento": float(dados_mc["prob_rebaixamento"]),
                "rotulo_coluna_inteligente": col_int["rotulo_objetivo"],
                "valor_coluna_inteligente": col_int["prob_objetivo"],
                "cor_coluna_inteligente": col_int["cor_badge"],
            }
        )

    return jsonify(
        {
            "modo": modo,
            "tabela": tabela_saida,
            "placares_exibidos": placares_sorteados_ou_provaveis,
            "jogos_manuais_contagem": len(placares_manuais),
            "n_simulacoes": 10_000,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
