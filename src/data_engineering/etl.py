"""
Pipeline de Engenharia de Dados (ETL) do Brasileirão Série A 2026.

Responsabilidades:
1. Ingestão do arquivo bruto `data/raw/BRA.csv` descartando colunas de odds (`usecols`).
2. Padronização de nomenclatura de clubes, siglas e estádios sede.
3. Ingestão e padronização do calendário restante (`data/raw/jogos_restantes.csv`).
4. Cálculo das Médias Gerais da Liga (Gols de Mandante e Gols de Visitante).
5. Cálculo da Tabela de Classificação Oficial (Critérios CBF: P > V > SG > GP)
   e das Forças Relativas de Ataque e Defesa (Mandante e Visitante).
6. Exportação dos artefatos limpos para `data/processed/`.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.data_engineering.clubes import (
    ESTADIO_POR_NOME,
    SIGLA_POR_NOME,
    padronizar_nome_clube,
)

# Caminhos padrão do projeto
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def carregar_partidas_realizadas(caminho_csv: Path = RAW_DIR / "BRA.csv") -> pd.DataFrame:
    """
    Lê o dataset bruto `BRA.csv` filtrando exclusivamente as colunas de interesse
    (descartando as colunas de odds na leitura) e padroniza clubes, siglas e estádios.
    """
    colunas_indices = [3, 4, 5, 6, 7, 8, 9, 10]
    nomes_colunas = [
        "rodada",
        "data",
        "horario",
        "mandante",
        "visitante",
        "gols_mandante",
        "gols_visitante",
        "resultado",
    ]

    df = pd.read_csv(
        caminho_csv,
        header=None,
        usecols=colunas_indices,
        names=nomes_colunas,
        encoding="utf-8",
    )

    # Tipagem e limpeza
    df["rodada"] = df["rodada"].astype(int)
    df["data"] = df["data"].astype(str).str.strip()
    df["horario"] = df["horario"].astype(str).str.strip()
    df["mandante"] = df["mandante"].apply(padronizar_nome_clube)
    df["visitante"] = df["visitante"].apply(padronizar_nome_clube)
    df["gols_mandante"] = df["gols_mandante"].astype(int)
    df["gols_visitante"] = df["gols_visitante"].astype(int)
    df["resultado"] = df["resultado"].astype(str).str.strip()

    # Enriquecimento com siglas e estádio sede do mandante
    df["sigla_mandante"] = df["mandante"].map(SIGLA_POR_NOME)
    df["sigla_visitante"] = df["visitante"].map(SIGLA_POR_NOME)
    df["estadio"] = df["mandante"].map(ESTADIO_POR_NOME)

    return df


def carregar_jogos_restantes(
    caminho_csv: Path = RAW_DIR / "jogos_restantes.csv",
) -> pd.DataFrame:
    """
    Lê e padroniza o calendário de confrontos pendentes (`jogos_restantes.csv`).
    """
    df = pd.read_csv(caminho_csv, encoding="utf-8")

    df["rodada"] = df["rodada"].astype(int)
    df["data"] = df["data"].astype(str).str.strip()
    df["horario"] = df["horario"].astype(str).str.strip()
    df["mandante"] = df["mandante"].apply(padronizar_nome_clube)
    df["visitante"] = df["visitante"].apply(padronizar_nome_clube)
    df["sigla_mandante"] = df["mandante"].map(SIGLA_POR_NOME)
    df["sigla_visitante"] = df["visitante"].map(SIGLA_POR_NOME)
    df["estadio"] = df["mandante"].map(ESTADIO_POR_NOME)

    return df


def calcular_metricas_liga(df_partidas: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula as médias gerais de gols da liga (Mandante e Visitante).
    """
    total_jogos = len(df_partidas)
    total_gols_mandante = int(df_partidas["gols_mandante"].sum())
    total_gols_visitante = int(df_partidas["gols_visitante"].sum())

    media_mandante = total_gols_mandante / total_jogos
    media_visitante = total_gols_visitante / total_jogos

    return {
        "total_jogos_realizados": total_jogos,
        "total_gols_mandante": total_gols_mandante,
        "total_gols_visitante": total_gols_visitante,
        "total_gols_campeonato": total_gols_mandante + total_gols_visitante,
        "media_gols_mandante": media_mandante,
        "media_gols_visitante": media_visitante,
        "media_gols_partida": (total_gols_mandante + total_gols_visitante) / total_jogos,
    }


def construir_tabela_e_forcas(
    df_partidas: pd.DataFrame, metricas_liga: Dict[str, float]
) -> pd.DataFrame:
    """
    Constrói a Tabela de Classificação Oficial do Brasileirão e calcula as
    Forças Relativas de Ataque e Defesa (em casa, fora e geral) de cada clube.

    Critérios de desempate oficiais considerados:
    1º Pontos (P)
    2º Vitórias (V)
    3º Saldo de Gols (SG)
    4º Gols Pró (GP)
    """
    clubes = sorted(SIGLA_POR_NOME.keys())
    mu_mandante = metricas_liga["media_gols_mandante"]
    mu_visitante = metricas_liga["media_gols_visitante"]

    registros = []

    for clube in clubes:
        jogos_casa = df_partidas[df_partidas["mandante"] == clube]
        jogos_fora = df_partidas[df_partidas["visitante"] == clube]

        # Contagem de partidas
        j_casa = len(jogos_casa)
        j_fora = len(jogos_fora)
        j_total = j_casa + j_fora

        # Resultados em casa
        v_casa = int((jogos_casa["gols_mandante"] > jogos_casa["gols_visitante"]).sum())
        e_casa = int((jogos_casa["gols_mandante"] == jogos_casa["gols_visitante"]).sum())
        d_casa = int((jogos_casa["gols_mandante"] < jogos_casa["gols_visitante"]).sum())

        # Resultados fora
        v_fora = int((jogos_fora["gols_visitante"] > jogos_fora["gols_mandante"]).sum())
        e_fora = int((jogos_fora["gols_visitante"] == jogos_fora["gols_mandante"]).sum())
        d_fora = int((jogos_fora["gols_visitante"] < jogos_fora["gols_mandante"]).sum())

        # Consolidado de resultados
        v = v_casa + v_fora
        e = e_casa + e_fora
        d = d_casa + d_fora
        pontos = (v * 3) + e

        # Gols em casa e fora
        gp_casa = int(jogos_casa["gols_mandante"].sum())
        gc_casa = int(jogos_casa["gols_visitante"].sum())
        gp_fora = int(jogos_fora["gols_visitante"].sum())
        gc_fora = int(jogos_fora["gols_mandante"].sum())

        gp = gp_casa + gp_fora
        gc = gc_casa + gc_fora
        sg = gp - gc

        aproveitamento = round((pontos / (j_total * 3)) * 100, 1) if j_total > 0 else 0.0

        # Médias por mando de campo
        media_gp_casa = gp_casa / j_casa if j_casa > 0 else 0.0
        media_gc_casa = gc_casa / j_casa if j_casa > 0 else 0.0
        media_gp_fora = gp_fora / j_fora if j_fora > 0 else 0.0
        media_gc_fora = gc_fora / j_fora if j_fora > 0 else 0.0

        # Índices de Força Relativa (normalizados pela média da liga)
        # - Ataque Mandante: Gols marcados em casa / Média de gols dos mandantes da liga
        # - Defesa Mandante: Gols sofridos em casa / Média de gols dos visitantes da liga
        # - Ataque Visitante: Gols marcados fora / Média de gols dos visitantes da liga
        # - Defesa Visitante: Gols sofridos fora / Média de gols dos mandantes da liga
        forca_ataque_mandante = media_gp_casa / mu_mandante
        forca_defesa_mandante = media_gc_casa / mu_visitante
        forca_ataque_visitante = media_gp_fora / mu_visitante
        forca_defesa_visitante = media_gc_fora / mu_mandante

        # Forças Gerais (exclusivamente para visualização executiva na Matriz de Quadrantes do Jupyter)
        forca_ataque_geral = (forca_ataque_mandante + forca_ataque_visitante) / 2.0
        forca_defesa_geral = (forca_defesa_mandante + forca_defesa_visitante) / 2.0

        registros.append(
            {
                "clube": clube,
                "sigla": SIGLA_POR_NOME[clube],
                "pontos": pontos,
                "jogos": j_total,
                "vitorias": v,
                "empates": e,
                "derrotas": d,
                "gols_pro": gp,
                "gols_contra": gc,
                "saldo_gols": sg,
                "aproveitamento": aproveitamento,
                "jogos_casa": j_casa,
                "gp_casa": gp_casa,
                "gc_casa": gc_casa,
                "jogos_fora": j_fora,
                "gp_fora": gp_fora,
                "gc_fora": gc_fora,
                "forca_ataque_mandante": forca_ataque_mandante,
                "forca_defesa_mandante": forca_defesa_mandante,
                "forca_ataque_visitante": forca_ataque_visitante,
                "forca_defesa_visitante": forca_defesa_visitante,
                "forca_ataque_geral": forca_ataque_geral,
                "forca_defesa_geral": forca_defesa_geral,
            }
        )

    df_tabela = pd.DataFrame(registros)

    # Ordenação pelos 4 critérios oficiais da CBF
    df_tabela = df_tabela.sort_values(
        by=["pontos", "vitorias", "saldo_gols", "gols_pro", "clube"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    df_tabela.insert(0, "posicao", range(1, len(df_tabela) + 1))

    return df_tabela


def executar_pipeline_etl() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Executa o pipeline completo de Engenharia de Dados e salva os arquivos em `data/processed/`.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_partidas = carregar_partidas_realizadas()
    df_restantes = carregar_jogos_restantes()
    metricas_liga = calcular_metricas_liga(df_partidas)
    df_tabela = construir_tabela_e_forcas(df_partidas, metricas_liga)

    # Exportação dos artefatos
    df_partidas.to_csv(PROCESSED_DIR / "partidas_limpas.csv", index=False, encoding="utf-8")
    df_restantes.to_csv(PROCESSED_DIR / "jogos_restantes_limpos.csv", index=False, encoding="utf-8")
    df_tabela.to_csv(PROCESSED_DIR / "tabela_atual.csv", index=False, encoding="utf-8")

    with open(PROCESSED_DIR / "metricas_liga.json", "w", encoding="utf-8") as f:
        json.dump(metricas_liga, f, indent=4, ensure_ascii=False)

    return df_partidas, df_restantes, df_tabela, metricas_liga


if __name__ == "__main__":
    partidas, restantes, tabela, metricas = executar_pipeline_etl()
    print("=== PIPELINE ETL CONCLUÍDO COM SUCESSO ===")
    print(f"Partidas realizadas processadas: {len(partidas)}")
    print(f"Jogos restantes mapeados:        {len(restantes)}")
    print(f"Total de jogos do campeonato:    {len(partidas) + len(restantes)}")
    print(f"Média de gols Mandante (Liga):   {metricas['media_gols_mandante']:.4f}")
    print(f"Média de gols Visitante (Liga):  {metricas['media_gols_visitante']:.4f}")
    print("\n=== TABELA DE CLASSIFICAÇÃO ATUAL ===")
    colunas_exibicao = [
        "posicao",
        "clube",
        "pontos",
        "jogos",
        "vitorias",
        "empates",
        "derrotas",
        "gols_pro",
        "gols_contra",
        "saldo_gols",
        "aproveitamento",
    ]
    print(tabela[colunas_exibicao].to_string(index=False))
