"""
Módulo de Visualização Executiva e BI (Matplotlib & Seaborn).

Gera os 4 gráficos principais do projeto em alta resolução (300 DPI):
1. Corrida pelo Título (% de probabilidade de título após 10.000 simulações).
2. Termômetro do Z4 (% de risco de rebaixamento dos clubes ameaçados).
3. Distribuição de Pontuação Final (Boxplot horizontal das 10.000 simulações para os 20 clubes).
4. Matriz de Quadrantes: Eficiência Ofensiva (Força de Ataque Geral) vs. Eficiência Defensiva (Força de Defesa Geral).
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "data" / "processed" / "figures"

# Configuração estética global
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def plotar_corrida_titulo(
    df_resumo: pd.DataFrame,
    salvar_em: Optional[Path] = FIGURES_DIR / "01_corrida_titulo.png",
    mostrar: bool = False,
) -> plt.Figure:
    """
    Gráfico 1: Barras horizontais destacando os candidatos ao título (Top 5 da tabela
    ou clubes com probabilidade > 0%).
    """
    df_top = df_resumo.head(5).copy()
    df_top = df_top.sort_values("prob_titulo", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    cores = ["#94A3B8" if p < 1.0 else ("#F59E0B" if p < 50.0 else "#10B981") for p in df_top["prob_titulo"]]
    barras = ax.barh(
        df_top["clube"],
        df_top["prob_titulo"],
        color=cores,
        height=0.58,
        edgecolor="#0F172A",
        linewidth=0.6,
    )

    for barra, (_, row) in zip(barras, df_top.iterrows()):
        largura = barra.get_width()
        pts_atual = int(row["pontos"])
        ax.text(
            largura + 1.5,
            barra.get_y() + barra.get_height() / 2,
            f"{largura:.2f}%  (Proj: {pts_atual} pts)",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1E293B",
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Probabilidade de Título (%) — 10.000 Simulações de Monte Carlo", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title(
        "Corrida pelo Título — Brasileirão Série A 2026 (Reta Final: 34ª a 38ª Rodada)",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=15,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)
    sns.despine(top=True, right=True)

    plt.tight_layout()
    if salvar_em:
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=300, bbox_inches="tight")
    if not mostrar:
        plt.close(fig)
    return fig


def plotar_termometro_z4(
    df_resumo: pd.DataFrame,
    salvar_em: Optional[Path] = FIGURES_DIR / "02_termometro_z4.png",
    mostrar: bool = False,
) -> plt.Figure:
    """
    Gráfico 2: Termômetro da Luta contra o Rebaixamento (clubes com risco de Z4 > 0.5%).
    """
    df_z4 = df_resumo[df_resumo["prob_rebaixamento"] >= 0.5].copy()
    df_z4 = df_z4.sort_values("prob_rebaixamento", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    cores = []
    for r in df_z4["prob_rebaixamento"]:
        if r >= 75.0:
            cores.append("#BE123C")  # Vermelho escuro crítico
        elif r >= 35.0:
            cores.append("#F43F5E")  # Vermelho alerta
        elif r >= 15.0:
            cores.append("#FB923C")  # Laranja atenção
        else:
            cores.append("#FBBF24")  # Amarelo risco moderado/baixo

    barras = ax.barh(
        df_z4["clube"],
        df_z4["prob_rebaixamento"],
        color=cores,
        height=0.62,
        edgecolor="#0F172A",
        linewidth=0.6,
    )

    for barra, (_, row) in zip(barras, df_z4.iterrows()):
        largura = barra.get_width()
        ax.text(
            largura + 1.5,
            barra.get_y() + barra.get_height() / 2,
            f"{largura:.1f}%  (Proj: {int(row['pontos'])} pts)",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#1E293B",
        )

    ax.axvline(50.0, color="#64748B", linestyle=":", linewidth=1.2, label="Limiar de 50% de Risco")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probabilidade de Rebaixamento ao Z4 (%) — 10.000 Simulações", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title(
        "Termômetro do Rebaixamento (Z4) — Brasileirão Série A 2026",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=15,
    )
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)
    sns.despine(top=True, right=True)

    plt.tight_layout()
    if salvar_em:
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=300, bbox_inches="tight")
    if not mostrar:
        plt.close(fig)
    return fig


def plotar_distribuicao_pontos(
    df_resumo: pd.DataFrame,
    matriz_pontos: np.ndarray,
    clubes_ordem: List[str],
    salvar_em: Optional[Path] = FIGURES_DIR / "03_distribuicao_pontos.png",
    mostrar: bool = False,
) -> plt.Figure:
    """
    Gráfico 3: Boxplot horizontal da distribuição probabilística da pontuação final
    dos 20 clubes nas 10.000 simulações de Monte Carlo.
    """
    mapa_idx = {c: i for i, c in enumerate(clubes_ordem)}
    clubes_ordenados = df_resumo["clube"].tolist()[::-1]  # Do 20º ao 1º para o topo ficar em cima
    dados_boxplot = [matriz_pontos[:, mapa_idx[c]] for c in clubes_ordenados]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    bp = ax.boxplot(
        dados_boxplot,
        orientation="horizontal",
        tick_labels=clubes_ordenados,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0F172A", "linewidth": 1.8},
        whiskerprops={"color": "#475569", "linewidth": 1.0},
        capprops={"color": "#475569", "linewidth": 1.0},
    )

    # Colore caixas conforme a posição projetada (Z4, Sul-Americana, Libertadores, Líderes)
    n_total = len(clubes_ordenados)
    for idx_plot, patch in enumerate(bp["boxes"]):
        pos_tabela = n_total - idx_plot  # 1 a 20
        if pos_tabela <= 4:
            patch.set_facecolor("#10B981")
            patch.set_alpha(0.8)
        elif pos_tabela <= 6:
            patch.set_facecolor("#3B82F6")
            patch.set_alpha(0.8)
        elif pos_tabela <= 12:
            patch.set_facecolor("#38BDF8")
            patch.set_alpha(0.75)
        elif pos_tabela >= 17:
            patch.set_facecolor("#F43F5E")
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor("#94A3B8")
            patch.set_alpha(0.7)

    # Linha mágica histórica dos 45 pontos contra o rebaixamento
    ax.axvline(45, color="#E11D48", linestyle="--", linewidth=1.3, label="Referência 45 pts (Corte Z4)")

    ax.set_xlabel("Pontuação Final ao Término da 38ª Rodada (Intervalo Interquartil e Mediana)", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title(
        "Distribuição Probabilística da Pontuação Final — 10.000 Campeonatos Simulados",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=15,
    )
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)
    sns.despine(top=True, right=True)

    plt.tight_layout()
    if salvar_em:
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=300, bbox_inches="tight")
    if not mostrar:
        plt.close(fig)
    return fig


def plotar_matriz_quadrantes_forcas(
    df_tabela: pd.DataFrame,
    salvar_em: Optional[Path] = FIGURES_DIR / "04_matriz_eficiencia.png",
    mostrar: bool = False,
) -> plt.Figure:
    """
    Gráfico 4: Matriz de Quadrantes (Eficiência Ofensiva vs. Eficiência Defensiva).
    - Eixo X: Força de Ataque Geral (> 1.0 = marca mais gols que a média da liga)
    - Eixo Y (invertido): Força de Defesa Geral (< 1.0 = sofre menos gols que a média da liga, defesa sólida no topo)
    """
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    x = df_tabela["forca_ataque_geral"]
    y = df_tabela["forca_defesa_geral"]

    scatter = ax.scatter(
        x,
        y,
        c=df_tabela["pontos"],
        cmap="viridis",
        s=150,
        edgecolors="#0F172A",
        linewidth=1.0,
        zorder=4,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Pontos Conquistados (Até a 34ª Rodada)", fontsize=10, fontweight="bold", color="#334155")

    # Eixos médios da liga (1.0, 1.0)
    ax.axvline(1.0, color="#475569", linestyle="--", linewidth=1.2, zorder=2)
    ax.axhline(1.0, color="#475569", linestyle="--", linewidth=1.2, zorder=2)

    # Margens amplas para que rótulos e caixas de quadrantes não colidam com pontos extremos
    x_min, x_max = float(x.min()) - 0.12, float(x.max()) + 0.18
    y_min, y_max = float(y.min()) - 0.15, float(y.max()) + 0.15
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invertido: menor y (defesa mais sólida) no topo

    # Rótulos enxutos (Sigla) com pequenos ajustes de offset para pontos próximos
    offsets_custom = {
        "CAM": (7, 5),
        "RBB": (7, -8),
        "SAO": (-32, -4),
        "INT": (7, 4),
        "COR": (-32, 4),
        "CAP": (-32, -4),
        "FLU": (7, -4),
        "GRE": (7, -8),
        "MIR": (7, 5),
        "FLA": (-34, -4),
    }
    for _, row in df_tabela.iterrows():
        sigla = row["sigla"]
        dx, dy = offsets_custom.get(sigla, (7, -4))
        ax.annotate(
            sigla,
            (row["forca_ataque_geral"], row["forca_defesa_geral"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
            zorder=5,
        )

    # Rótulos explicativos dos 4 quadrantes posicionados nos cantos com folga
    ax.text(
        x_max - 0.02,
        y_min + 0.03,
        "ELITE (Ataque Forte & Defesa Sólida)",
        ha="right",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#047857",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#D1FAE5", "edgecolor": "#10B981", "alpha": 0.9},
        zorder=3,
    )
    ax.text(
        x_min + 0.02,
        y_min + 0.03,
        "RETRANCADOS (Ataque Fraco & Defesa Sólida)",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#0369A1",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#E0F2FE", "edgecolor": "#38BDF8", "alpha": 0.9},
        zorder=3,
    )
    ax.text(
        x_max - 0.02,
        y_max - 0.03,
        "FRANCO-ATIRADORES (Ataque Forte & Defesa Vulnerável)",
        ha="right",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#B45309",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FEF3C7", "edgecolor": "#F59E0B", "alpha": 0.9},
        zorder=3,
    )
    ax.text(
        x_min + 0.02,
        y_max - 0.03,
        "CRÍTICO (Ataque Fraco & Defesa Vulnerável)",
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#BE123C",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFE4E6", "edgecolor": "#F43F5E", "alpha": 0.9},
        zorder=3,
    )

    ax.set_xlabel("Índice de Força Relativa de Ataque (Acima de 1.0 = Melhor que a Média da Liga)", fontsize=11, fontweight="bold", color="#334155")
    ax.set_ylabel("Índice de Força Relativa de Defesa (Abaixo de 1.0 = Sofre Menos Gols que a Média)", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title(
        "Matriz de Quadrantes: Eficiência Ofensiva vs. Defensiva — Brasileirão 2026",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
        pad=15,
    )
    ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E1")

    plt.tight_layout()
    if salvar_em:
        salvar_em.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(salvar_em, dpi=300, bbox_inches="tight")
    if not mostrar:
        plt.close(fig)
    return fig


def gerar_todos_graficos_bi(
    df_tabela: pd.DataFrame,
    resultado_mc: Dict[str, object],
) -> Dict[str, Path]:
    """
    Gera e exporta os 4 gráficos executivos de BI em `data/processed/figures/`.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    caminhos = {
        "corrida_titulo": FIGURES_DIR / "01_corrida_titulo.png",
        "termometro_z4": FIGURES_DIR / "02_termometro_z4.png",
        "distribuicao_pontos": FIGURES_DIR / "03_distribuicao_pontos.png",
        "matriz_eficiencia": FIGURES_DIR / "04_matriz_eficiencia.png",
    }

    df_resumo = resultado_mc["df_resumo"]
    matriz_pontos = resultado_mc["matriz_pontos_simulados"]
    clubes_ordem = resultado_mc["clubes_ordem"]

    plotar_corrida_titulo(df_resumo, salvar_em=caminhos["corrida_titulo"], mostrar=False)
    plotar_termometro_z4(df_resumo, salvar_em=caminhos["termometro_z4"], mostrar=False)
    plotar_distribuicao_pontos(
        df_resumo, matriz_pontos, clubes_ordem, salvar_em=caminhos["distribuicao_pontos"], mostrar=False
    )
    plotar_matriz_quadrantes_forcas(df_tabela, salvar_em=caminhos["matriz_eficiencia"], mostrar=False)

    return caminhos


if __name__ == "__main__":
    from src.data_engineering.etl import executar_pipeline_etl
    from src.data_science.monte_carlo import executar_simulacao_monte_carlo
    from src.data_science.poisson import prever_todos_jogos_restantes

    _, df_restantes, df_tabela, metricas = executar_pipeline_etl()
    df_previstos = prever_todos_jogos_restantes(df_restantes, df_tabela, metricas)
    resultado_mc = executar_simulacao_monte_carlo(df_tabela, df_previstos, seed=42)

    arquivos = gerar_todos_graficos_bi(df_tabela, resultado_mc)
    print("=== GRÁFICOS DE BI GERADOS COM SUCESSO ===")
    for nome, cam in arquivos.items():
        print(f"- {nome}: {cam}")
