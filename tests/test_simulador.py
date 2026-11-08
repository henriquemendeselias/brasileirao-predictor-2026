"""
Suíte de Testes Unitários e de Integração para o Simulador do Brasileirão 2026.

Compatível tanto com `python -m unittest` (biblioteca padrão, sem dependências extras)
quanto com `pytest`.

Valida:
1. Engenharia de Dados (ETL):
   - Total de jogos realizados (338) + restantes (42) = 380 jogos (19 rodadas * 2 turnos * 10 jogos).
   - 20 clubes únicos e padronizados, cada um com exatamente 38 jogos na soma total (realizados + restantes).
   - Integridade matemática da tabela: soma de vitórias == soma de derrotas, soma de GP == soma de GC, saldo total == 0.
2. Ciência de Dados (Distribuição de Poisson):
   - Positividade estrita de lambda_mandante e lambda_visitante.
   - Matriz de probabilidades 9x9 (k_max=8) com soma próxima de 1.0 (> 0.999) e soma 1X2 normalizada == 100%.
3. Ciência de Dados (Simulação de Monte Carlo):
   - Soma das probabilidades de título entre os 20 clubes == 100%.
   - Soma das probabilidades de Libertadores (G6) == 600%, Sul-Americana == 600%, Z4 == 400%.
   - Respeito absoluto aos placares travados manualmente pelo usuário (`placares_manuais`).
"""

import unittest
import numpy as np

from src.data_engineering.etl import (
    calcular_metricas_liga,
    carregar_jogos_restantes,
    carregar_partidas_realizadas,
    construir_tabela_e_forcas,
)
from src.data_science.monte_carlo import executar_simulacao_monte_carlo
from src.data_science.poisson import (
    K_MAX_DEFAULT,
    prever_confronto_poisson,
    prever_todos_jogos_restantes,
)


class TestSimuladorBrasileirao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_partidas = carregar_partidas_realizadas()
        cls.df_restantes = carregar_jogos_restantes()
        cls.metricas = calcular_metricas_liga(cls.df_partidas)
        cls.df_tabela = construir_tabela_e_forcas(cls.df_partidas, cls.metricas)

    def test_01_integridade_campeonato_380_jogos(self):
        """Garante que os jogos realizados (339) + restantes (41) fecham os 380 jogos de 20 clubes."""
        self.assertEqual(len(self.df_partidas), 339)
        self.assertEqual(len(self.df_restantes), 41)
        self.assertEqual(len(self.df_partidas) + len(self.df_restantes), 380)
        self.assertEqual(len(self.df_tabela), 20)

        for _, row in self.df_tabela.iterrows():
            clube = row["clube"]
            jogos_realizados = int(row["jogos"])
            jogos_futuros = int(
                (self.df_restantes["mandante"] == clube).sum()
                + (self.df_restantes["visitante"] == clube).sum()
            )
            self.assertEqual(
                jogos_realizados + jogos_futuros,
                38,
                f"Clube {clube} não soma 38 jogos!",
            )

    def test_02_consistencia_matematica_tabela(self):
        """Valida invariantes da tabela de classificação (V == D, GP == GC, Soma SG == 0, P == 3V + E)."""
        self.assertEqual(
            int(self.df_tabela["vitorias"].sum()),
            int(self.df_tabela["derrotas"].sum()),
        )
        self.assertEqual(int(self.df_tabela["empates"].sum()) % 2, 0)
        self.assertEqual(
            int(self.df_tabela["gols_pro"].sum()),
            int(self.df_tabela["gols_contra"].sum()),
        )
        self.assertEqual(
            int(self.df_tabela["gols_pro"].sum()),
            self.metricas["total_gols_campeonato"],
        )
        self.assertEqual(int(self.df_tabela["saldo_gols"].sum()), 0)

        pontos_calculados = (self.df_tabela["vitorias"] * 3) + self.df_tabela["empates"]
        self.assertTrue((self.df_tabela["pontos"] == pontos_calculados).all())

    def test_03_modelo_poisson_propriedades_probabilisticas(self):
        """Valida que os lambdas são positivos, a matriz é 9x9 e as probabilidades somam 100%."""
        res = prever_confronto_poisson(
            "Flamengo", "Grêmio", self.df_tabela, self.metricas, k_max=K_MAX_DEFAULT
        )
        self.assertGreater(res["lambda_mandante"], 0.0)
        self.assertGreater(res["lambda_visitante"], 0.0)

        matriz = res["matriz_placares"]
        self.assertEqual(matriz.shape, (K_MAX_DEFAULT + 1, K_MAX_DEFAULT + 1))
        self.assertTrue(np.all(matriz >= 0.0))
        self.assertGreater(res["soma_prob_matriz"], 0.999)

        soma_1x2 = (
            res["prob_vitoria_mandante"]
            + res["prob_empate"]
            + res["prob_vitoria_visitante"]
        )
        self.assertAlmostEqual(soma_1x2, 100.0, delta=0.05)

    def test_04_monte_carlo_invariantes_e_placares_manuais(self):
        """Valida invariantes estocásticos de Monte Carlo e o travamento de placares manuais."""
        df_previstos = prever_todos_jogos_restantes(
            self.df_restantes, self.df_tabela, self.metricas
        )
        resultado = executar_simulacao_monte_carlo(
            df_tabela_atual=self.df_tabela,
            df_jogos_previstos=df_previstos,
            n_simulacoes=2000,
            seed=42,
        )
        df_res = resultado["df_resumo"]

        self.assertTrue((df_res["jogos"] == 38).all())
        self.assertAlmostEqual(float(df_res["prob_titulo"].sum()), 100.0, delta=0.1)
        self.assertAlmostEqual(float(df_res["prob_libertadores"].sum()), 600.0, delta=0.1)
        self.assertAlmostEqual(float(df_res["prob_sulamericana"].sum()), 600.0, delta=0.1)
        self.assertAlmostEqual(float(df_res["prob_rebaixamento"].sum()), 400.0, delta=0.1)

        # Teste de travamento total: se todos os 42 jogos forem fixados manualmente, a probabilidade do 1º é 100%
        todos_travados = {
            int(row["id_jogo"]): (1, 0) for _, row in df_previstos.iterrows()
        }
        res_travado = executar_simulacao_monte_carlo(
            df_tabela_atual=self.df_tabela,
            df_jogos_previstos=df_previstos,
            placares_manuais=todos_travados,
            n_simulacoes=500,
            seed=42,
        )
        df_travado = res_travado["df_resumo"]
        self.assertEqual(float(df_travado.iloc[0]["prob_titulo"]), 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
