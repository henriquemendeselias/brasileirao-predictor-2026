"""
Dicionário mestre de padronização dos 20 clubes da Série A 2026.

Mapeia a nomenclatura bruta presente em `data/raw/BRA.csv` e `data/raw/jogos_restantes.csv`
para:
- nome: Nome oficial em português com acentuação (padrão ge.globo)
- sigla: Sigla oficial de 3 letras
- estadio: Estádio sede padrão do clube como mandante
"""

from typing import Dict, TypedDict


class InfoClube(TypedDict):
    nome: str
    sigla: str
    estadio: str


MAPA_CLUBES: Dict[str, InfoClube] = {
    "Athletico-PR": {
        "nome": "Athletico-PR",
        "sigla": "CAP",
        "estadio": "Arena da Baixada",
    },
    "Atletico-MG": {
        "nome": "Atlético-MG",
        "sigla": "CAM",
        "estadio": "Arena MRV",
    },
    "Bahia": {
        "nome": "Bahia",
        "sigla": "BAH",
        "estadio": "Arena Fonte Nova",
    },
    "Botafogo RJ": {
        "nome": "Botafogo",
        "sigla": "BOT",
        "estadio": "Engenhão",
    },
    "Bragantino": {
        "nome": "Red Bull Bragantino",
        "sigla": "RBB",
        "estadio": "Cícero de Souza Marques",
    },
    "Chapecoense-SC": {
        "nome": "Chapecoense",
        "sigla": "CHA",
        "estadio": "Arena Condá",
    },
    "Corinthians": {
        "nome": "Corinthians",
        "sigla": "COR",
        "estadio": "Neo Química Arena",
    },
    "Coritiba": {
        "nome": "Coritiba",
        "sigla": "CFC",
        "estadio": "Couto Pereira",
    },
    "Cruzeiro": {
        "nome": "Cruzeiro",
        "sigla": "CRU",
        "estadio": "Mineirão",
    },
    "Flamengo RJ": {
        "nome": "Flamengo",
        "sigla": "FLA",
        "estadio": "Maracanã",
    },
    "Fluminense": {
        "nome": "Fluminense",
        "sigla": "FLU",
        "estadio": "Maracanã",
    },
    "Gremio": {
        "nome": "Grêmio",
        "sigla": "GRE",
        "estadio": "Arena do Grêmio",
    },
    "Internacional": {
        "nome": "Internacional",
        "sigla": "INT",
        "estadio": "Beira-Rio",
    },
    "Mirassol": {
        "nome": "Mirassol",
        "sigla": "MIR",
        "estadio": "Campos Maia",
    },
    "Palmeiras": {
        "nome": "Palmeiras",
        "sigla": "PAL",
        "estadio": "Allianz Parque",
    },
    "Remo": {
        "nome": "Remo",
        "sigla": "REM",
        "estadio": "Mangueirão",
    },
    "Santos": {
        "nome": "Santos",
        "sigla": "SAN",
        "estadio": "Vila Belmiro",
    },
    "Sao Paulo": {
        "nome": "São Paulo",
        "sigla": "SAO",
        "estadio": "Morumbis",
    },
    "Vasco": {
        "nome": "Vasco da Gama",
        "sigla": "VAS",
        "estadio": "São Januário",
    },
    "Vitoria": {
        "nome": "Vitória",
        "sigla": "VIT",
        "estadio": "Barradão",
    },
}


# Índices auxiliares por Nome Oficial (pós-padronização)
SIGLA_POR_NOME: Dict[str, str] = {info["nome"]: info["sigla"] for info in MAPA_CLUBES.values()}
ESTADIO_POR_NOME: Dict[str, str] = {info["nome"]: info["estadio"] for info in MAPA_CLUBES.values()}


def padronizar_nome_clube(nome_bruto: str) -> str:
    """Retorna o nome oficial do clube a partir do nome bruto do CSV."""
    nome_limpo = nome_bruto.strip()
    if nome_limpo not in MAPA_CLUBES:
        raise ValueError(f"Clube não reconhecido no dicionário mestre: '{nome_bruto}'")
    return MAPA_CLUBES[nome_limpo]["nome"]
