/**
 * Frontend Interativo — Preditor do Brasileirão 2026.
 * Layout Editorial Esportivo (inspirado no Simulador do ge.globo e Opta Analyst).
 */

let estadoCalendario = {};
let rodadaAtual = 35;
let placaresManuais = {};
let placaresSimulados = {};
let timerDebounceSimulacao = null;

const CORES_CLUBES = {
  FLA: "#c52613",
  PAL: "#006437",
  CRU: "#003a94",
  FLU: "#8a1538",
  CAP: "#c4161c",
  BAH: "#005ca9",
  CAM: "#111827",
  RBB: "#d30d2b",
  SAN: "#1e293b",
  SAO: "#cc0000",
  BOT: "#111827",
  CFC: "#005426",
  VAS: "#111827",
  MIR: "#eab308",
  VIT: "#dc2626",
  INT: "#e11d48",
  COR: "#111827",
  GRE: "#0284c7",
  REM: "#1e3a8a",
  CHA: "#15803d",
};

function renderizarEscudoHTML(sigla, nomeClube) {
  const corFundo = CORES_CLUBES[sigla] || "#334155";
  const caminhoImg = `/static/img/escudos/${sigla}.png`;
  return `
    <img
      src="${caminhoImg}"
      alt="${nomeClube}"
      class="club-shield"
      onerror="this.style.display='none'; this.nextElementSibling.style.display='inline-flex';"
    />
    <span class="club-fallback" style="display:none; background:${corFundo}; color:#fff; border-color:${corFundo};">
      ${sigla}
    </span>
  `;
}

function classeZonaPosicao(posicao) {
  if (posicao <= 4) return "zone-liberta";
  if (posicao <= 6) return "zone-preliberta";
  if (posicao <= 12) return "zone-sula";
  if (posicao >= 17) return "zone-z4";
  return "zone-neutra";
}

function renderizarMetricaMC(rotulo, valor, tom) {
  const pct = Math.max(0, Math.min(100, Number(valor)));
  return `
    <span class="mc-item mc-tone-${tom}">
      <i class="mc-sq"></i>
      <span class="mc-label">${rotulo}</span>
      <strong class="mc-val">${pct.toFixed(1)}%</strong>
    </span>
  `;
}

function renderizarTabela(linhasTabela, modo = "manual") {
  const tbody = document.getElementById("tbody-classificacao");
  tbody.innerHTML = "";

  linhasTabela.forEach((item) => {
    const tr = document.createElement("tr");
    tr.className = classeZonaPosicao(item.posicao);

    const disputaSulaEZ4 =
      (item.posicao >= 13 && item.posicao <= 16 && (item.prob_sulamericana > 0 || item.prob_rebaixamento > 0)) ||
      (item.prob_sulamericana >= 1.0 && item.prob_rebaixamento >= 1.0);

    let conteudoProjecaoHTML = "";
    if (disputaSulaEZ4) {
      conteudoProjecaoHTML = `
        <div class="mc-inline-dual">
          ${renderizarMetricaMC("Sula", item.prob_sulamericana, "sky")}
          <span class="mc-sep"></span>
          ${renderizarMetricaMC("Z4", item.prob_rebaixamento, "rose")}
        </div>
      `;
    } else {
      const tom = item.cor_coluna_inteligente || "slate";
      conteudoProjecaoHTML = `
        <div class="mc-inline-single">
          ${renderizarMetricaMC(item.rotulo_coluna_inteligente, item.valor_coluna_inteligente, tom)}
        </div>
      `;
    }

    const popoverMC = `
      <div class="mc-tooltip">
        <div class="mc-tooltip-title">${item.clube} — 50.000x Monte Carlo</div>
        <div class="mc-tooltip-row"><span>Título (1º)</span><strong>${item.prob_titulo.toFixed(2)}%</strong></div>
        <div class="mc-tooltip-row"><span>Libertadores (G6)</span><strong>${item.prob_libertadores.toFixed(2)}%</strong></div>
        <div class="mc-tooltip-row"><span>Sul-Americana (7º–12º)</span><strong>${item.prob_sulamericana.toFixed(2)}%</strong></div>
        <div class="mc-tooltip-row"><span>Risco Z4 (17º–20º)</span><strong>${item.prob_rebaixamento.toFixed(2)}%</strong></div>
      </div>
    `;

    tr.innerHTML = `
      <td class="cell-pos">${item.posicao}º</td>
      <td class="cell-club">
        <div class="club-wrap">
          ${renderizarEscudoHTML(item.sigla, item.clube)}
          <span>${item.clube}</span>
        </div>
      </td>
      <td class="cell-pts">${item.pontos}</td>
      <td>${item.jogos}</td>
      <td>${item.vitorias}</td>
      <td>${item.empates}</td>
      <td>${item.derrotas}</td>
      <td>${item.gols_pro}</td>
      <td>${item.gols_contra}</td>
      <td>${item.saldo_gols}</td>
      <td>${Math.round(item.aproveitamento)}</td>
      <td class="cell-smart">
        <div class="smart-cell-wrap">
          ${conteudoProjecaoHTML}
          ${popoverMC}
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });

  const badgeModo = document.getElementById("status-modo-badge");
  const todos38 = linhasTabela.every((r) => r.jogos === 38);
  if (modo === "sortear_dinamico") {
    badgeModo.textContent = "Simulação Estocástica via Monte Carlo";
    badgeModo.classList.add("simulated");
  } else if (modo === "preencher_provaveis" || todos38) {
    badgeModo.textContent = "Projeção Mais Provável via Poisson";
    badgeModo.classList.add("simulated");
  } else {
    const qtdManuais = Object.keys(placaresManuais).length;
    badgeModo.textContent =
      qtdManuais > 0
        ? `Classificação Parcial (+${qtdManuais} palpite(s) manual(is))`
        : "Classificação Atual (339/380 jogos)";
    badgeModo.classList.remove("simulated");
  }

  const contador = document.getElementById("contador-palpites");
  const nManuais = Object.keys(placaresManuais).length;
  contador.textContent = `${nManuais} de 41 jogos futuros travados manualmente`;
}

function renderizarRodada(numeroRodada) {
  rodadaAtual = Number(numeroRodada);
  const jogos = estadoCalendario[String(rodadaAtual)] || [];

  document.getElementById("titulo-rodada").textContent = `${rodadaAtual}ª Rodada`;
  const pendentesNaRodada = jogos.filter((j) => !j.realizado).length;
  document.getElementById("subtitulo-rodada").textContent =
    pendentesNaRodada === 0
      ? "Resultados Oficiais"
      : `${pendentesNaRodada} partida(s) aberta(s)`;

  document.getElementById("btn-rodada-ant").disabled = rodadaAtual <= 1;
  document.getElementById("btn-rodada-prox").disabled = rodadaAtual >= 38;

  document.querySelectorAll(".quick-pill").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.round) === rodadaAtual);
  });

  const container = document.getElementById("lista-partidas-rodada");
  container.innerHTML = "";

  jogos.forEach((jogo) => {
    const card = document.createElement("div");
    const idFut = jogo.id_jogo_futuro;

    const palpiteManual = idFut !== null ? placaresManuais[idFut] : null;
    const palpiteSim = idFut !== null ? placaresSimulados[idFut] : null;

    const ehManual = Array.isArray(palpiteManual) && palpiteManual[0] !== "" && palpiteManual[1] !== "";
    const ehSimulado = !ehManual && Array.isArray(palpiteSim) && palpiteSim[0] !== "" && palpiteSim[1] !== "";

    let classeCard = "match-card";
    if (ehManual) classeCard += " user-edited";
    else if (ehSimulado) classeCard += " sim-filled";
    card.className = classeCard;

    let centroPlacarHTML = "";
    let rodapePoissonHTML = "";

    if (jogo.realizado) {
      centroPlacarHTML = `
        <div class="score-center">
          <span class="score-fixed">${jogo.gols_mandante}</span>
          <span class="score-x">X</span>
          <span class="score-fixed">${jogo.gols_visitante}</span>
        </div>
      `;
    } else {
      const valM = ehManual ? palpiteManual[0] : ehSimulado ? palpiteSim[0] : "";
      const valV = ehManual ? palpiteManual[1] : ehSimulado ? palpiteSim[1] : "";

      centroPlacarHTML = `
        <div class="score-center">
          <input
            type="text"
            inputmode="numeric"
            maxlength="2"
            class="score-input"
            data-id-futuro="${idFut}"
            data-lado="mandante"
            placeholder=""
            value="${valM}"
          />
          <span class="score-x">X</span>
          <input
            type="text"
            inputmode="numeric"
            maxlength="2"
            class="score-input"
            data-id-futuro="${idFut}"
            data-lado="visitante"
            placeholder=""
            value="${valV}"
          />
        </div>
      `;

      rodapePoissonHTML = `
        <div class="poisson-hint">
          <span class="poisson-lambda" title="Expectativa de gols (Lambda Mandante x Lambda Visitante)">
            λ ${jogo.lambda_mandante.toFixed(2)} vs ${jogo.lambda_visitante.toFixed(2)}
          </span>
          <span
            class="poisson-link"
            data-id-futuro="${idFut}"
            data-gm="${jogo.gols_mandante_provavel}"
            data-gv="${jogo.gols_visitante_provavel}"
            title="Aplicar este placar (${jogo.prob_vitoria_mandante.toFixed(0)}% Casa | ${jogo.prob_empate.toFixed(0)}% Empate | ${jogo.prob_vitoria_visitante.toFixed(0)}% Fora)"
          >
            Provável: <strong>${jogo.placar_provavel}</strong>
          </span>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="match-meta">${jogo.data} &bull; ${jogo.horario} | ${jogo.estadio}</div>
      <div class="match-row">
        <div class="team-side home" title="${jogo.mandante}">
          ${renderizarEscudoHTML(jogo.sigla_mandante, jogo.mandante)}
          <span class="team-sigla">${jogo.sigla_mandante}</span>
        </div>
        ${centroPlacarHTML}
        <div class="team-side away" title="${jogo.visitante}">
          <span class="team-sigla">${jogo.sigla_visitante}</span>
          ${renderizarEscudoHTML(jogo.sigla_visitante, jogo.visitante)}
        </div>
      </div>
      ${rodapePoissonHTML}
    `;

    container.appendChild(card);
  });

  // Eventos nos inputs de placar
  container.querySelectorAll(".score-input").forEach((input) => {
    input.addEventListener("input", aoEditarPlacarManual);
  });

  // Evento de clique rápido na sugestão de Poisson de um jogo específico
  container.querySelectorAll(".poisson-link").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idFut = Number(btn.dataset.idFuturo);
      const gm = Number(btn.dataset.gm);
      const gv = Number(btn.dataset.gv);
      placaresManuais[idFut] = [gm, gv];
      renderizarRodada(rodadaAtual);
      enviarSimulacaoServidor("manual");
    });
  });
}

function aoEditarPlacarManual(e) {
  e.target.value = e.target.value.replace(/[^0-9]/g, "");

  const idFut = Number(e.target.dataset.idFuturo);
  const inputsDoJogo = document.querySelectorAll(`.score-input[data-id-futuro="${idFut}"]`);
  let valM = "";
  let valV = "";
  inputsDoJogo.forEach((inp) => {
    if (inp.dataset.lado === "mandante") valM = inp.value.trim();
    if (inp.dataset.lado === "visitante") valV = inp.value.trim();
  });

  const card = e.target.closest(".match-card");
  if (valM !== "" && valV !== "") {
    placaresManuais[idFut] = [Number(valM), Number(valV)];
    delete placaresSimulados[idFut];
    if (card) {
      card.classList.remove("sim-filled");
      card.classList.add("user-edited");
    }
    agendarSimulacaoManual();
  } else if (valM === "" && valV === "") {
    delete placaresManuais[idFut];
    delete placaresSimulados[idFut];
    if (card) {
      card.classList.remove("user-edited");
      card.classList.remove("sim-filled");
    }
    agendarSimulacaoManual();
  }
}

function agendarSimulacaoManual() {
  if (timerDebounceSimulacao) clearTimeout(timerDebounceSimulacao);
  timerDebounceSimulacao = setTimeout(() => {
    enviarSimulacaoServidor("manual");
  }, 100);
}

async function enviarSimulacaoServidor(modo = "manual") {
  try {
    const resp = await fetch("/api/simular", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        placares_manuais: placaresManuais,
        placares_simulados: placaresSimulados,
        modo: modo,
      }),
    });
    const dados = await resp.json();

    if (modo === "sortear_dinamico" || modo === "preencher_provaveis") {
      placaresSimulados = {};
      const exibidos = dados.placares_exibidos || {};
      Object.keys(exibidos).forEach((k) => {
        const idNum = Number(k);
        if (!(idNum in placaresManuais)) {
          placaresSimulados[idNum] = exibidos[k];
        }
      });
      renderizarRodada(rodadaAtual);
    }

    renderizarTabela(dados.tabela, modo);
  } catch (err) {
    console.error("Erro ao simular:", err);
  }
}

async function carregarEstadoInicial() {
  const resp = await fetch("/api/estado");
  const dados = await resp.json();
  estadoCalendario = dados.rodadas;
  rodadaAtual = dados.rodada_inicial || 35;
  renderizarTabela(dados.tabela, "manual");
  renderizarRodada(rodadaAtual);
}

document.addEventListener("DOMContentLoaded", () => {
  carregarEstadoInicial();

  document.getElementById("btn-rodada-ant").addEventListener("click", () => {
    if (rodadaAtual > 1) renderizarRodada(rodadaAtual - 1);
  });

  document.getElementById("btn-rodada-prox").addEventListener("click", () => {
    if (rodadaAtual < 38) renderizarRodada(rodadaAtual + 1);
  });

  document.querySelectorAll(".quick-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      renderizarRodada(Number(btn.dataset.round));
    });
  });

  document.getElementById("btn-preencher-poisson").addEventListener("click", () => {
    enviarSimulacaoServidor("preencher_provaveis");
  });

  document.getElementById("btn-simular-mc").addEventListener("click", () => {
    enviarSimulacaoServidor("sortear_dinamico");
  });

  document.getElementById("btn-limpar").addEventListener("click", () => {
    placaresManuais = {};
    placaresSimulados = {};
    carregarEstadoInicial();
  });

  const btnTema = document.getElementById("btn-tema");
  btnTema.addEventListener("click", () => {
    const atual = document.documentElement.getAttribute("data-theme");
    if (atual === "dark") {
      document.documentElement.removeAttribute("data-theme");
      localStorage.setItem("tema_preditor_2026", "light");
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
      localStorage.setItem("tema_preditor_2026", "dark");
    }
  });
});
