/* Decision Lab typed-question UI: SSE-driven single-state evaluation. */

const API_BASE = "";
const BAR_MAX_WIDTH = 220;   // px for probability bars

const els = {
  status: document.getElementById("status"),
  agentCards: document.getElementById("agent-cards"),
  stateSelect: document.getElementById("state-select"),
  customText: document.getElementById("custom-text"),
  runBtn: document.getElementById("run-btn"),
  stateTypeBadge: document.getElementById("state-type-badge"),
  stateText: document.getElementById("state-text"),
  results: document.getElementById("results"),
  summary: document.getElementById("summary"),
};

const state = {
  agents: [],
  questions: {},        // {qid: {type, options?/levels?/question}}
  selectedAgent: null,
  running: false,
  loadedStates: [],     // cached state list for lookup
};

/* ---------- server status ---------- */

async function pollStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const body = await res.json();
    setReady(body.ready);
  } catch {
    setReady(false);
  }
}

function setReady(ready) {
  els.status.classList.toggle("ready", ready);
  els.status.classList.toggle("down", !ready);
  els.status.title = ready ? "server ready" : "server unavailable";
  els.runBtn.disabled = !ready || state.running;
  els.runDynamicBtn.disabled = !ready;
  if (ready && els.agentCards.querySelector(".loading")) {
    loadAgents();
    loadStates();
    loadQuestions();
  }
}

/* ---------- model selection ---------- */

async function loadAgents() {
  try {
    const res = await fetch(`${API_BASE}/api/agents`);
    const body = await res.json();
    state.agents = body.agents;
    renderAgentCards();
  } catch (err) {
    els.agentCards.innerHTML = `<p class="loading">Failed to load models: ${err}</p>`;
  }
}

function renderAgentCards() {
  els.agentCards.innerHTML = "";
  for (const agent of state.agents) {
    const label = document.createElement("label");
    label.className = "agent-card";
    label.innerHTML = `
      <input type="radio" name="agent" value="${agent.agent_id}">
      <div class="name">${agent.name}</div>
      <div class="desc">${agent.description}</div>
    `;
    label.addEventListener("click", () => selectAgent(agent.agent_id));
    els.agentCards.appendChild(label);
  }
  selectAgent(state.agents[0]?.agent_id);
}

function selectAgent(agentId) {
  state.selectedAgent = agentId;
  for (const card of els.agentCards.querySelectorAll(".agent-card")) {
    const input = card.querySelector("input");
    card.classList.toggle("selected", input.value === agentId);
    input.checked = input.value === agentId;
  }
}

/* ---------- states + question bank ---------- */

async function loadStates() {
  try {
    const res = await fetch(`${API_BASE}/api/states`);
    const body = await res.json();
    els.stateSelect.innerHTML = "";
    for (const s of body.states) {
      const opt = document.createElement("option");
      opt.value = s.doc_id;
      opt.textContent = `#${s.doc_id} · ${s.state_type}`;
      els.stateSelect.appendChild(opt);
    }
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = "Custom text (paste below)";
    els.stateSelect.appendChild(customOpt);
    state.loadedStates = body.states;
    if (body.states.length > 0) renderState(body.states[0]);
  } catch (err) {
    els.stateSelect.innerHTML = `<option>Failed to load states: ${err}</option>`;
  }
}

function toggleCustomText() {
  const isCustom = els.stateSelect.value === "custom";
  els.customText.classList.toggle("hidden", !isCustom);
}

function renderSelectedState() {
  const val = els.stateSelect.value;
  if (val === "custom") {
    showCustomState();
    return;
  }
  const docId = Number(val);
  const s = state.loadedStates.find(st => st.doc_id === docId);
  if (s) renderState(s);
}

async function loadQuestions() {
  try {
    const res = await fetch(`${API_BASE}/api/questions`);
    const body = await res.json();
    state.questions = body.questions;
  } catch {
    state.questions = {};
  }
}

function questionType(qid) {
  return state.questions[qid]?.type ?? "choice";
}

/* ---------- state panel ---------- */

function renderState(s) {
  els.stateText.classList.remove("loading");
  els.stateText.textContent = s.text;
  els.stateTypeBadge.textContent = s.state_type;
  els.stateTypeBadge.classList.remove("hidden");
}

function showCustomState() {
  const text = els.customText.value.trim();
  els.stateText.classList.remove("loading");
  els.stateText.textContent = text || "(empty — type some text above)";
  els.stateTypeBadge.textContent = "custom";
  els.stateTypeBadge.classList.remove("hidden");
}

function currentStatePayload() {
  if (els.stateSelect.value === "custom") {
    return { custom_text: els.customText.value };
  }
  return { doc_id: Number(els.stateSelect.value) };
}

/* ---------- results rendering ---------- */

function clearResults() {
  els.results.innerHTML = "";
  els.summary.classList.add("hidden");
}

function renderAgentResult(out) {
  const card = document.createElement("div");
  card.className = "agent-result";

  const acc = out.mean_accuracy;
  const accText = acc === null ? "no gold labels" : `${(acc * 100).toFixed(1)}% correct`;
  const accCls = acc === null ? "" : (acc >= 0.75 ? "mark-good" : (acc >= 0.5 ? "" : "mark-bad"));

  const header = document.createElement("div");
  header.className = "agent-result-header";
  header.innerHTML = `
    <div class="name">${escapeHtml(out.agent_name)}</div>
    <div class="meta">${accText} · ${out.latency_ms} ms · parse failures: ${out.parse_failures}</div>
  `;
  card.appendChild(header);

  const table = document.createElement("table");
  table.className = "question-table";
  table.innerHTML = `
    <thead>
      <tr><th>Question</th><th>Type</th><th>Predicted</th><th>Distribution</th><th>Conf.</th><th>Gold</th><th></th></tr>
    </thead>`;
  const tbody = document.createElement("tbody");
  for (const q of out.questions) {
    tbody.appendChild(renderQuestionRow(q));
  }
  table.appendChild(tbody);
  card.appendChild(table);

  if (out.raw_output) {
    const raw = document.createElement("details");
    raw.className = "raw-output-details";
    raw.innerHTML = `<summary>LM raw output</summary><pre>${escapeHtml(out.raw_output)}</pre>`;
    card.appendChild(raw);
  }

  els.results.appendChild(card);
}

function renderQuestionRow(q) {
  const tr = document.createElement("tr");
  if (q.correct === true) tr.classList.add("mark-good-row");
  else if (q.correct === false) tr.classList.add("mark-bad-row");

  const mark = q.correct === null ? "" : (q.correct ? "✓" : "✗");
  const cls = q.correct === false ? "mark-bad" : "mark-good";
  const conf = q.confidence === null || q.confidence === undefined
    ? "—" : q.confidence.toFixed(3);

  tr.innerHTML = `
    <td>${escapeHtml(q.question_id)}</td>
    <td><span class="type-badge type-${q.question_type}">${q.question_type}</span></td>
    <td class="predicted">${escapeHtml(q.predicted_label)}</td>
    <td class="dist-cell"></td>
    <td>${conf}</td>
    <td class="gold">${escapeHtml(q.gold_label)}</td>
    <td class="${cls}">${mark}</td>
  `;
  tr.querySelector(".dist-cell").appendChild(renderDistribution(q));
  return tr;
}

function renderDistribution(q) {
  const wrap = document.createElement("div");
  wrap.className = "dist";
  if (!q.distribution) {
    wrap.innerHTML = `<span class="dist-none">—</span>`;
    return wrap;
  }
  const entries = Object.entries(q.distribution)
    .sort((a, b) => b[1] - a[1]);
  for (const [label, prob] of entries) {
    const row = document.createElement("div");
    row.className = "dist-row";
    const pct = (prob * 100).toFixed(1);
    row.innerHTML = `
      <span class="dist-label">${escapeHtml(label)}</span>
      <span class="dist-bar-track"><span class="dist-bar" style="width:${(prob * BAR_MAX_WIDTH).toFixed(0)}px"></span></span>
      <span class="dist-pct">${pct}%</span>
    `;
    wrap.appendChild(row);
  }
  if (isNearUniform(q.distribution)) {
    const warn = document.createElement("div");
    warn.className = "dist-warning";
    warn.textContent = "near-uniform — answer unreliable (state text likely outside the trained shape/domain)";
    wrap.appendChild(warn);
  }
  return wrap;
}

// A distribution whose top probability is barely above uniform carries no
// decision signal — the head fell back to mush (e.g. unseen state shape or
// out-of-domain options). Flag it so the UI never presents mush as an answer.
// Margin below the top option: uniform = 1/n, mush cutoff = halfway to certain.
function isNearUniform(distribution) {
  const probs = Object.values(distribution);
  const n = probs.length;
  if (n < 2) return false;
  const top = Math.max(...probs);
  return top < 1 / n + 0.5 * (1 - 1 / n);
}

function showSummary(agentOutputs) {
  const judged = agentOutputs.filter((o) => o.mean_accuracy !== null);
  if (judged.length === 0) {
    els.summary.className = "summary success";
    els.summary.textContent = "Done. (No gold labels — custom text is scored by inspection only.)";
    return;
  }
  const best = judged.reduce((a, b) => (b.mean_accuracy > a.mean_accuracy ? b : a));
  els.summary.className = "summary success";
  els.summary.innerHTML =
    `Best: <strong>${escapeHtml(best.agent_name)}</strong> — ` +
    `${(best.mean_accuracy * 100).toFixed(1)}% correct in ${best.latency_ms} ms`;
}

function showError(message) {
  els.summary.className = "summary failure";
  els.summary.innerHTML = `<strong>Error:</strong> ${escapeHtml(message)}`;
  els.summary.classList.remove("hidden");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

/* ---------- SSE consumption ---------- */

async function postSSE(path, body, onEvent) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = parseSSEBlock(block);
      if (event) onEvent(event);
    }
  }
}

function parseSSEBlock(block) {
  let event = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  return data ? { event, data: JSON.parse(data) } : null;
}

/* ---------- run ---------- */

async function runOne() {
  if (!state.selectedAgent || state.running) return;
  state.running = true;
  setButtonsDisabled(true);
  clearResults();
  if (els.stateSelect.value === "custom") showCustomState();

  try {
    await postSSE("/api/run", {
      agent_id: state.selectedAgent,
      ...currentStatePayload(),
    }, (event) => {
      if (event.event === "start") {
        renderState(event.data.state);
      } else if (event.event === "result") {
        renderAgentResult(event.data);
        showSummary([event.data]);
      }
    });
  } catch (err) {
    showError(err.message);
  } finally {
    state.running = false;
    setButtonsDisabled(false);
  }
}

function setButtonsDisabled(disabled) {
  els.runBtn.disabled = disabled || !state.selectedAgent;
}

/* ---------- init ---------- */

els.runBtn.addEventListener("click", runOne);
els.customText.addEventListener("input", showCustomState);
els.stateSelect.addEventListener("change", () => {
  toggleCustomText();
  renderSelectedState();
});

/* ---------- dynamic question builder ---------- */

const DYNAMIC_API_BASE = `${API_BASE}/api/decide-dynamic`;
const QUESTION_TEMPLATES = [
  {
    type: "choice",
    options: ["first", "second", "third"],
    question: "",
  },
  {
    type: "score",
    levels: ["Bottom", "Low", "Mid", "High", "Top"],
    question: "",
  },
  {
    type: "noul",
    question: "Is this actionable?",
  },
];

function addDynamicQuestion(qConfig) {
  const cfg = qConfig || { type: "choice", options: ["option-a", "option-b"], question: "" };
  const idx = Date.now();
  const card = document.createElement("div");
  card.className = "dynamic-question-card";
  card.dataset.idx = idx;

  const cardNum = els.dynamicBuilder.querySelectorAll(".dynamic-question-card").length + 1;
  const typeLabel = cfg.type === "noul" ? "Noul (bool)" : cfg.type === "score" ? "Score" : "Choice";

  card.innerHTML = `
    <div class="dq-header">
      <span class="dq-card-num">#${cardNum}</span>
      <select class="dq-type">
        <option value="choice" ${cfg.type === "choice" ? "selected" : ""}>Choice</option>
        <option value="score" ${cfg.type === "score" ? "selected" : ""}>Score</option>
        <option value="noul" ${cfg.type === "noul" ? "selected" : ""}>Noul (bool)</option>
      </select>
      <button class="dq-remove" title="Remove this question">✕</button>
    </div>
    <textarea class="dq-question-text" rows="2"
              placeholder="What question are you asking? e.g. 'What is the sentiment?'">${escapeHtml(cfg.question || "")}</textarea>
    <div class="dq-options-label">Options:</div>
    <div class="dq-options">
      ${renderOptionInputs(cfg)}
    </div>
    <button class="dq-add-option">+ Add option</button>
  `;

  card.querySelector(".dq-remove").addEventListener("click", () => {
    card.remove();
    renumberCards();
  });
  card.querySelector(".dq-type").addEventListener("change", (e) => {
    const optsDiv = card.querySelector(".dq-options");
    const addBtn = card.querySelector(".dq-add-option");
    const newType = e.target.value;
    if (newType === "noul") {
      optsDiv.innerHTML = `<div class="dq-noul-hint">Boolean (true/false) — values are always fixed.</div>`;
      addBtn.classList.add("hidden");
    } else {
      const existing = collectOptions(card);
      const defaults = newType === "choice"
        ? (existing.length >= 2 ? existing : ["option-a", "option-b"])
        : (existing.length >= 2 ? existing : ["Low", "High"]);
      optsDiv.innerHTML = renderOptionInputs({ type: newType, [newType === "choice" ? "options" : "levels"]: defaults });
      addBtn.classList.remove("hidden");
    }
  });
  card.querySelector(".dq-add-option").addEventListener("click", () => {
    const optsDiv = card.querySelector(".dq-options");
    const rows = optsDiv.querySelectorAll(".dq-option-row");
    const row = document.createElement("div");
    row.className = "dq-option-row";
    row.innerHTML = `
      <input type="text" class="dq-option-input" value="new-option-${rows.length + 1}">
      <button class="dq-option-remove">✕</button>
    `;
    row.querySelector(".dq-option-remove").addEventListener("click", () => row.remove());
    optsDiv.appendChild(row);
  });

  // Prepend so new cards appear at the top (immediately visible)
  const firstCard = els.dynamicBuilder.querySelector(".dynamic-question-card");
  if (firstCard) {
    els.dynamicBuilder.insertBefore(card, firstCard);
  } else {
    els.dynamicBuilder.appendChild(card);
  }
  renumberCards();
}

function renumberCards() {
  const cards = els.dynamicBuilder.querySelectorAll(".dynamic-question-card");
  cards.forEach((card, i) => {
    const num = card.querySelector(".dq-card-num");
    if (num) num.textContent = `#${i + 1}`;
  });
}

function renderOptionInputs(cfg) {
  const items = cfg.type === "choice" ? (cfg.options || ["option-a", "option-b"])
              : cfg.type === "score" ? (cfg.levels || ["Low", "High"])
              : [];
  if (cfg.type === "noul") return `<div class="dq-noul-hint">Boolean (true/false) — values are always fixed.</div>`;
  return items.map((v, i) => `
    <div class="dq-option-row">
      <input type="text" class="dq-option-input" value="${escapeHtml(v)}">
      ${i >= 2 ? '<button class="dq-option-remove">✕</button>' : ''}
    </div>
  `).join("");
}

function collectOptions(card) {
  const inputs = card.querySelectorAll(".dq-option-input");
  return Array.from(inputs).map(el => el.value.trim()).filter(Boolean);
}

function collectDynamicQuestions() {
  const questions = [];
  for (const card of els.dynamicBuilder.querySelectorAll(".dynamic-question-card")) {
    const type = card.querySelector(".dq-type").value;
    const questionText = card.querySelector(".dq-question-text").value.trim();
    if (type === "noul") {
      questions.push({ type: "noul", question: questionText || "Is this true?" });
    } else if (type === "choice") {
      const options = collectOptions(card);
      if (options.length < 2) continue;
      questions.push({ type: "choice", options, question: questionText });
    } else {
      const levels = collectOptions(card);
      if (levels.length < 2) continue;
      questions.push({ type: "score", levels, question: questionText });
    }
  }
  return questions;
}

async function runDynamic() {
  const questions = collectDynamicQuestions();
  if (questions.length === 0) {
    els.dynamicSummary.className = "summary failure";
    els.dynamicSummary.textContent = "Add at least one valid question with ≥2 options/levels.";
    els.dynamicSummary.classList.remove("hidden");
    return;
  }

  els.dynamicSummary.className = "summary";
  els.dynamicSummary.textContent = "Running…";
  els.dynamicSummary.classList.remove("hidden");
  els.runDynamicBtn.disabled = true;

  try {
    const res = await fetch(DYNAMIC_API_BASE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...currentStatePayload(),
        questions,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderDynamicResults(data);
  } catch (err) {
    els.dynamicSummary.className = "summary failure";
    els.dynamicSummary.innerHTML = `<strong>Error:</strong> ${escapeHtml(err.message)}`;
    els.dynamicSummary.classList.remove("hidden");
  } finally {
    els.runDynamicBtn.disabled = false;
  }
}

function renderDynamicResults(data) {
  const card = document.createElement("div");
  card.className = "agent-result";

  const header = document.createElement("div");
  header.className = "agent-result-header";
  header.innerHTML = `
    <div class="name">Dynamic Head</div>
    <div class="meta">${data.latency_ms} ms · ${data.answers.length} questions</div>
  `;
  card.appendChild(header);

  const table = document.createElement("table");
  table.className = "question-table";
  table.innerHTML = `
    <thead>
      <tr><th>Question</th><th>Type</th><th>Predicted</th><th>Distribution</th><th>Conf.</th></tr>
    </thead>`;
  const tbody = document.createElement("tbody");

  for (const ans of data.answers) {
    const tr = document.createElement("tr");
    const conf = ans.confidence ? ans.confidence.toFixed(3) : "—";
    const type = ans.distribution
      ? (Object.keys(ans.distribution).length === 2
          && "true" in ans.distribution ? "noul" : "choice")
      : "score";

    tr.innerHTML = `
      <td>${escapeHtml(ans.question || "—")}</td>
      <td><span class="type-badge type-${type}">${type}</span></td>
      <td class="predicted">${escapeHtml(String(ans.predicted ?? "—"))}</td>
      <td class="dist-cell"></td>
      <td>${conf}</td>
    `;
    tr.querySelector(".dist-cell").appendChild(renderDistribution({ distribution: ans.distribution }));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  card.appendChild(table);

  // Replace previous dynamic results
  const existing = els.results.querySelector(".dynamic-result");
  if (existing) existing.remove();
  card.classList.add("dynamic-result");
  els.results.prepend(card);

  els.dynamicSummary.className = "summary success";
  els.dynamicSummary.textContent =
    `Done — ${data.answers.length} questions answered in ${data.latency_ms} ms`;
  els.dynamicSummary.classList.remove("hidden");
}

function loadDefaultQuestions() {
  for (const tmpl of QUESTION_TEMPLATES) {
    addDynamicQuestion(tmpl);
  }
}

els.addQuestionBtn = document.getElementById("add-question-btn");
els.runDynamicBtn = document.getElementById("run-dynamic-btn");
els.dynamicBuilder = document.getElementById("dynamic-builder");
els.dynamicSummary = document.getElementById("dynamic-summary");

els.addQuestionBtn.addEventListener("click", () => addDynamicQuestion());
els.runDynamicBtn.addEventListener("click", runDynamic);

// Load default templates on init
loadDefaultQuestions();

pollStatus();
setInterval(pollStatus, 5000);