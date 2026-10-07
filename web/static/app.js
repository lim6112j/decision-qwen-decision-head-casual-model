/* Decision Lab web simulator: SSE-driven gridworld playback. */

const STEP_PACE_MS = 350;   // rendering delay between steps (head agents decide in ~ms)
const API_BASE = "";

const els = {
  status: document.getElementById("status"),
  agentCards: document.getElementById("agent-cards"),
  layoutSelect: document.getElementById("layout-select"),
  runBtn: document.getElementById("run-btn"),
  compareBtn: document.getElementById("compare-btn"),
  grid: document.getElementById("grid"),
  summary: document.getElementById("summary"),
  stepRows: document.getElementById("step-rows"),
  stepCount: document.getElementById("step-count"),
  comparePanel: document.getElementById("compare-panel"),
  compareRows: document.getElementById("compare-rows"),
  compareLabel: document.getElementById("compare-layout-label"),
};

const state = {
  agents: [],
  selectedAgent: null,
  gridSpec: null,      // {rows, cols, walls, goal_pos}
  agentCell: null,     // [r, c] currently rendered
  running: false,
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
  els.compareBtn.disabled = !ready || state.running;
  if (ready && els.agentCards.querySelector(".loading")) {
    loadAgents();
    loadLayouts();
  }
}

/* ---------- model selection (3 cards) ---------- */

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
  // default to the trained head
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

/* ---------- layouts ---------- */

async function loadLayouts() {
  try {
    const res = await fetch(`${API_BASE}/api/layouts`);
    const body = await res.json();
    els.layoutSelect.innerHTML = "";
    for (const layout of body.layouts) {
      const opt = document.createElement("option");
      opt.value = layout.layout_id;
      opt.textContent =
        `Layout ${layout.layout_id} (${layout.rows}×${layout.cols}, optimal ${layout.optimal_from_start})`;
      els.layoutSelect.appendChild(opt);
    }
    const randomOpt = document.createElement("option");
    randomOpt.value = "random";
    randomOpt.textContent = "Random layout (new each run)";
    els.layoutSelect.appendChild(randomOpt);
  } catch (err) {
    els.layoutSelect.innerHTML = `<option>Failed to load layouts: ${err}</option>`;
  }
}

function selectedLayoutId() {
  const value = els.layoutSelect.value;
  return value === "random" ? null : Number(value);
}

/* ---------- grid rendering ---------- */

function renderGrid(spec) {
  state.gridSpec = spec;
  state.agentCell = spec.agent_pos;
  els.grid.style.gridTemplateColumns = `repeat(${spec.cols}, 44px)`;
  els.grid.innerHTML = "";
  for (let r = 0; r < spec.rows; r++) {
    for (let c = 0; c < spec.cols; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.pos = `${r},${c}`;
      if (spec.walls[r][c]) cell.classList.add("wall");
      if (r === spec.goal_pos[0] && c === spec.goal_pos[1]) cell.classList.add("goal");
      els.grid.appendChild(cell);
    }
  }
  paintAgent(spec.agent_pos);
}

function paintAgent(pos) {
  if (state.agentCell) {
    const prev = cellAt(state.agentCell);
    if (prev) {
      prev.classList.remove("agent", "agent-goal");
      if (isGoal(state.agentCell)) prev.classList.add("goal");
    }
  }
  const cell = cellAt(pos);
  if (cell) {
    cell.classList.add("agent");
    if (isGoal(pos)) cell.classList.add("agent-goal");
  }
  state.agentCell = pos;
}

function cellAt(pos) {
  return els.grid.querySelector(`[data-pos="${pos[0]},${pos[1]}"]`);
}

function isGoal(pos) {
  return state.gridSpec &&
    pos[0] === state.gridSpec.goal_pos[0] && pos[1] === state.gridSpec.goal_pos[1];
}

/* ---------- step log ---------- */

function clearLog() {
  els.stepRows.innerHTML = "";
  els.stepCount.textContent = "";
  els.summary.classList.add("hidden");
}

function appendStep(step) {
  const tr = document.createElement("tr");
  const mark = step.correct ? "✓" : "✗";
  const cls = step.correct ? "mark-good" : "mark-bad";
  tr.className = `step-row-latest ${cls}`;
  tr.innerHTML = `
    <td>${step.step_idx}</td>
    <td>${step.action_name ?? "parse-fail"}</td>
    <td>${step.optimal_action_name}</td>
    <td class="${cls}">${mark}</td>
    <td>${step.latency_ms} ms</td>
    <td class="raw-output">${escapeHtml(step.raw_output)}</td>
  `;
  const prev = els.stepRows.querySelector(".step-row-latest");
  if (prev) prev.classList.remove("step-row-latest");
  els.stepRows.appendChild(tr);
  els.stepCount.textContent = `(${step.step_idx + 1})`;
}

function showSummary(summary) {
  const cls = summary.success ? "success" : "failure";
  const verdict = summary.success
    ? `Reached the goal in ${summary.steps_used} steps (optimal ${summary.optimal_steps}).`
    : `Did not reach the goal in ${summary.steps_used} steps (optimal ${summary.optimal_steps}).`;
  els.summary.className = `summary ${cls}`;
  els.summary.innerHTML = `${verdict}<br>
    Action accuracy: ${(summary.action_accuracy * 100).toFixed(1)}% ·
    Parse failures: ${summary.parse_failures}`;
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

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ---------- run / compare ---------- */

async function runEpisode() {
  if (!state.selectedAgent || state.running) return;
  state.running = true;
  setButtonsDisabled(true);
  clearLog();
  els.comparePanel.classList.add("hidden");

  try {
    await postSSE("/api/run", {
      agent_id: state.selectedAgent,
      layout_id: selectedLayoutId(),
    }, async (event) => {
      if (event.event === "start") {
        renderGrid(event.data.grid);
      } else if (event.event === "step") {
        await sleep(STEP_PACE_MS);
        paintAgent(event.data.new_pos);
        appendStep(event.data);
      } else if (event.event === "summary") {
        showSummary(event.data);
      }
    });
  } catch (err) {
    showError(err.message);
  } finally {
    state.running = false;
    setButtonsDisabled(false);
  }
}

async function runCompare() {
  if (state.running) return;
  state.running = true;
  setButtonsDisabled(true);
  clearLog();
  els.compareRows.innerHTML = "";
  els.comparePanel.classList.remove("hidden");
  els.compareLabel.textContent = `(layout ${els.layoutSelect.selectedOptions[0]?.textContent ?? ""})`;

  const summaries = [];
  try {
    await postSSE("/api/compare", { layout_id: selectedLayoutId() }, async (event) => {
      if (event.event === "agent_start") {
        els.stepRows.innerHTML = "";
      } else if (event.event === "start") {
        renderGrid(event.data.grid);
      } else if (event.event === "step") {
        await sleep(STEP_PACE_MS);
        paintAgent(event.data.new_pos);
        appendStep(event.data);
      } else if (event.event === "summary") {
        summaries.push(event.data);
        appendCompareRow(event.data, summaries);
      }
    });
  } catch (err) {
    showError(err.message);
  } finally {
    state.running = false;
    setButtonsDisabled(false);
  }
}

function appendCompareRow(summary, summaries) {
  const agent = state.agents.find(a => a.agent_id === summary.agent_id);
  const name = agent ? agent.name : `Model ${summaries.length}`;
  const tr = document.createElement("tr");
  const cls = summary.success ? "mark-good" : "mark-bad";
  tr.innerHTML = `
    <td>${name}</td>
    <td class="${cls}">${summary.success ? "reached goal" : "failed"}</td>
    <td>${summary.steps_used}</td>
    <td>${summary.optimal_steps}</td>
    <td>${(summary.action_accuracy * 100).toFixed(1)}%</td>
    <td>${summary.parse_failures}</td>
  `;
  els.compareRows.appendChild(tr);
}

function setButtonsDisabled(disabled) {
  els.runBtn.disabled = disabled;
  els.compareBtn.disabled = disabled;
}

/* ---------- init ---------- */

els.runBtn.addEventListener("click", runEpisode);
els.compareBtn.addEventListener("click", runCompare);
pollStatus();
setInterval(pollStatus, 5000);
