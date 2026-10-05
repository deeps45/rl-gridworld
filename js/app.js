import { GridWorld, PRESETS, ACTION_NAMES, mulberry32 } from './env.js';
import { ValueIteration, PolicyIteration, TDControl, argmaxAll } from './algorithms.js';
import { drawGrid, readTokens, formatValue } from './render.js';
import { LineChart } from './chart.js';

const $ = (id) => document.getElementById(id);

const ALGOS = {
  vi: {
    explain: 'Sweeps every state and replaces its value with the best one-step lookahead, updating in place. Each Step is one sweep, and it stops once the largest change falls below 1e-4.',
    rule: 'V(s) ← maxₐ Σₛ′ p(s′|s,a) [ r + γ·V(s′) ]',
  },
  pi: {
    explain: 'Alternates full policy evaluation with greedy improvement, starting from "always go up". Each Step evaluates the current policy to convergence and then switches every state to its best action.',
    rule: 'V ← V^π ;  π(s) ← argmaxₐ Σₛ′ p(s′|s,a) [ r + γ·V(s′) ]',
  },
  qlearning: {
    explain: 'Model-free and off-policy. The agent only sees sampled transitions and bootstraps from the best next action while acting ε-greedily. Each Step runs one episode.',
    rule: 'Q(s,a) ← Q(s,a) + α [ r + γ·maxₐ′ Q(s′,a′) − Q(s,a) ]',
  },
  sarsa: {
    explain: 'Model-free and on-policy. It bootstraps from the action it will actually take next, so it learns a safer route near hazards while exploring. Compare it with Q-learning on "Cliff walk".',
    rule: 'Q(s,a) ← Q(s,a) + α [ r + γ·Q(s′,a′) − Q(s,a) ]',
  },
};

const SPEEDS = [ // [ms between ticks, episodes per tick for TD training]
  [700, 1], [300, 1], [120, 3], [40, 10], [0, 50],
];
const MAX_TRAIN_EPISODES = 20000;

const presetNames = [...Object.keys(PRESETS), 'Empty 8×8'];

const state = {
  env: null,
  algo: 'vi',
  agent: null,
  view: 'values',
  tool: 'inspect',
  mode: 'idle', // idle | running | animating
  anim: null,
  agentPos: null,
  trail: [],
  hover: null,
  lastTick: 0,
  tokens: null,
};

const isTD = () => state.algo === 'qlearning' || state.algo === 'sarsa';
const num = (id) => parseFloat($(id).value);

function loadPreset(name) {
  if (name === 'Empty 8×8') {
    const lines = Array.from({ length: 8 }, () => '........');
    lines[7] = 'S.......';
    lines[0] = '.......G';
    state.env = GridWorld.fromStrings(lines, { stepReward: -0.1, goalReward: 10, pitReward: -10, slip: 0 });
  } else {
    const { lines, opts } = PRESETS[name];
    state.env = GridWorld.fromStrings(lines, opts);
  }
  const e = state.env;
  $('stepReward').value = e.stepReward;
  $('goalReward').value = e.goalReward;
  $('pitReward').value = e.pitReward;
  $('slip').value = e.slip;
  // Undiscounted problems (the cliff) read best with γ = 1.
  $('gamma').value = name === 'Cliff walk' ? 1 : name === 'Classic 4×3' ? 0.99 : 0.9;
  syncOutputs();
  resetAgent();
}

function syncOutputs() {
  for (const id of ['gamma', 'slip', 'alpha', 'epsilon']) $(`${id}Out`).textContent = (+$(id).value).toFixed(2);
}

function resetAgent() {
  stop();
  const env = state.env;
  const gamma = num('gamma');
  if (state.algo === 'vi') state.agent = new ValueIteration(env, { gamma });
  else if (state.algo === 'pi') state.agent = new PolicyIteration(env, { gamma });
  else {
    state.agent = new TDControl(env, {
      method: state.algo, gamma, alpha: num('alpha'), epsilon: num('epsilon'),
      rng: mulberry32(Math.trunc(num('seed')) || 0),
    });
  }
  state.agentPos = null;
  state.trail = [];
  render();
}

// ---------- running & animation ----------

function stop() {
  state.mode = 'idle';
  state.anim = null;
  updateButtons();
}

function startLoop() {
  state.lastTick = 0;
  requestAnimationFrame(loop);
}

function loop(ts) {
  if (state.mode === 'idle') { render(); return; }
  const [interval, perTick] = SPEEDS[+$('speed').value];
  if (ts - state.lastTick >= interval) {
    state.lastTick = ts;
    if (state.mode === 'running') tickRun(perTick);
    else if (state.mode === 'animating') tickAnim();
    render();
  }
  requestAnimationFrame(loop);
}

function tickRun(perTick) {
  const a = state.agent;
  if (isTD()) {
    for (let i = 0; i < perTick; i++) a.runEpisode();
    if (a.history.length >= MAX_TRAIN_EPISODES) stop();
  } else {
    a.step();
    if (a.converged) stop();
  }
}

function tickAnim() {
  const res = state.anim.next();
  if (res.done) {
    stop();
    return;
  }
  const { s, s2, done } = res.value;
  if (!state.trail.length) state.trail.push(s);
  state.trail.push(s2);
  state.agentPos = s2;
  if (done) stop();
}

function* greedyRollout() {
  const env = state.env, agent = state.agent;
  const rng = Math.random;
  let s = env.startState();
  for (let t = 0; t < 200; t++) {
    if (env.isTerminal(s)) return;
    const best = agent.policyAction ? [agent.policyAction(s)] : argmaxAll(agent.qValues(s));
    const a = best[Math.floor(rng() * best.length)];
    const tr = env.step(s, a, rng);
    yield { s, a, ...tr };
    if (tr.done) return;
    s = tr.s2;
  }
}

function beginAnimation(gen) {
  state.mode = 'animating';
  state.anim = gen;
  state.agentPos = state.env.startState();
  state.trail = [];
  updateButtons();
  startLoop();
}

// ---------- UI ----------

function updateButtons() {
  $('runBtn').textContent = state.mode === 'running' ? 'Pause' : 'Run';
  $('watchBtn').hidden = !isTD();
  for (const el of document.querySelectorAll('.field.td')) el.hidden = !isTD();
  const a = state.agent;
  $('stepBtn').disabled = false;
  $('runBtn').disabled = !isTD() && a?.converged && state.mode !== 'running';
}

function stat(k, v) { return `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`; }

function renderStats() {
  const a = state.agent;
  let html = '';
  if (state.algo === 'vi') {
    const d = a.history.at(-1);
    html = stat('Sweeps', a.history.length) + stat('Max ΔV', d === undefined ? '–' : d.toExponential(1))
      + stat('Status', a.converged ? 'Converged' : 'Running');
  } else if (state.algo === 'pi') {
    html = stat('Iterations', a.history.length) + stat('Eval sweeps', a.history.length ? a.evalSweeps : '–')
      + stat('Status', a.converged ? 'Policy stable' : a.history.length ? `${a.history.at(-1)} changed` : 'Not started');
  } else {
    const h = a.history, recent = h.slice(-50);
    const avg = recent.length ? recent.reduce((x, y) => x + y, 0) / recent.length : null;
    html = stat('Episodes', h.length) + stat('Last return', h.length ? formatValue(h.at(-1)) : '–')
      + stat('Avg (last 50)', avg === null ? '–' : formatValue(avg)) + stat('Last length', h.length ? a.lastLength : '–');
  }
  $('stats').innerHTML = html;
}

function movingAverage(xs, k) {
  const out = [];
  let sum = 0;
  xs.forEach((x, i) => { sum += x; if (i >= k) sum -= xs[i - k]; out.push(sum / Math.min(i + 1, k)); });
  return out;
}

function renderChart() {
  const tk = state.tokens, a = state.agent;
  const cs = getComputedStyle(document.documentElement);
  const c1 = cs.getPropertyValue('--series-1').trim(), c2 = cs.getPropertyValue('--series-2').trim();
  let data, legend = '';
  if (state.algo === 'vi') {
    $('chartTitle').textContent = 'Largest value change per sweep (log scale)';
    data = { series: [{ name: 'max ΔV', values: a.history, color: c1 }], log: true, xLabel: 'Sweep',
      format: (v) => v.toExponential(2), empty: 'Press Step or Run to start sweeping' };
  } else if (state.algo === 'pi') {
    $('chartTitle').textContent = 'States whose action changed per iteration';
    data = { series: [{ name: 'changed', values: a.history, color: c1 }], zeroBased: true, xLabel: 'Iteration',
      format: (v) => String(Math.round(v)), empty: 'Press Step or Run to start iterating' };
  } else {
    $('chartTitle').textContent = 'Return per episode';
    const k = Math.max(5, Math.min(100, Math.floor(a.history.length / 20)));
    data = {
      series: [
        { name: 'Return', values: a.history, color: c1, width: 1.25, alpha: 0.45 },
        { name: `${k}-episode avg`, values: movingAverage(a.history, k), color: c2, width: 2 },
      ],
      xLabel: 'Episode', format: (v) => formatValue(v), empty: 'Press Run to train, or Step for one episode',
    };
    legend = data.series.map((s) => `<span><i class="swatch" style="background:${s.color}"></i>${s.name}</span>`).join('');
  }
  $('chartLegend').innerHTML = legend;
  chart.setData(data, tk);
}

function renderCellTip() {
  const tip = $('cellTip');
  const s = state.hover;
  if (s == null || state.env.isWall(s)) { tip.hidden = true; return; }
  const env = state.env, a = state.agent;
  const [r, c] = env.rc(s);
  const type = env.cells[s];
  let html = `<div class="tt-title">Cell (${r}, ${c})${type !== 'empty' ? ` · ${type}` : ''}</div>`;
  if (env.isTerminal(s)) {
    html += `<div class="tt-row">Terminal, reward on entry<b>${formatValue(env.reward(s))}</b></div>`;
  } else {
    const qs = a.qValues(s);
    const best = argmaxAll(qs);
    html += `<div class="tt-row">V(s)<b>${formatValue(a.value(s))}</b></div>`;
    qs.forEach((q, i) => {
      const isBest = best.length < 4 && best.includes(i);
      html += `<div class="tt-row${isBest ? ' best' : ''}">Q(s, ${ACTION_NAMES[i]})<b>${formatValue(q)}</b></div>`;
    });
  }
  tip.innerHTML = html;
  tip.hidden = false;
  const wrap = $('gridWrap');
  const cell = wrap.clientWidth / env.cols;
  let x = (c + 1) * cell + 8, y = r * cell;
  if (x + tip.offsetWidth > wrap.clientWidth) x = c * cell - tip.offsetWidth - 8;
  if (x < 0) x = 4;
  y = Math.min(y, Math.max(0, wrap.clientHeight - tip.offsetHeight));
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function render() {
  state.tokens = readTokens(document.documentElement);
  drawGrid($('grid'), {
    env: state.env, agent: state.agent, view: state.view, showNumbers: $('showNumbers').checked,
    agentPos: state.agentPos, trail: state.trail, hover: state.hover, tokens: state.tokens,
  });
  renderStats();
  renderChart();
  renderCellTip();
  updateButtons();
}

// ---------- editing ----------

let painting = false;

function cellFromEvent(e) {
  const canvas = $('grid');
  const rect = canvas.getBoundingClientRect();
  const cell = rect.width / state.env.cols;
  const c = Math.floor((e.clientX - rect.left) / cell), r = Math.floor((e.clientY - rect.top) / cell);
  if (r < 0 || c < 0 || r >= state.env.rows || c >= state.env.cols) return null;
  return state.env.idx(r, c);
}

function paint(s) {
  const env = state.env;
  const tool = state.tool;
  if (s == null || env.cells[s] === tool) return;
  if (tool === 'start') {
    const old = env.cells.indexOf('start');
    if (old >= 0) env.cells[old] = 'empty';
  } else if (env.cells[s] === 'start' && env.cells.filter((x) => x !== 'wall').length <= 2) {
    return;
  }
  env.cells[s] = tool;
  resetAgent();
}

function setupGridEvents() {
  const canvas = $('grid');
  canvas.addEventListener('pointerdown', (e) => {
    const s = cellFromEvent(e);
    if (state.tool === 'inspect') { state.hover = s; render(); return; }
    painting = true;
    canvas.setPointerCapture(e.pointerId);
    paint(s);
  });
  canvas.addEventListener('pointermove', (e) => {
    const s = cellFromEvent(e);
    if (painting && state.tool !== 'start') paint(s);
    if (s !== state.hover) { state.hover = s; if (state.mode === 'idle') render(); }
  });
  const end = () => { painting = false; };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => { state.hover = null; if (state.mode === 'idle') render(); });
}

function segment(id, attr, onPick) {
  const root = $(id);
  root.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    for (const b of root.querySelectorAll('button')) b.classList.toggle('on', b === btn);
    onPick(btn.dataset[attr]);
  });
}

function setAlgo(algo) {
  state.algo = algo;
  $('explain').textContent = ALGOS[algo].explain;
  $('rule').textContent = ALGOS[algo].rule;
  resetAgent();
}

// ---------- wire up ----------

const chart = new LineChart($('chart'), $('chartTip'));

for (const name of presetNames) {
  const opt = document.createElement('option');
  opt.textContent = name;
  $('preset').append(opt);
}
$('preset').value = 'Four rooms';
$('preset').addEventListener('change', (e) => loadPreset(e.target.value));

segment('tools', 'tool', (t) => { state.tool = t; });
segment('viewSeg', 'view', (v) => { state.view = v; render(); });
segment('algoSeg', 'algo', setAlgo);
$('showNumbers').addEventListener('change', render);

$('stepBtn').addEventListener('click', () => {
  if (state.mode !== 'idle') stop();
  state.agentPos = null; state.trail = [];
  if (isTD()) state.agent.runEpisode(); else state.agent.step();
  render();
});
$('runBtn').addEventListener('click', () => {
  if (state.mode === 'running') { stop(); return; }
  state.agentPos = null; state.trail = [];
  state.mode = 'running';
  updateButtons();
  startLoop();
});
$('watchBtn').addEventListener('click', () => beginAnimation(state.agent.episode()));
$('playBtn').addEventListener('click', () => beginAnimation(greedyRollout()));
$('resetBtn').addEventListener('click', resetAgent);

// Model parameters change the MDP or the fixed point, so they restart learning.
for (const id of ['gamma', 'slip', 'stepReward', 'goalReward', 'pitReward', 'seed']) {
  $(id).addEventListener('change', () => {
    const env = state.env;
    env.slip = num('slip');
    for (const k of ['stepReward', 'goalReward', 'pitReward']) if (Number.isFinite(num(k))) env[k] = num(k);
    resetAgent();
  });
}
for (const id of ['gamma', 'slip', 'alpha', 'epsilon']) $(id).addEventListener('input', syncOutputs);
// Learning-rate and exploration tweaks apply live, mid-training.
for (const id of ['alpha', 'epsilon']) {
  $(id).addEventListener('input', () => { if (isTD()) state.agent[id] = num(id); });
}

setupGridEvents();
new ResizeObserver(() => render()).observe($('gridWrap'));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);

loadPreset('Four rooms');
setAlgo('vi');
