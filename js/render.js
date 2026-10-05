// Canvas renderer for the grid: value heatmap or per-action Q triangles,
// greedy-policy arrows, terminal labels, and the agent with its trail.

import { argmaxAll } from './algorithms.js';

export function readTokens(el) {
  const cs = getComputedStyle(el);
  const get = (n) => cs.getPropertyValue(n).trim();
  return {
    surface: get('--surface-2'), cellBg: get('--cell'), gridLine: get('--grid-line'),
    wall: get('--wall'), goal: get('--goal'), pit: get('--pit'),
    pos: get('--div-pos'), neg: get('--div-neg'), mid: get('--div-mid'),
    ink: get('--text-primary'), ink2: get('--text-secondary'),
    agent: get('--agent'), onFill: get('--on-fill'),
  };
}

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}

function mix(a, b, t) {
  const A = hexToRgb(a), B = hexToRgb(b);
  return A.map((x, i) => Math.round(x + (B[i] - x) * t));
}

const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

function luminance([r, g, b]) {
  const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}

// Diverging scale: red (negative) <- gray (0) -> blue (positive).
function valueColor(v, scale, tk) {
  const t = Math.max(-1, Math.min(1, scale ? v / scale : 0));
  return t >= 0 ? mix(tk.mid, tk.pos, t) : mix(tk.mid, tk.neg, -t);
}

export function formatValue(v) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function drawArrow(ctx, cx, cy, a, len, color, width) {
  const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const [dx, dy] = dirs[a];
  const x2 = cx + dx * len, y2 = cy + dy * len;
  const head = Math.max(4, len * 0.45);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(x2 - dx * head * 0.6, y2 - dy * head * 0.6);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - dx * head - dy * head * 0.55, y2 - dy * head + dx * head * 0.55);
  ctx.lineTo(x2 - dx * head + dy * head * 0.55, y2 - dy * head - dx * head * 0.55);
  ctx.closePath();
  ctx.fill();
}

export function drawGrid(canvas, { env, agent, view, showNumbers, agentPos, trail, hover, tokens: tk }) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const cell = W / env.cols;
  const H = cell * env.rows;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  canvas.style.height = `${H}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  // Shared color scale across all non-terminal cells so colors are comparable.
  let scale = 0;
  for (let s = 0; s < env.nStates; s++) {
    if (env.isWall(s) || env.isTerminal(s)) continue;
    const vals = view === 'q' ? agent.qValues(s) : [agent.value(s)];
    for (const v of vals) scale = Math.max(scale, Math.abs(v));
  }

  const font = (px, weight = 500) => `${weight} ${px}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`;
  const numPx = Math.max(9, Math.min(14, cell * 0.22));
  const pad = 1; // 2px surface gap between neighbouring cells

  for (let s = 0; s < env.nStates; s++) {
    const [r, c] = env.rc(s);
    const x = c * cell + pad, y = r * cell + pad, w = cell - 2 * pad;
    const cx = x + w / 2, cy = y + w / 2;
    const type = env.cells[s];

    if (type === 'wall') {
      ctx.fillStyle = tk.wall;
      ctx.fillRect(x, y, w, w);
      continue;
    }
    if (type === 'goal' || type === 'pit') {
      ctx.fillStyle = type === 'goal' ? tk.goal : tk.pit;
      ctx.fillRect(x, y, w, w);
      ctx.fillStyle = tk.onFill;
      ctx.font = font(Math.max(10, Math.min(16, cell * 0.26)), 650);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const rew = type === 'goal' ? env.goalReward : env.pitReward;
      ctx.fillText(`${rew > 0 ? '+' : ''}${rew}`, cx, cy);
      continue;
    }

    const qs = agent.qValues(s);
    if (view === 'q') {
      // Four triangles meeting at the centre: top=up, right, bottom=down, left.
      const corners = [[x, y], [x + w, y], [x + w, y + w], [x, y + w]];
      for (let a = 0; a < 4; a++) {
        const p1 = corners[a], p2 = corners[(a + 1) % 4];
        ctx.fillStyle = rgb(valueColor(qs[a], scale, tk));
        ctx.beginPath();
        ctx.moveTo(cx, cy); ctx.lineTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = tk.surface;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y); ctx.lineTo(x + w, y + w); ctx.moveTo(x + w, y); ctx.lineTo(x, y + w);
      ctx.stroke();
    } else {
      const col = valueColor(agent.value(s), scale, tk);
      ctx.fillStyle = rgb(col);
      ctx.fillRect(x, y, w, w);
    }

    const bg = view === 'q' ? hexToRgb(tk.mid) : valueColor(agent.value(s), scale, tk);
    const dark = luminance(bg) < 0.35;
    const ink = dark ? '#ffffff' : '#0b0b0b';

    // Greedy policy arrows (all tied best actions); skipped when every action ties.
    const policyAction = agent.policyAction?.(s);
    const best = policyAction !== undefined ? [policyAction] : argmaxAll(qs);
    const untrained = agent.visited && !agent.visited(s);
    if (best.length < 4 && !untrained) {
      const len = w * (showNumbers && view === 'values' ? 0.2 : 0.3);
      const acy = showNumbers && view === 'values' ? cy - w * 0.12 : cy;
      for (const a of best) drawArrow(ctx, cx, acy, a, len, ink, Math.max(1.5, w * 0.04));
    }

    if (showNumbers && cell >= 40) {
      ctx.fillStyle = ink;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      if (view === 'q' && cell >= 64) {
        ctx.font = font(Math.max(8, numPx * 0.75));
        ctx.fillText(formatValue(qs[0]), cx, y + w * 0.14);
        ctx.fillText(formatValue(qs[2]), cx, y + w * 0.86);
        ctx.textAlign = 'left';
        ctx.fillText(formatValue(qs[3]), x + 2, cy);
        ctx.textAlign = 'right';
        ctx.fillText(formatValue(qs[1]), x + w - 2, cy);
      } else if (view === 'values') {
        ctx.font = font(numPx);
        ctx.fillText(formatValue(agent.value(s)), cx, cy + w * 0.24);
      }
    }

    if (type === 'start') {
      ctx.font = font(Math.max(9, cell * 0.18), 700);
      ctx.fillStyle = ink;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('S', x + 3, y + 2);
    }
  }

  // Trail of the current episode, then the agent on top.
  if (trail?.length) {
    ctx.fillStyle = tk.agent;
    ctx.globalAlpha = 0.45;
    for (const s of trail) {
      const [r, c] = env.rc(s);
      ctx.beginPath();
      ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, Math.max(2.5, cell * 0.07), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
  if (agentPos != null) {
    const [r, c] = env.rc(agentPos);
    ctx.beginPath();
    ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, Math.max(6, cell * 0.22), 0, Math.PI * 2);
    ctx.fillStyle = tk.agent;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = tk.surface;
    ctx.stroke();
  }

  if (hover != null) {
    const [r, c] = env.rc(hover);
    ctx.strokeStyle = tk.ink;
    ctx.lineWidth = 2;
    ctx.strokeRect(c * cell + 1, r * cell + 1, cell - 2, cell - 2);
  }

  return { cell, scale };
}
