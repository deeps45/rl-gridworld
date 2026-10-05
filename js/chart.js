// Minimal canvas line chart with a hover crosshair. One y-axis, recessive grid,
// 2px lines, legend handled in HTML by the caller.

function niceTicks(min, max, count = 4) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? 10 * mag;
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

export class LineChart {
  constructor(canvas, tooltip) {
    this.canvas = canvas;
    this.tooltip = tooltip;
    this.hoverIndex = null;
    this.data = null;
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerleave', () => { this.hoverIndex = null; this.tooltip.hidden = true; this.draw(); });
  }

  // data: { series: [{ name, values, color, width, alpha }], log, xLabel, format }
  setData(data, tokens) {
    this.data = data;
    this.tokens = tokens;
    this.draw();
  }

  layout() {
    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    return { W, H, left: 52, right: 12, top: 10, bottom: 28 };
  }

  onMove(e) {
    if (!this.data) return;
    const n = this.data.series[0].values.length;
    if (!n) return;
    const rect = this.canvas.getBoundingClientRect();
    const { W, left, right } = this.layout();
    const x = e.clientX - rect.left;
    const t = (x - left) / (W - left - right);
    this.hoverIndex = Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
    this.draw();
  }

  draw() {
    const { canvas, data, tokens: tk } = this;
    if (!data || !tk) return;
    const dpr = window.devicePixelRatio || 1;
    const { W, H, left, right, top, bottom } = this.layout();
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = '12px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';

    const n = data.series[0].values.length;
    const tf = data.log ? (v) => Math.log10(Math.max(v, 1e-12)) : (v) => v;
    if (n === 0) {
      ctx.fillStyle = tk.ink2;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(data.empty ?? 'No data yet', W / 2, H / 2);
      this.tooltip.hidden = true;
      return;
    }

    let min = Infinity, max = -Infinity;
    for (const s of data.series) for (const v of s.values) { const y = tf(v); min = Math.min(min, y); max = Math.max(max, y); }
    let ticks;
    if (data.log) {
      const lo = Math.floor(min), hi = Math.max(Math.ceil(max), lo + 1);
      const stride = Math.ceil((hi - lo) / 5);
      ticks = [];
      for (let v = lo; v <= hi; v += stride) ticks.push(v);
      min = lo; max = ticks[ticks.length - 1];
    } else {
      if (data.zeroBased) { min = Math.min(0, min); max = Math.max(0, max); }
      ticks = niceTicks(min, max);
      min = ticks[0]; max = ticks[ticks.length - 1];
    }

    const pw = W - left - right, ph = H - top - bottom;
    const X = (i) => left + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const Y = (y) => top + ph - ((y - min) / (max - min || 1)) * ph;

    // Grid + y labels
    ctx.strokeStyle = tk.gridLine;
    ctx.lineWidth = 1;
    ctx.fillStyle = tk.ink2;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of ticks) {
      const y = Math.round(Y(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(W - right, y); ctx.stroke();
      ctx.fillText(data.log ? `1e${t}` : data.format(t), left - 8, y);
    }
    // x labels: first and last episode/iteration
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xt = niceTicks(1, n, 4).filter((v) => v >= 1 && v <= n && Number.isInteger(v));
    for (const v of xt) ctx.fillText(String(v), X(v - 1), H - bottom + 8);

    for (const s of data.series) {
      ctx.strokeStyle = s.color;
      ctx.globalAlpha = s.alpha ?? 1;
      ctx.lineWidth = s.width ?? 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      s.values.forEach((v, i) => { const px = X(i), py = Y(tf(v)); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
      if (n <= 30) {
        ctx.fillStyle = s.color;
        s.values.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(tf(v)), 4, 0, Math.PI * 2); ctx.fill(); });
      }
      ctx.globalAlpha = 1;
    }

    const hi = this.hoverIndex;
    if (hi != null && hi < n) {
      const x = Math.round(X(hi)) + 0.5;
      ctx.strokeStyle = tk.ink2;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
      for (const s of data.series) {
        ctx.beginPath();
        ctx.arc(X(hi), Y(tf(s.values[hi])), 4.5, 0, Math.PI * 2);
        ctx.fillStyle = s.color; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = tk.surface; ctx.stroke();
      }
      const rows = data.series.map((s) =>
        `<div class="tt-row"><span class="swatch" style="background:${s.color}"></span>${s.name}<b>${data.format(s.values[hi])}</b></div>`).join('');
      this.tooltip.innerHTML = `<div class="tt-title">${data.xLabel} ${hi + 1}</div>${rows}`;
      this.tooltip.hidden = false;
      const tw = this.tooltip.offsetWidth;
      this.tooltip.style.left = `${Math.min(Math.max(0, x + 12), W - tw)}px`;
      this.tooltip.style.top = `${top}px`;
    } else {
      this.tooltip.hidden = true;
    }
  }
}
