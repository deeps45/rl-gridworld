// Tabular RL algorithms over a GridWorld. Every agent exposes the same view
// interface so the renderer doesn't care which one is active:
//   value(s) -> number, qValues(s) -> number[4], history -> number[]

function expectedQ(env, V, s, a, gamma) {
  let q = 0;
  for (const { p, s2, r, done } of env.transitions(s, a)) q += p * (r + (done ? 0 : gamma * V[s2]));
  return q;
}

export function argmaxAll(qs, eps = 1e-9) {
  const m = Math.max(...qs);
  const best = [];
  qs.forEach((q, a) => { if (q >= m - eps) best.push(a); });
  return best;
}

function isActive(env, s) { return !env.isWall(s) && !env.isTerminal(s); }

export class ValueIteration {
  constructor(env, { gamma = 0.9, theta = 1e-4 } = {}) {
    this.env = env;
    this.gamma = gamma;
    this.theta = theta;
    this.V = new Float64Array(env.nStates);
    this.history = []; // max |ΔV| per sweep
    this.converged = false;
  }

  // One in-place (Gauss-Seidel) sweep of the Bellman optimality backup.
  step() {
    if (this.converged) return 0;
    let delta = 0;
    for (let s = 0; s < this.env.nStates; s++) {
      if (!isActive(this.env, s)) continue;
      const old = this.V[s];
      this.V[s] = Math.max(...this.qValues(s));
      delta = Math.max(delta, Math.abs(old - this.V[s]));
    }
    this.history.push(delta);
    this.converged = delta < this.theta;
    return delta;
  }

  value(s) { return this.V[s]; }
  qValues(s) { return [0, 1, 2, 3].map((a) => expectedQ(this.env, this.V, s, a, this.gamma)); }
}

export class PolicyIteration {
  constructor(env, { gamma = 0.9, theta = 1e-4 } = {}) {
    this.env = env;
    this.gamma = gamma;
    this.theta = theta;
    this.V = new Float64Array(env.nStates);
    this.pi = new Int8Array(env.nStates); // start with "always up"
    this.history = []; // number of states whose action changed per improvement
    this.evalSweeps = 0;
    this.converged = false;
  }

  evaluate(maxSweeps = 1000) {
    let sweeps = 0;
    for (; sweeps < maxSweeps; sweeps++) {
      let delta = 0;
      for (let s = 0; s < this.env.nStates; s++) {
        if (!isActive(this.env, s)) continue;
        const old = this.V[s];
        this.V[s] = expectedQ(this.env, this.V, s, this.pi[s], this.gamma);
        delta = Math.max(delta, Math.abs(old - this.V[s]));
      }
      if (delta < this.theta) { sweeps++; break; }
    }
    return sweeps;
  }

  improve() {
    let changes = 0;
    for (let s = 0; s < this.env.nStates; s++) {
      if (!isActive(this.env, s)) continue;
      const best = argmaxAll(this.qValues(s));
      // Keep the current action on ties so the loop terminates.
      if (!best.includes(this.pi[s])) { this.pi[s] = best[0]; changes++; }
    }
    return changes;
  }

  // One full iteration: evaluate the current policy, then act greedily on it.
  step() {
    if (this.converged) return 0;
    this.evalSweeps = this.evaluate();
    const changes = this.improve();
    this.history.push(changes);
    this.converged = changes === 0;
    return changes;
  }

  value(s) { return this.V[s]; }
  qValues(s) { return [0, 1, 2, 3].map((a) => expectedQ(this.env, this.V, s, a, this.gamma)); }
  policyAction(s) { return this.pi[s]; }
}

// Model-free control: Q-learning (off-policy) or SARSA (on-policy), ε-greedy.
export class TDControl {
  constructor(env, { method = 'qlearning', gamma = 0.9, alpha = 0.5, epsilon = 0.1, maxSteps = 300, rng = Math.random } = {}) {
    this.env = env;
    this.method = method;
    this.gamma = gamma;
    this.alpha = alpha;
    this.epsilon = epsilon;
    this.maxSteps = maxSteps;
    this.rng = rng;
    this.Q = new Float64Array(env.nStates * 4);
    this.history = []; // undiscounted return per episode
    this.lastLength = 0;
  }

  qValues(s) { return [0, 1, 2, 3].map((a) => this.Q[s * 4 + a]); }
  value(s) { return Math.max(...this.qValues(s)); }
  visited(s) { return this.qValues(s).some((q) => q !== 0); }

  greedy(s) {
    const best = argmaxAll(this.qValues(s));
    return best[Math.floor(this.rng() * best.length)];
  }

  act(s) {
    return this.rng() < this.epsilon ? Math.floor(this.rng() * 4) : this.greedy(s);
  }

  // Yields every transition so the UI can animate an episode step by step.
  *episode() {
    const env = this.env;
    let s = env.startState();
    let a = this.act(s);
    let G = 0, t = 0;
    for (; t < this.maxSteps; t++) {
      const { s2, r, done } = env.step(s, a, this.rng);
      G += r;
      const a2 = done ? 0 : this.act(s2);
      let next = 0;
      if (!done) next = this.method === 'sarsa' ? this.Q[s2 * 4 + a2] : this.value(s2);
      const i = s * 4 + a;
      this.Q[i] += this.alpha * (r + this.gamma * next - this.Q[i]);
      yield { s, a, r, s2, done };
      if (done) { t++; break; }
      s = s2;
      a = a2;
    }
    this.lastLength = t;
    this.history.push(G);
    return G;
  }

  runEpisode() {
    const it = this.episode();
    let res;
    do { res = it.next(); } while (!res.done);
    return res.value;
  }
}
