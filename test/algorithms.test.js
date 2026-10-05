import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GridWorld, PRESETS, mulberry32 } from '../js/env.js';
import { ValueIteration, PolicyIteration, TDControl, argmaxAll } from '../js/algorithms.js';

function solve(Agent, env, opts) {
  const agent = new Agent(env, opts);
  for (let i = 0; i < 2000 && !agent.converged; i++) agent.step();
  assert.ok(agent.converged, 'did not converge');
  return agent;
}

test('transition probabilities sum to 1 and merge wall bounces', () => {
  const env = GridWorld.fromStrings(['S.', '#G'], { slip: 0.2 });
  for (let s = 0; s < env.nStates; s++) {
    if (env.isWall(s) || env.isTerminal(s)) continue;
    for (let a = 0; a < 4; a++) {
      const total = env.transitions(s, a).reduce((x, t) => x + t.p, 0);
      assert.ok(Math.abs(total - 1) < 1e-12);
    }
  }
  // From the top-left corner, "up" bounces off the edge, and slipping left also bounces: 0.8 + 0.1 stay.
  const up = env.transitions(0, 0);
  assert.equal(up.find((t) => t.s2 === 0).p.toFixed(2), '0.90');
});

test('value iteration on a deterministic corridor gives discounted rewards', () => {
  // S . . G with step reward 0, goal reward 1, gamma 0.9.
  const env = GridWorld.fromStrings(['S..G'], { stepReward: 0, goalReward: 1, slip: 0 });
  const vi = solve(ValueIteration, env, { gamma: 0.9, theta: 1e-10 });
  assert.ok(Math.abs(vi.V[2] - 1) < 1e-9);
  assert.ok(Math.abs(vi.V[1] - 0.9) < 1e-9);
  assert.ok(Math.abs(vi.V[0] - 0.81) < 1e-9);
  assert.deepEqual(argmaxAll(vi.qValues(0)), [1]); // go right
});

test('policy iteration agrees with value iteration on the stochastic 4x3 world', () => {
  const { lines, opts } = PRESETS['Classic 4×3'];
  const env = GridWorld.fromStrings(lines, opts);
  const vi = solve(ValueIteration, env, { gamma: 0.99, theta: 1e-10 });
  const pi = solve(PolicyIteration, env, { gamma: 0.99, theta: 1e-10 });
  for (let s = 0; s < env.nStates; s++) {
    assert.ok(Math.abs(vi.V[s] - pi.V[s]) < 1e-5, `state ${s}: ${vi.V[s]} vs ${pi.V[s]}`);
  }
  // Textbook result (Russell & Norvig): from start, the optimal utility is ~0.705 with gamma=1;
  // with gamma=0.99 it's slightly lower but still clearly positive.
  assert.ok(vi.V[env.startState()] > 0.6);
});

test('Q-learning finds the cliff-edge path; SARSA trains on the cliff without crashing', () => {
  const { lines, opts } = PRESETS['Cliff walk'];
  const env = GridWorld.fromStrings(lines, opts);
  const q = new TDControl(env, { method: 'qlearning', gamma: 1, alpha: 0.5, epsilon: 0.1, rng: mulberry32(1) });
  for (let i = 0; i < 500; i++) q.runEpisode();
  // Greedy rollout along the optimal path: up, 9×right, down = 11 steps, return -11 (goal reward 0, step -1).
  q.epsilon = 0;
  let s = env.startState(), G = 0;
  for (let t = 0; t < 50; t++) {
    const { s2, r, done } = env.step(s, q.greedy(s), Math.random);
    G += r; s = s2;
    if (done) break;
  }
  assert.equal(G, -11);

  const sarsa = new TDControl(env, { method: 'sarsa', gamma: 1, alpha: 0.5, epsilon: 0.1, rng: mulberry32(1) });
  for (let i = 0; i < 300; i++) sarsa.runEpisode();
  assert.equal(sarsa.history.length, 300);
});

test('same seed gives identical training runs', () => {
  const { lines, opts } = PRESETS['Four rooms'];
  const run = () => {
    const env = GridWorld.fromStrings(lines, opts);
    const a = new TDControl(env, { rng: mulberry32(42) });
    for (let i = 0; i < 50; i++) a.runEpisode();
    return a.history;
  };
  assert.deepEqual(run(), run());
});
