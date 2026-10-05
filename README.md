# Gridworld RL Lab

**Live app:** https://deeps45.github.io/rl-gridworld/

An interactive gridworld for comparing Value Iteration, Policy Iteration, Q-learning and SARSA. You can draw walls, goals and pits, change γ, slip probability, α and ε, and watch the value heatmap, the greedy-policy arrows and the learning curves update. Everything runs in the browser.

- `js/env.js`: the gridworld MDP (stochastic slip, terminal rewards) and a seeded PRNG
- `js/algorithms.js`: VI (in-place sweeps), PI (evaluate + improve), Q-learning and SARSA
- `js/render.js`, `js/chart.js`: canvas heatmap, Q-triangle view and learning-curve chart
- `test/`: algorithm tests (`npm test`)
