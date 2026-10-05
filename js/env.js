// Gridworld MDP: cells are empty, wall, goal, pit, or start.
// Goal and pit are terminal. Moving into a wall or off the grid leaves the agent in place.

export const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
export const ACTION_NAMES = ['Up', 'Right', 'Down', 'Left'];

const CHAR_TO_CELL = { '.': 'empty', '#': 'wall', 'G': 'goal', 'P': 'pit', 'S': 'start' };

export class GridWorld {
  constructor(rows, cols, cells, opts = {}) {
    this.rows = rows;
    this.cols = cols;
    this.cells = cells ?? new Array(rows * cols).fill('empty');
    this.stepReward = opts.stepReward ?? -0.1;
    this.goalReward = opts.goalReward ?? 10;
    this.pitReward = opts.pitReward ?? -10;
    this.slip = opts.slip ?? 0;
  }

  static fromStrings(lines, opts) {
    const rows = lines.length, cols = lines[0].length;
    const cells = [];
    for (const line of lines) for (const ch of line) cells.push(CHAR_TO_CELL[ch] ?? 'empty');
    return new GridWorld(rows, cols, cells, opts);
  }

  get nStates() { return this.rows * this.cols; }
  rc(s) { return [Math.floor(s / this.cols), s % this.cols]; }
  idx(r, c) { return r * this.cols + c; }
  isWall(s) { return this.cells[s] === 'wall'; }
  isTerminal(s) { return this.cells[s] === 'goal' || this.cells[s] === 'pit'; }

  startState() {
    const s = this.cells.indexOf('start');
    if (s >= 0) return s;
    return this.cells.findIndex((c) => c === 'empty');
  }

  move(s, a) {
    const [r, c] = this.rc(s);
    const nr = r + ACTIONS[a][0], nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= this.rows || nc < 0 || nc >= this.cols) return s;
    const s2 = this.idx(nr, nc);
    return this.isWall(s2) ? s : s2;
  }

  reward(s2) {
    const bonus = this.cells[s2] === 'goal' ? this.goalReward : this.cells[s2] === 'pit' ? this.pitReward : 0;
    return this.stepReward + bonus;
  }

  // Full transition model: the intended move with prob 1 - slip, and each
  // perpendicular move with prob slip / 2. Outcomes landing in the same cell are merged.
  transitions(s, a) {
    const outcomes = [[a, 1 - this.slip], [(a + 1) % 4, this.slip / 2], [(a + 3) % 4, this.slip / 2]];
    const merged = new Map();
    for (const [act, p] of outcomes) {
      if (p <= 0) continue;
      const s2 = this.move(s, act);
      merged.set(s2, (merged.get(s2) ?? 0) + p);
    }
    return [...merged].map(([s2, p]) => ({ p, s2, r: this.reward(s2), done: this.isTerminal(s2) }));
  }

  step(s, a, rng) {
    const ts = this.transitions(s, a);
    let u = rng(), acc = 0;
    for (const t of ts) {
      acc += t.p;
      if (u < acc) return t;
    }
    return ts[ts.length - 1];
  }
}

// Small seeded PRNG so a training run can be replayed exactly.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const PRESETS = {
  'Classic 4×3': {
    lines: ['...G', '.#.P', 'S...'],
    opts: { stepReward: -0.04, goalReward: 1, pitReward: -1, slip: 0.2 },
  },
  'Cliff walk': {
    lines: ['..........', '..........', '..........', 'SPPPPPPPPG'],
    opts: { stepReward: -1, goalReward: 0, pitReward: -100, slip: 0 },
  },
  'Four rooms': {
    lines: [
      '.....#.....',
      '.....#.....',
      '...........',
      '.....#.....',
      '.....#.....',
      '#.####.....',
      '.....###.##',
      '.....#.....',
      '.....#.....',
      '...........',
      'S....#....G',
    ],
    opts: { stepReward: -0.1, goalReward: 10, pitReward: -10, slip: 0 },
  },
  'Maze': {
    lines: [
      'S.#.....',
      '.##.###.',
      '....#...',
      '###.#.#.',
      '....#.#P',
      '.####.#.',
      '......#G',
    ],
    opts: { stepReward: -0.1, goalReward: 10, pitReward: -10, slip: 0 },
  },
};
