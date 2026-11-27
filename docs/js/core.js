/*
 * 007 v10.2 website - game core: rules engine, solved 2-player game, opponent model, bot, learning and
 * after-game re-tuning.  A line-by-line port of 007-premode-pygame-v10.2.py and 007-train-bot.py;
 * tests/run_tests.js proves that it computes the same results as the Python game.
 * To match Python exactly: Python's sum() -> pysum(), dicts keyed by player id -> Map, max()/min() keep the first best.
 * Works in the browser (global `Core`) and in Node (require).
 */
(function (root) {
  "use strict";
  // the 676 solved values of the 2-player game (js/v2p.js)
  const V2P = (typeof module !== "undefined" && typeof require !== "undefined") ? require("./v2p.js") : root.V2P_TABLE;

  // ============================================================== constants and tuned variables
  const MG_COST = 10;
  const CAP = 12;                       // bullets above this count as CAP in the solved table
  const GAMMA = 0.985;                  // small discount: winning sooner is a little better
  const STREAK_K = 0.04;                // bullet bonus per round the bot has just defended in a row
  // the bullet bonus counts at most this many Defense rounds
  const STREAK_K_CAP = 3;
  // starting guess: P(a human defends again | it has defended 1, 2, 3, 4+ rounds in a row)
  const STREAK_PRIOR = { 1: 0.50, 2: 0.33, 3: 0.15, 4: 0.05 };
  // how a human usually splits the moves that are not Defense
  const HUMAN_REST = { reload: 1.0, shoot: 0.25, bomb: 0.10, reflect: 0.10 };
  // win / loss lessons: max effect, evidence needed, recency weight, fading per game
  const LEARN_BETA = 0.35, LEARN_K = 6.0, LEARN_RECENCY = 0.85, LEARN_GAME_DECAY = 0.98, HUMAN_GAME_DECAY = 0.97;
  // the player counts each bot seat plays in (and learns from)
  const SLOT_MODES = { 1: [2, 3, 5], 2: [3, 5], 3: [5], 4: [5] };
  // number of bot seats
  const MAX_SLOT = 4;
  // starting values of the two re-tuned variables
  const TUNABLE_DEFAULTS = { STREAK_PRIOR_WEIGHT: 6.0, PATTERN_ROUNDS: 8.0 };
  // difficulty: softmax sharpness, random mistakes, exploiting, safety limit, look-ahead samples
  const LEVELS = {
    easy: { tau: 0.45, mistake: 0.35, exploit: 0.0, tol: 0.0, samples: 60 },
    normal: { tau: 0.12, mistake: 0.03, exploit: 0.5, tol: 0.12, samples: 100 },
    hard: { tau: 0.05, mistake: 0.0, exploit: 0.85, tol: 0.30, samples: 160 },
  };
  // the two variables the after-game re-tuning adjusts (STREAK_PRIOR_WEIGHT, ThinkingBot.PATTERN_ROUNDS in Python)
  const P = { STREAK_PRIOR_WEIGHT: 6.0, PATTERN_ROUNDS: 8.0 };

  // ============================================================== Python-compatible helpers
  /** Add numbers exactly the way Python 3.12's sum() does (Neumaier compensated summation). */
  function pysum(values) {
    let f = 0.0, c = 0.0;
    for (const x of values) {
      const t = f + x;
      if (Math.abs(f) >= Math.abs(x)) c += (f - t) + x; else c += (x - t) + f;
      f = t;
    }
    if (c !== 0 && Number.isFinite(c)) f += c;
    return f;
  }
  // true if object o has its own key k
  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  /** Return the index of the first element with the largest key, like Python's max(). */
  function argmaxFirst(arr, key) {
    let bi = 0, bk = key(arr[0]);
    for (let i = 1; i < arr.length; i++) { const k = key(arr[i]); if (cmp(k, bk) > 0) { bi = i; bk = k; } }
    return bi;
  }
  /** Compare numbers or arrays of numbers / booleans like Python tuples. */
  function cmp(a, b) {
    if (Array.isArray(a)) {
      for (let i = 0; i < Math.min(a.length, b.length); i++) { const c = cmp(a[i], b[i]); if (c) return c; }
      return a.length - b.length;
    }
    const x = +a, y = +b;
    return x < y ? -1 : x > y ? 1 : 0;
  }
  /** Random numbers: the browser's, or a copy of the tests' mulberry32 generator when seeded. */
  class Rng {
    // no seed: the browser's Math.random; with a seed: the tests' mulberry32 generator
    constructor(seed) {
      if (seed === undefined || seed === null) { this.random = () => Math.random(); return; }
      let a = seed | 0;
      this.random = function () {
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    // a random element of arr
    choice(arr) { return arr[Math.floor(this.random() * arr.length)]; }
  }

  // ============================================================== LAYER 0 - rules engine
  function makePlayer(pid, name, human = false) {
    return { id: pid, name: name, bullets: 0, has_reflect: true, alive: true, human: human };
  }
  // the players still in the game
  const alivePlayers = (players) => players.filter((p) => p.alive);
  // a Bomb costs one bullet per other living player
  const bombCost = (players) => Math.max(1, alivePlayers(players).length - 1);

  // every action player p may choose right now
  function legalActions(p, players) {
    if (!p.alive) return [];
    const acts = ["reload", "defense"];
    const b = p.bullets;
    if (b >= 1) acts.push("shoot");
    if (alivePlayers(players).length >= 2 && b >= bombCost(players)) acts.push("bomb");
    if (p.has_reflect) acts.push("reflect");
    if (b >= MG_COST) acts.push("mg");
    return acts;
  }

  // the six actions
  const ACTIONS = ["reload", "shoot", "defense", "bomb", "reflect", "mg"];

  /** Resolve one round in which all players act at the same time (see resolve_round in Python). */
  function resolveRound(players, actions) {
    const nw = players.map((p) => Object.assign({}, p));
    const byId = {};
    for (const p of nw) byId[p.id] = p;
    const live = nw.filter((p) => p.alive).map((p) => p.id);
    const bcost = Math.max(1, live.length - 1);
    const acts = {};
    for (const pid of live) {
      const p = byId[pid];
      const raw = (actions instanceof Map ? actions.get(pid) : actions[pid]) || {};
      let act = has(raw, "action") ? raw.action : "reload";
      let tgt = has(raw, "target") ? raw.target : null;
      if (tgt === undefined) tgt = null;
      if (act === "shoot" && p.bullets < 1) act = "reload";
      else if (act === "mg" && p.bullets < MG_COST) act = "reload";
      else if (act === "bomb" && (p.bullets < bcost || live.length < 2)) act = "reload";
      else if (act === "reflect" && !p.has_reflect) act = "defense";
      else if (!ACTIONS.includes(act)) act = "reload";
      if (act === "shoot" || act === "mg") {
        if (!live.includes(tgt) || tgt === pid) {
          const others = live.filter((i) => i !== pid);
          tgt = others.length ? others[0] : null;
          if (tgt === null) act = "reload";
        }
      } else tgt = null;
      acts[pid] = { action: act, target: tgt };
    }
    for (const pid of live) {                                   // resources are spent immediately
      const p = byId[pid], act = acts[pid].action;
      if (act === "shoot") p.bullets -= 1;
      else if (act === "mg") p.bullets -= MG_COST;
      else if (act === "bomb") p.bullets -= bcost;
      else if (act === "reflect") p.has_reflect = false;
    }
    const attackSet = new Set(), attacks = [], mgHits = [];
    // record one Shoot / Bomb attack a -> t (once)
    const addAttack = (a, t) => { const k = a + "," + t; if (!attackSet.has(k)) { attackSet.add(k); attacks.push([a, t]); } };
    for (const pid of live) {
      const a = acts[pid];
      if (a.action === "shoot") addAttack(pid, a.target);
      else if (a.action === "bomb") { for (const t of live) if (t !== pid) addAttack(pid, t); }
      else if (a.action === "mg") mgHits.push([pid, a.target]);
    }
    // sort [attacker, target] pairs the way Python sorts tuples
    const pairSort = (x, y) => (x[0] - y[0]) || (x[1] - y[1]);
    const deaths = new Set(), events = [];
    const mgSet = new Set(mgHits.map(([a, t]) => a + "," + t));
    for (const [a, t] of mgHits.slice().sort(pairSort)) {
      if (mgSet.has(t + "," + a)) {
        if (a < t) events.push(["mg_clash", a, t]);
        deaths.add(a); deaths.add(t);
      } else { events.push(["mg", a, t]); deaths.add(t); }
    }
    const seenCancel = new Set();
    for (const [a, t] of attacks.slice().sort(pairSort)) {
      const ta = acts[t].action;
      if (ta === "reflect") { deaths.add(a); events.push(["reflected", a, t]); }
      else if (ta === "defense") events.push(["blocked", a, t]);
      else if (attackSet.has(t + "," + a)) {
        const key = Math.min(a, t) + "," + Math.max(a, t);
        if (!seenCancel.has(key)) { seenCancel.add(key); events.push(["cancel", a, t]); }
      } else { deaths.add(t); events.push(["hit", a, t]); }
    }
    for (const pid of live) {
      const p = byId[pid];
      if (deaths.has(pid)) p.alive = false;
      else if (acts[pid].action === "reload") p.bullets += 1;
    }
    return { players: nw, deaths: Array.from(deaths).sort((x, y) => x - y), events: events, actions: acts };
  }

  // ============================================================== LAYER 1 - solved 2-player game
  /** Solve a zero-sum matrix game by linear programming (simplex, Bland's rule): [row mix, column mix, value]. */
  function solveMatrixGame(A) {
    const m = A.length, n = A[0].length;
    let lo = Infinity;
    for (const r of A) for (const v of r) if (v < lo) lo = v;
    const shift = 1.0 - lo;
    let T = [];
    for (let i = 0; i < m; i++) {
      const row = [];
      for (let j = 0; j < n; j++) row.push(A[i][j] + shift);
      for (let k = 0; k < m; k++) row.push(k === i ? 1.0 : 0.0);
      row.push(1.0);
      T.push(row);
    }
    let z = [];
    for (let j = 0; j < n; j++) z.push(-1.0);
    for (let k = 0; k < m; k++) z.push(0.0);
    z.push(0.0);
    const basis = [];
    for (let i = 0; i < m; i++) basis.push(n + i);
    const eps = 1e-11, W = n + m + 1;
    for (let it = 0; it < 1000; it++) {
      let col = -1;
      for (let c = 0; c < n + m; c++) if (z[c] < -eps) { col = c; break; }
      if (col < 0) break;
      let row = -1, best = null;
      for (let i = 0; i < m; i++) {
        const a = T[i][col];
        if (a > eps) {
          const r = T[i][W - 1] / a;
          if (best === null || r < best - 1e-12 || (Math.abs(r - best) <= 1e-12 && basis[i] < basis[row])) { best = r; row = i; }
        }
      }
      if (row < 0) break;
      const pv = T[row][col];
      T[row] = T[row].map((v) => v / pv);
      for (let i = 0; i < m; i++) {
        if (i !== row) {
          const f = T[i][col];
          if (f !== 0.0) { const Ti = T[i], Tr = T[row]; const out = new Array(W); for (let k = 0; k < W; k++) out[k] = Ti[k] - f * Tr[k]; T[i] = out; }
        }
      }
      const f = z[col], Tr = T[row];
      const nz = new Array(W);
      for (let k = 0; k < W; k++) nz[k] = z[k] - f * Tr[k];
      z = nz;
      basis[row] = col;
    }
    const S = z[W - 1];
    const y = new Array(n).fill(0.0);
    basis.forEach((b, i) => { if (b < n) y[b] = T[i][W - 1]; });
    const x = [];
    for (let i = 0; i < m; i++) x.push(Math.max(0.0, z[n + i]));
    const sx = pysum(x) || 1.0, sy = pysum(y) || 1.0;
    return [x.map((v) => v / sx), y.map((v) => v / sy), 1.0 / S - shift];
  }

  // actions of one side in a 2-player position (no Bomb: with two players it equals Shoot)
  function legal2(b, r) {
    const acts = ["reload", "defense"];
    if (b >= 1) acts.push("shoot");
    if (r) acts.push("reflect");
    if (b >= MG_COST) acts.push("mg");
    return acts;
  }
  // index of a 2-player position in the solved table
  function s2idx(b1, b2, r1, r2) {
    return ((Math.min(b1, CAP) * (CAP + 1) + Math.min(b2, CAP)) * 2 + (r1 ? 1 : 0)) * 2 + (r2 ? 1 : 0);
  }
  // play one 2-player round: ["T", result] if it ends the game, else ["S", next position, bullet swing]
  function trans2(b1, b2, r1, r2, a1, a2) {
    const ps = [makePlayer(0, "a"), makePlayer(1, "b")];
    ps[0].bullets = b1; ps[1].bullets = b2;
    ps[0].has_reflect = !!r1; ps[1].has_reflect = !!r2;
    const out = resolveRound(ps, { 0: { action: a1, target: 1 }, 1: { action: a2, target: 0 } });
    const q = out.players;
    if (!q[0].alive && !q[1].alive) return ["T", 0.0, 0.0];
    if (!q[0].alive) return ["T", -1.0, 0.0];
    if (!q[1].alive) return ["T", 1.0, 0.0];
    const shape = (q[0].bullets - b1) - (q[1].bullets - b2);
    return ["S", s2idx(q[0].bullets, q[1].bullets, q[0].has_reflect, q[1].has_reflect), shape];
  }
  // solved positions, so each one is solved only once
  const DUEL_CACHE = new Map();
  /** Equilibrium of the current 2-player position: [p1 actions, p2 actions, p1 mix, p2 mix, value, table]. */
  function duelSolution(p1, p2, boost = 0) {
    const b1 = p1.bullets, b2 = p2.bullets, r1 = !!p1.has_reflect, r2 = !!p2.has_reflect;
    boost = Math.max(0, Math.min(Math.trunc(boost), STREAK_K_CAP));
    const key = b1 + "," + b2 + "," + r1 + "," + r2 + "," + boost;
    const hit = DUEL_CACHE.get(key);
    if (hit) return hit;
    const kEff = STREAK_K * boost;
    const la = legal2(b1, r1), lo = legal2(b2, r2);
    const A = la.map((a) => lo.map((o) => {
      const t = trans2(b1, b2, r1, r2, a, o);
      return t[0] === "T" ? t[1] : GAMMA * V2P[t[1]] + kEff * t[2];
    }));
    const [x, y, v] = solveMatrixGame(A);
    if (DUEL_CACHE.size > 20000) DUEL_CACHE.clear();
    const sol = [la, lo, x, y, v, A];
    DUEL_CACHE.set(key, sol);
    return sol;
  }
  // value of a 2-player position for `me`, from the solved table
  const duelValue = (me, opp) => V2P[s2idx(me.bullets, opp.bullets, me.has_reflect, opp.has_reflect)];

  // ============================================================== LAYER 2 - opponent model
  const bucket = (b) => (b === 0 ? 0 : b <= 2 ? 1 : b <= 5 ? 2 : b <= 8 ? 3 : b === 9 ? 4 : 5);
  // group a Defense streak into 1, 2, 3 or '4 or more'
  const streakBucket = (d) => Math.min(Math.trunc(d), 4);
  // key of the (bullet band, last move) table
  const lastKey = (bk, last) => bk + "|" + (last === null || last === undefined ? "None" : last);
  // add v to map[k], starting from 0
  const addTo = (map, k, v) => map.set(k, (map.get(k) || 0.0) + v);

  /** Model one opponent's habits from the moves it has made (OppMemory in Python). */
  class OppMemory {
    // decay = how much older observations still count after each new one
    constructor(decay = 0.93) {
      this.DECAY = decay;
      this.c0 = new Map(); this.c1 = new Map(); this.c2 = new Map(); this.cs = new Map(); this.tgt = new Map();
      this.last = null; this.streak = 0; this.base = null; this.base_w = 0.0;
    }
    // a new game: forget the last move and the streak, keep the counts
    resetGame() { this.last = null; this.streak = 0; }
    // attach the counts saved from earlier games as a fixed base
    attachBase(base) { this.base = base; this.base_w = pysum(base.c0.values()) > 0 ? 1.0 : 0.0; }
    // count one observed move and update the Defense streak
    update(bullets, action, target) {
      const bk = bucket(bullets), sb = streakBucket(this.streak);
      for (const k of this.c0.keys()) this.c0.set(k, this.c0.get(k) * this.DECAY);
      addTo(this.c0, action, 1.0);
      for (const [table, key] of [[this.c1, bk], [this.c2, lastKey(bk, this.last)], [this.cs, sb]]) {
        if (!table.has(key)) table.set(key, new Map());
        const row = table.get(key);
        for (const k of row.keys()) row.set(k, row.get(k) * this.DECAY);
        addTo(row, action, 1.0);
      }
      if ((action === "shoot" || action === "mg") && target !== null && target !== undefined) {
        for (const k of Array.from(this.tgt.keys())) this.tgt.set(k, this.tgt.get(k) * this.DECAY);
        addTo(this.tgt, target, 1.0);
      }
      this.streak = action === "defense" ? this.streak + 1 : 0;
      this.last = action;
    }
    // live counts + saved base counts of one table row
    _comb(name, key) {
      const out = new Map();
      for (const [src, w] of [[this, 1.0], [this.base, this.base_w]]) {
        if (src === null || w <= 0) continue;
        const t = src[name];
        const row = key === undefined ? t : t.get(key);
        if (row && row.size) for (const [k, v] of row) addTo(out, k, w * v);
      }
      return out;
    }
    /** Estimate how likely each action in `names` is: [probabilities or null, number of observations]. */
    dist(bullets, names, mergeBomb, last) {
      if (last === undefined) last = this.last;
      // count Bomb as Shoot when asked
      const merged = (row) => {
        const out = new Map();
        for (const [k, v] of row) addTo(out, mergeBomb && k === "bomb" ? "shoot" : k, v);
        return out;
      };
      const r0 = merged(this._comb("c0"));
      const nAll = pysum(r0.values());
      if (nAll <= 0) return [null, 0.0];
      const s0 = pysum(names.map((a) => r0.get(a) || 0.0));
      if (s0 <= 0) return [null, 0.0];
      let p = {};
      for (const a of names) p[a] = (r0.get(a) || 0.0) / s0;
      const bk = bucket(bullets);
      for (const row of [this._comb("c1", bk), this._comb("c2", lastKey(bk, last))]) {
        if (row.size) {
          const r = merged(row);
          const n = pysum(names.map((a) => r.get(a) || 0.0));
          const q = {};
          for (const a of names) q[a] = ((r.get(a) || 0.0) + 2.0 * p[a]) / (n + 2.0);
          p = q;
        }
      }
      return [p, nAll];
    }
    /** Probability that this opponent defends again after its current Defense streak: [p or null, n]. */
    streakDefProb(d) {
      if (d === undefined || d === null) d = this.streak;
      if (d <= 0) return [null, 0.0];
      const sb = streakBucket(d);
      const row = this._comb("cs", sb);
      const n = pysum(row.values());
      const pr = STREAK_PRIOR[sb];
      return [((row.get("defense") || 0.0) + P.STREAK_PRIOR_WEIGHT * pr) / (n + P.STREAK_PRIOR_WEIGHT), n];
    }
    // how often this opponent has shot at each player
    targetShares() {
      const tot = pysum(this.tgt.values());
      if (tot < 2.0) return [null, tot];
      const out = new Map();
      for (const [k, v] of this.tgt) out.set(k, v / tot);
      return [out, tot];
    }
    // multiply every count by f (older games count less)
    scaleInplace(f) {
      const rows = [this.c0].concat(Array.from(this.c1.values()), Array.from(this.c2.values()),
        Array.from(this.cs.values()), [this.tgt]);
      for (const row of rows) for (const k of row.keys()) row.set(k, row.get(k) * f);
    }
    /** Saved form: the counts rounded to 4 decimals (like the Python memory files), in insertion order. */
    toJSON() {
      const r4 = (v) => Number(v.toFixed(4));
      const pairs = (row) => Array.from(row, ([k, v]) => [k, r4(v)]);
      return { decay: this.DECAY, c0: pairs(this.c0), c1: Array.from(this.c1, ([k, r]) => [k, pairs(r)]),
        c2: Array.from(this.c2, ([k, r]) => [k, pairs(r)]), cs: Array.from(this.cs, ([k, r]) => [k, pairs(r)]),
        tgt: pairs(this.tgt) };
    }
    // rebuild a model from its saved form
    static fromJSON(d, decay) {
      const m = new OppMemory(decay === undefined ? (d.decay === undefined ? 0.93 : d.decay) : decay);
      const toMap = (pairs) => new Map(pairs || []);
      m.c0 = toMap(d.c0);
      for (const [k, r] of d.c1 || []) m.c1.set(k, toMap(r));
      for (const [k, r] of d.c2 || []) m.c2.set(k, toMap(r));
      for (const [k, r] of d.cs || []) m.cs.set(k, toMap(r));
      m.tgt = toMap(d.tgt);
      return m;
    }
  }

  // split the non-Defense probability the way a human usually plays
  function humanRest(names, emp = null, n = 0.0) {
    let base = {};
    for (const a of names) if (a !== "defense") base[a] = has(HUMAN_REST, a) ? HUMAN_REST[a] : 0.0;
    if (names.includes("mg")) { base.mg = 3.0; base.reload = 0.2; }
    if (emp) {
      const obs = {};
      for (const a of Object.keys(base)) obs[a] = has(emp, a) ? emp[a] : 0.0;
      const so = pysum(Object.values(obs)), sb = pysum(Object.values(base));
      if (so > 1e-9 && sb > 1e-9) {
        const w = n / (n + 8.0), nb = {};
        for (const a of Object.keys(base)) nb[a] = (1 - w) * base[a] / sb + w * obs[a] / so;
        base = nb;
      }
    }
    return base;
  }

  // set the Defense share of a distribution from the Defense-streak model
  function applyStreak(p, d, mem, restHint = null) {
    if (d <= 0 || !has(p, "defense")) return [p, false];
    const pd = mem.streakDefProb(d)[0];
    const src = (restHint && Object.keys(restHint).length) ? restHint : p;
    let rest = {};
    for (const [a, w] of Object.entries(src)) if (a !== "defense" && has(p, a)) rest[a] = w;
    if (!Object.keys(rest).length) return [p, false];
    let s = pysum(Object.values(rest));
    if (s <= 1e-12) {
      const r2 = {};
      for (const a of Object.keys(rest)) r2[a] = 1.0;
      rest = has(r2, "reload") ? { reload: 1.0 } : r2;
      s = pysum(Object.values(rest));
    }
    const out = {};
    for (const [a, w] of Object.entries(rest)) out[a] = (1.0 - pd) * w / s;
    out.defense = pd;
    return [out, true];
  }

  // ============================================================== learning from results
  const situationKey = (myB, maxOppB, humanStreak) => bucket(myB) + "|" + bucket(maxOppB) + "|" + Math.min(Math.trunc(humanStreak), 3);

  // how well each action worked out in each situation in earlier games
  class DecisionStats {
    // empty statistics: situation -> action -> [advantage sum, weight]
    constructor() { this.t = new Map(); }
    // record the outcome of one decision
    add(key, action, adv, w) {
      if (!this.t.has(key)) this.t.set(key, new Map());
      const acts = this.t.get(key);
      if (!acts.has(action)) acts.set(action, [0.0, 0.0]);
      const cell = acts.get(action);
      cell[0] += adv * w; cell[1] += w;
    }
    // multiplier the lessons give this action in this situation (1 +- at most 0.35)
    factor(key, action) {
      const acts = this.t.get(key);
      const cell = acts ? acts.get(action) : undefined;
      if (!cell) return 1.0;
      const m = Math.max(-1.0, Math.min(1.0, cell[0] / (cell[1] + LEARN_K)));
      return 1.0 + LEARN_BETA * m;
    }
    // make older games count less
    decay(f) { for (const acts of this.t.values()) for (const cell of acts.values()) { cell[0] *= f; cell[1] *= f; } }
    // saved form, rounded to 4 decimals
    toJSON() {
      const r4 = (v) => Number(v.toFixed(4));
      return Array.from(this.t, ([k, acts]) => [k, Array.from(acts, ([a, c]) => [a, [r4(c[0]), r4(c[1])]])]);
    }
    // rebuild the statistics from the saved form
    static fromJSON(d) {
      const s = new DecisionStats();
      for (const [k, acts] of d || []) s.t.set(k, new Map(acts.map(([a, c]) => [a, c.slice()])));
      return s;
    }
  }

  // ============================================================== LAYER 1+2+3 - the bot
  function softmax(vals, tau) {
    const m = Math.max.apply(null, vals);
    const e = vals.map((v) => Math.exp((v - m) / tau));
    const s = pysum(e);
    return e.map((v) => v / s);
  }
  // draw an index with the given probabilities
  function sampleIndex(probs, rng) {
    const r = rng.random();
    let acc = 0.0;
    for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r <= acc) return i; }
    return probs.length - 1;
  }
  /** Score a position for `meId` between -1 (lost) and +1 (won). */
  function evalState(players, meId) {
    const me = players.find((p) => p.id === meId);
    const live = players.filter((p) => p.alive);
    if (!live.length) return 0.0;
    if (!me.alive) return -1.0;
    const others = live.filter((p) => p.id !== meId);
    if (!others.length) return 1.0;
    if (others.length === 1) return GAMMA * duelValue(me, others[0]);
    const n = live.length;
    let v = 0.6 * (2.0 / n - 1.0);
    v += 0.04 * Math.min(me.bullets, 10) + (me.bullets >= MG_COST ? 0.15 : 0.0);
    if (me.has_reflect) v += 0.08;
    for (const o of others) {
      v -= 0.02 * Math.min(o.bullets, 10);
      if (o.has_reflect) v -= 0.02;
      if (o.bullets >= MG_COST) v -= 0.5 / (n - 1);
      else if (o.bullets === MG_COST - 1) v -= 0.2 / (n - 1);
    }
    return Math.max(-0.9, Math.min(0.9, v));
  }

  /** Play as the game's bot: solved 2-player game, opponent models, Monte-Carlo look-ahead (ThinkingBot). */
  class ThinkingBot {
    // one bot for one seat and match; state = its saved memory (or null)
    constructor(pid, difficulty = "normal", rng = null, samples = null, state = null) {
      this.pid = pid;
      this.cfg = Object.assign({}, LEVELS[difficulty]);
      if (samples) this.cfg.samples = samples;
      this.rng = rng || new Rng();
      this.mem = new Map();
      this.last_probs = [];
      this.my_streak = 0;
      this.state = state;
      if (state !== null && state !== undefined) this._mem(0).attachBase(state.human);
    }
    // in-game model of one opponent, created when needed
    _mem(pid) { if (!this.mem.has(pid)) this.mem.set(pid, new OppMemory()); return this.mem.get(pid); }
    // after a round: update the opponent models and the bot's own Defense streak
    observe(playersBefore, actions) {
      for (const p of playersBefore) {
        if (p.alive && p.id !== this.pid && has(actions, p.id)) {
          const a = actions[p.id];
          this._mem(p.id).update(p.bullets, a.action, a.target === undefined ? null : a.target);
        }
      }
      const mine = actions[this.pid];
      if (mine) this.my_streak = mine.action === "defense" ? this.my_streak + 1 : 0;
    }
    // multipliers from the lessons of earlier games
    _learnFactors(me, opps, names) {
      const out = {};
      if (this.state === null || this.state === undefined) { for (const a of names) out[a] = 1.0; return out; }
      const hm = this.mem.get(0);
      const hs = (hm !== undefined && opps.some((o) => o.id === 0)) ? hm.streak : 0;
      const key = situationKey(me.bullets, Math.max.apply(null, opps.map((o) => o.bullets)), hs);
      for (const a of names) out[a] = this.state.learn.factor(key, a);
      return out;
    }
    // this round's action: solved 2-player strategy with one opponent, look-ahead with more
    choose(players, roundNo = 0) {
      const me = players.find((p) => p.id === this.pid);
      const opps = players.filter((p) => p.alive && p.id !== this.pid);
      if (!opps.length) return { action: "reload", target: null };
      const ban = opps.every((o) => o.bullets === 0) ? ["defense", "reflect"] : [];
      if (opps.length === 1) return this._duel(me, opps[0], ban);
      return this._melee(me, opps, players, ban);
    }
    // one opponent left: equilibrium mix, leaned toward the predicted move within the safety limit
    _duel(me, opp, ban = []) {
      const [la, lo, x, y, v, A] = duelSolution(me, opp, this.my_streak);
      const mem = this._mem(opp.id);
      const [emp, n] = mem.dist(opp.bullets, lo, true);
      let q = y.slice();
      if (emp) {
        const lamM = n / (n + 3.0);
        q = lo.map((a, j) => (1 - lamM) * y[j] + lamM * emp[lo[j]]);
        const s = pysum(q);
        q = q.map((w) => w / s);
      }
      let pattern = false;
      if (opp.human) {
        const hint = humanRest(lo, emp, n);
        const qd0 = {};
        lo.forEach((a, j) => { qd0[a] = q[j]; });
        const [qd, pat] = applyStreak(qd0, mem.streak, mem, hint);
        pattern = pat;
        q = lo.map((a) => qd[a]);
      }
      const nEff = n + (pattern ? P.PATTERN_ROUNDS : 0.0);
      const ev = la.map((a, i) => pysum(lo.map((o, j) => q[j] * A[i][j])));
      const soft = softmax(ev, 0.03);
      let lam = (emp || pattern) ? this.cfg.exploit * (nEff / (nEff + 4.0)) : 0.0;
      let mix = x.slice();
      for (let k = 0; k < 8; k++) {
        mix = la.map((a, i) => (1 - lam) * x[i] + lam * soft[i]);
        let worst = Infinity;
        for (let j = 0; j < lo.length; j++) { const w = pysum(la.map((a, i) => mix[i] * A[i][j])); if (w < worst) worst = w; }
        if (worst >= v - this.cfg.tol) break;
        lam *= 0.6;
      }
      const fac = this._learnFactors(me, [opp], la);
      mix = mix.map((m, i) => m * fac[la[i]]);
      let s = pysum(mix);
      mix = mix.map((m) => m / s);
      const mk = this.cfg.mistake;
      mix = mix.map((m) => (1 - mk) * m + mk / la.length);
      if (ban.length) {
        mix = mix.map((m, i) => (ban.includes(la[i]) ? 0.0 : m));
        if (pysum(mix) <= 1e-9) mix = la.map((a) => (ban.includes(a) ? 0.0 : 1.0));
      }
      s = pysum(mix);
      mix = mix.map((m) => m / s);
      this.last_probs = la.map((a, i) => [a, mix[i]]).sort((p1, p2) => (p2[1] - p1[1]));   // stable, like Python
      const a = la[sampleIndex(mix, this.rng)];
      return { action: a, target: (a === "shoot" || a === "mg") ? opp.id : null };
    }
    // predict one opponent's next move for the look-ahead: [action, target, probability]
    _oppProfile(o, alive, viewer) {
      const rivals = alive.filter((p) => p.id !== o.id);
      const rival = rivals[argmaxFirst(rivals, (p) => [p.bullets, p.id === viewer.id ? 1 : 0])];
      const [la, lo, x] = duelSolution(o, rival);
      let prior = {};
      la.forEach((a, i) => { prior[a] = x[i]; });
      const crowd = Math.max(0, rivals.filter((p) => p.bullets >= 1).length - 1);
      prior.reload /= (1.0 + 0.6 * crowd);
      prior.defense *= (1.0 + 0.3 * crowd);
      const n = alive.length;
      if (o.bullets >= n - 1 && has(prior, "shoot")) {
        const mv = prior.shoot * 0.25;
        prior.shoot -= mv;
        prior.bomb = mv;
      }
      let s = pysum(Object.values(prior));
      const np0 = {};
      for (const [a, w] of Object.entries(prior)) np0[a] = w / s;
      prior = np0;
      const mem = this.mem.get(o.id);
      if (mem) {
        const [emp, nn] = mem.dist(o.bullets, Object.keys(prior), false);
        if (emp) {
          const lam = nn / (nn + (o.human ? 6.0 : 10.0));
          const np1 = {};
          for (const a of Object.keys(prior)) np1[a] = (1 - lam) * prior[a] + lam * emp[a];
          prior = np1;
        }
      }
      if (mem && o.human) {
        const names = Object.keys(prior);
        const [e2, n2] = mem.dist(o.bullets, names, false);
        prior = applyStreak(prior, mem.streak, mem, humanRest(names, e2, n2))[0];
      }
      let tw = new Map();
      for (const p of rivals) tw.set(p.id, 1.0 + 0.5 * p.bullets + (p.bullets >= 8 ? 1.5 : 0.0));
      if (mem) {
        const [shares, cnt] = mem.targetShares();
        if (shares) {
          const mu = cnt / (cnt + 6.0);
          const tot = pysum(tw.values());
          const tw2 = new Map();
          for (const [t, w] of tw) tw2.set(t, (1 - mu) * w / tot + mu * (shares.get(t) || 0.0));
          tw = tw2;
        }
      }
      const tsum = pysum(tw.values());
      const out = [];
      for (const [a, pa] of Object.entries(prior)) {
        if (pa <= 0) continue;
        if (a === "shoot" || a === "mg") { for (const [t, w] of tw) out.push([a, t, pa * w / tsum]); }
        else out.push([a, null, pa]);
      }
      return out;
    }
    // two or more opponents: try every own action against sampled guesses of the others
    _melee(me, opps, players, ban = []) {
      const alive = players.filter((p) => p.alive);
      const rng = this.rng, cfg = this.cfg;
      const cum = [];
      for (const o of opps) {
        const dist = this._oppProfile(o, alive, me);
        const c = [];
        let acc = 0.0;
        for (const [, , p] of dist) { acc += p; c.push(acc); }
        cum.push([o.id, dist, c, acc]);
      }
      const profiles = [];
      for (let s = 0; s < cfg.samples; s++) {
        const prof = new Map();
        for (const [oid, dist, c, tot] of cum) {
          const r = rng.random() * tot;
          let k = 0;
          while (k < c.length - 1 && r > c[k]) k++;
          prof.set(oid, { action: dist[k][0], target: dist[k][1] });
        }
        profiles.push(prof);
      }
      const cands = [];
      for (const a of legalActions(me, players)) {
        if (ban.includes(a)) continue;
        if (a === "shoot" || a === "mg") { for (const o of opps) cands.push([a, o.id]); }
        else cands.push([a, null]);
      }
      const ev = [];
      for (const [a, t] of cands) {
        let tot = 0.0;
        for (const prof of profiles) {
          const acts = new Map(prof);
          acts.set(me.id, { action: a, target: t });
          tot += evalState(resolveRound(players, acts).players, me.id);
        }
        ev.push(tot / profiles.length);
      }
      let probs = softmax(ev, cfg.tau);
      const fac = this._learnFactors(me, opps, Array.from(new Set(cands.map((c) => c[0]))));
      probs = probs.map((p, i) => p * fac[cands[i][0]]);
      const s = pysum(probs);
      probs = probs.map((p) => p / s);
      const mk = cfg.mistake;
      probs = probs.map((p) => (1 - mk) * p + mk / cands.length);
      const agg = {};
      cands.forEach(([a], i) => { agg[a] = (agg[a] || 0.0) + probs[i]; });
      this.last_probs = Object.entries(agg).sort((p1, p2) => (p2[1] - p1[1]));
      const [a, t] = cands[sampleIndex(probs, rng)];
      return { action: a, target: t };
    }
  }

  // ============================================================== memory of the bot seats + records
  function newBotState(slot) {
    return { slot: slot, modes: SLOT_MODES[slot].slice(), games: 0, wins: 0, losses: 0, draws: 0,
      human: new OppMemory(1.0), learn: new DecisionStats() };
  }
  // saved form of one seat's memory
  const botStateToJSON = (st) => ({ version: 1, slot: st.slot, modes_players: st.modes, games: st.games,
    bot_wins: st.wins, bot_losses: st.losses, draws: st.draws, human_model: st.human.toJSON(),
    decision_stats: st.learn.toJSON() });
  // rebuild one seat's memory from its saved form
  function botStateFromJSON(d, slot) {
    const st = newBotState(slot);
    st.games = d.games || 0; st.wins = d.bot_wins || 0; st.losses = d.bot_losses || 0; st.draws = d.draws || 0;
    st.human = OppMemory.fromJSON(d.human_model || {}, 1.0);
    st.learn = DecisionStats.fromJSON(d.decision_stats || []);
    return st;
  }
  /** Turn a record's round (string keys) into an actions object with number keys. */
  function roundActions(rd) {
    const acts = {};
    for (const k of Object.keys(rd).map(Number).sort((a, b) => a - b)) acts[k] = rd[String(k)];
    return acts;
  }
  /** Let one bot seat learn from one finished game (apply_record in Python). */
  function applyRecord(st, rec, weight = 1.0, sim = false) {
    const slot = st.slot, n = rec.n;
    if (slot >= n) return false;
    const hm = st.human, ls = st.learn;
    if (!sim) { hm.scaleInplace(HUMAN_GAME_DECAY); ls.decay(LEARN_GAME_DECAY); hm.resetGame(); }
    let players = [];
    for (let i = 0; i < n; i++) players.push(makePlayer(i, "P" + i));
    let hs = 0;
    const decisions = [];
    for (const rd of rec.rounds) {
      const acts = roundActions(rd);
      const me = players[slot];
      if (me.alive && has(acts, slot)) {
        const others = players.filter((p) => p.alive && p.id !== slot).map((p) => p.bullets);
        const mo = others.length ? Math.max.apply(null, others) : 0;
        decisions.push([situationKey(me.bullets, mo, hs), acts[slot].action]);
      }
      const h = players[0];
      if (h.alive && has(acts, 0)) {
        if (!sim) hm.update(h.bullets, acts[0].action, acts[0].target === undefined ? null : acts[0].target);
        hs = acts[0].action === "defense" ? hs + 1 : 0;
      } else hs = 0;
      players = resolveRound(players, acts).players;
    }
    const winner = rec.winner === undefined ? null : rec.winner;
    const res = winner === slot ? 1.0 : (winner === null ? 0.0 : -1.0);
    const adv = res - (2.0 / n - 1.0);
    decisions.slice().reverse().forEach(([key, action], j) => {
      const w = weight * Math.pow(LEARN_RECENCY, j);
      ls.add(key, action, adv * w, w);
    });
    if (!sim) { st.games += 1; st.wins += res > 0 ? 1 : 0; st.losses += res < 0 ? 1 : 0; st.draws += res === 0 ? 1 : 0; }
    return true;
  }
  // local time as YYYY-MM-DD HH:MM:SS
  function timeString(d = new Date()) {
    const z = (v) => String(v).padStart(2, "0");
    return `${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())} ${z(d.getHours())}:${z(d.getMinutes())}:${z(d.getSeconds())}`;
  }
  // the record of a finished game (saved, and used by REPLAY)
  function makeRecord(recId, cfg, n, history, winner) {
    return { v: 1, id: recId, time: timeString(), n: n, mode: n - 1, difficulty: cfg.difficulty || "normal",
      timer: cfg.timer || 10, winner: winner,
      rounds: history.map((r) => { const o = {}; for (const pid of Object.keys(r)) o[pid] = { action: r[pid].action, target: r[pid].target === undefined ? null : r[pid].target }; return o; }) };
  }
  // put the tuned variables into force
  function applyParams(params) {
    P.STREAK_PRIOR_WEIGHT = Number(params.STREAK_PRIOR_WEIGHT);
    P.PATTERN_ROUNDS = Number(params.PATTERN_ROUNDS);
  }

  /** Read and write the bots' memory and the game records (the browser's localStorage, or memory in tests). */
  class BotStore {
    // backend = the browser's localStorage, or a MemoryBackend
    constructor(backend, prefix = "007-v10.2:") {
      this.b = backend || new MemoryBackend();
      this.prefix = prefix;
    }
    // read one stored JSON value
    _get(k) { try { const s = this.b.getItem(this.prefix + k); return s ? JSON.parse(s) : null; } catch (e) { return null; } }
    // write one stored JSON value
    _set(k, v) { this.b.setItem(this.prefix + k, JSON.stringify(v)); }
    // one seat's memory, or an empty memory
    load(slot) { const d = this._get("bot" + slot); try { return d ? botStateFromJSON(d, slot) : newBotState(slot); } catch (e) { return newBotState(slot); } }
    // save one seat's memory
    save(st) { this._set("bot" + st.slot, botStateToJSON(st)); }
    // every recorded game
    records() { const r = this._get("records"); return Array.isArray(r) ? r : []; }
    // number of the next game record
    nextId() { return this.records().length + 1; }
    // the tuned variables (defaults where nothing is stored)
    loadParams() {
      const out = Object.assign({}, TUNABLE_DEFAULTS);
      const d = this._get("params");
      if (d) {
        for (const k of Object.keys(TUNABLE_DEFAULTS)) if (typeof d[k] === "number" && d[k] >= 0) out[k] = d[k];
        out.history = d.history || [];
      }
      return out;
    }
    // store the tuned variables
    saveParams(params) { this._set("params", params); }
    // save a finished game and let every seat that played learn from it
    addGame(rec) {
      const recs = this.records();
      recs.push(rec);
      this._set("records", recs);
      for (let slot = 1; slot <= MAX_SLOT; slot++) {
        if (slot < rec.n) { const st = this.load(slot); applyRecord(st, rec); this.save(st); }
      }
    }
    // delete everything this game has stored
    clear() { for (const k of ["records", "params", "bot1", "bot2", "bot3", "bot4"]) this.b.removeItem(this.prefix + k); }
    // count the recorded games, wins, losses and draws
    static summary(records) {
      const s = { games: records.length, wins: 0, losses: 0, draws: 0, by_mode: {} };
      for (const r of records) {
        const w = r.winner === undefined ? null : r.winner;
        const k = w === 0 ? "wins" : (w === null ? "draws" : "losses");
        s[k] += 1;
        const m = s.by_mode[r.n] || (s.by_mode[r.n] = [0, 0, 0]);
        m[w === 0 ? 0 : (w === null ? 2 : 1)] += 1;
      }
      return s;
    }
  }
  // storage in memory only (tests, or when the browser blocks localStorage)
  class MemoryBackend {
    constructor() { this.m = new Map(); }
    getItem(k) { return this.m.has(k) ? this.m.get(k) : null; }
    setItem(k, v) { this.m.set(k, String(v)); }
    removeItem(k) { this.m.delete(k); }
  }

  // ============================================================== after-game re-tuning (007-train-bot.py)
  const GRID_W = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0];
  // values tried for PATTERN_ROUNDS
  const GRID_R = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0];
  // evidence needed, and the gain needed, before a tuned value may change
  const MIN_MOVES = 100, MIN_DUELS = 50, MIN_GAIN_LOSS = 0.0005, MIN_GAIN_PAYOFF = 0.002;

  // the model's prediction of the human's next move
  function predictHuman(mem, bullets, legal) {
    const [emp, n] = mem.dist(bullets, legal, false);
    let p = {};
    if (emp) Object.assign(p, emp); else for (const a of legal) p[a] = 1.0;
    const s = pysum(Object.values(p));
    const q = {};
    for (const [a, w] of Object.entries(p)) q[a] = w / s;
    return applyStreak(q, mem.streak, mem, humanRest(legal, emp, n))[0];
  }
  // how well the model predicts the human's moves with prior weight w (lower = better)
  function scorePrediction(records, w) {
    const old = P.STREAK_PRIOR_WEIGHT;
    P.STREAK_PRIOR_WEIGHT = w;
    try {
      const st = newBotState(1);
      let loss = 0.0, uni = 0.0, cnt = 0;
      for (const rec of records) {
        const mem = new OppMemory();
        mem.attachBase(st.human);
        let players = [];
        for (let i = 0; i < rec.n; i++) players.push(makePlayer(i, "P" + i, i === 0));
        for (const rd of rec.rounds) {
          const acts = roundActions(rd);
          const h = players[0];
          if (h.alive && has(acts, 0)) {
            const legal = legalActions(h, players);
            const a = acts[0].action;
            const p = predictHuman(mem, h.bullets, legal);
            if (has(p, a)) { loss -= Math.log(Math.max(p[a], 1e-4)); uni += Math.log(legal.length); cnt += 1; }
            mem.update(h.bullets, a, acts[0].target === undefined ? null : acts[0].target);
          }
          players = resolveRound(players, acts).players;
        }
        applyRecord(st, rec);
      }
      return cnt ? [loss / cnt, uni / cnt, cnt] : [null, null, 0];
    } finally { P.STREAK_PRIOR_WEIGHT = old; }
  }
  // the Hard bot's average payoff against the human's real 1-v-1 moves with PATTERN_ROUNDS pr
  function scorePayoff(records, pr) {
    const old = P.PATTERN_ROUNDS;
    P.PATTERN_ROUNDS = pr;
    try {
      const states = {};
      for (let s = 1; s <= MAX_SLOT; s++) states[s] = newBotState(s);
      let tot = 0.0, cnt = 0;
      for (const rec of records) {
        const n = rec.n;
        const bots = new Map();
        for (let i = 1; i < n; i++) bots.set(i, new ThinkingBot(i, "hard", new Rng(1), null, states[i]));
        let players = [];
        for (let i = 0; i < n; i++) players.push(makePlayer(i, "P" + i, i === 0));
        rec.rounds.forEach((rd, idx) => {
          const rnd = idx + 1;
          const acts = roundActions(rd);
          const pub = players.map((p) => Object.assign({}, p));
          const live = players.filter((p) => p.alive);
          if (live.length === 2 && players[0].alive && has(acts, 0)) {
            const pid = live.find((p) => p.id !== 0).id;
            const b = bots.get(pid);
            b.choose(pub, rnd);
            const mix = {};
            for (const [a, p] of b.last_probs) mix[a] = p;
            const [la, lo, , , , A] = duelSolution(pub[pid], pub[0], b.my_streak);
            let ha = acts[0].action;
            if (ha === "bomb") ha = "shoot";
            const j = lo.indexOf(ha);
            if (j >= 0) { tot += pysum(la.map((a, i) => (has(mix, a) ? mix[a] : 0.0) * A[i][j])); cnt += 1; }
          }
          const out = resolveRound(players, acts);
          for (const b of bots.values()) b.observe(pub, out.actions);
          players = out.players;
        });
        for (let s = 1; s < n; s++) applyRecord(states[s], rec);
      }
      return cnt ? [tot / cnt, cnt] : [null, 0];
    } finally { P.PATTERN_ROUNDS = old; }
  }
  // step from cur toward best on the grid, at most maxStep steps
  function _move(grid, cur, best, maxStep) {
    let ic = 0;
    for (let i = 1; i < grid.length; i++) if (Math.abs(grid[i] - cur) < Math.abs(grid[ic] - cur)) ic = i;
    let ib = grid.indexOf(best);
    if (maxStep !== null && maxStep !== undefined) ib = ic + Math.max(-maxStep, Math.min(maxStep, ib - ic));
    return grid[ib];
  }
  // numbers sorted, duplicates removed
  const sortedUnique = (arr) => Array.from(new Set(arr)).sort((a, b) => a - b);
  /** Re-fit STREAK_PRIOR_WEIGHT and PATTERN_ROUNDS on the recorded games (update_params in Python). */
  function updateParams(store, records, maxStep = null) {
    const params = store.loadParams();
    const curW = params.STREAK_PRIOR_WEIGHT, curR = params.PATTERN_ROUNDS;
    const entry = { time: timeString(), games: records.length };
    let newW = curW, newR = curR;
    const gw = sortedUnique(GRID_W.concat([curW]));
    const sc = new Map(gw.map((w) => [w, scorePrediction(records, w)]));
    const moves = sc.get(curW)[2];
    entry.moves = moves; entry.uniform_loss = sc.get(curW)[1];
    if (moves >= MIN_MOVES) {
      const best = gw[argmaxFirst(gw, (w) => -sc.get(w)[0])];
      if (sc.get(curW)[0] - sc.get(best)[0] >= MIN_GAIN_LOSS) newW = _move(gw, curW, best, maxStep);
      entry.pred_loss = sc.get(newW)[0];
    }
    P.STREAK_PRIOR_WEIGHT = newW;
    try {
      const gr = sortedUnique(GRID_R.concat([curR]));
      const ps = new Map(gr.map((r) => [r, scorePayoff(records, r)]));
      const duels = ps.get(curR)[1];
      entry.duels = duels;
      if (duels >= MIN_DUELS) {
        const best = gr[argmaxFirst(gr, (r) => ps.get(r)[0])];
        if (ps.get(best)[0] - ps.get(curR)[0] >= MIN_GAIN_PAYOFF) newR = _move(gr, curR, best, maxStep);
        entry.payoff = ps.get(newR)[0];
      }
    } finally { P.STREAK_PRIOR_WEIGHT = curW; }
    entry.STREAK_PRIOR_WEIGHT = newW; entry.PATTERN_ROUNDS = newR;
    params.STREAK_PRIOR_WEIGHT = newW; params.PATTERN_ROUNDS = newR;
    params.history = (params.history || []).concat([entry]).slice(-50);
    store.saveParams(params);
    return params;
  }
  /** Re-fit the tuned variables after a game, at most one small step (retune_after_game in Python). */
  const retuneAfterGame = (store) => updateParams(store, store.records(), 1);

  const api = {
    MG_COST, CAP, GAMMA, LEVELS, P, MAX_SLOT, TUNABLE_DEFAULTS, GRID_W, GRID_R,
    pysum, Rng, makePlayer, alivePlayers, bombCost, legalActions, resolveRound,
    solveMatrixGame, legal2, s2idx, trans2, duelSolution, duelValue,
    bucket, streakBucket, OppMemory, humanRest, applyStreak, situationKey, DecisionStats,
    softmax, sampleIndex, evalState, ThinkingBot,
    newBotState, botStateToJSON, botStateFromJSON, roundActions, applyRecord, makeRecord, applyParams,
    BotStore, MemoryBackend, scorePrediction, scorePayoff, updateParams, retuneAfterGame,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Core = api;
})(typeof window !== "undefined" ? window : this);
