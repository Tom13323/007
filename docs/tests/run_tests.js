/*
 * Checks that js/core.js computes exactly what the Python game computes.
 *
 *     node tests/run_tests.js
 *
 * The expected results in tests/cases/ come from tools/export_from_python.py (run that first).
 * Numbers must agree to 1e-9; names, orders, chosen actions and targets must agree exactly.
 */
"use strict";
const path = require("path");
const fs = require("fs");
const C = require(path.join(__dirname, "..", "js", "core.js"));

const CASES = path.join(__dirname, "cases");
// read one file of expected results
const load = (name) => JSON.parse(fs.readFileSync(path.join(CASES, name), "utf8"));
const TOL = 1e-9;
let failures = 0;

// numbers equal to 1e-9, anything else exactly
function close(a, b) {
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= TOL * Math.max(1, Math.abs(a), Math.abs(b));
  return a === b;
}
/** Compare two JSON-like values; returns the first difference as a string, or null. */
function diff(a, b, where = "") {
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return `${where}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`;
    if (a.length !== b.length) return `${where}: length ${a.length} vs ${b.length}`;
    for (let i = 0; i < a.length; i++) { const d = diff(a[i], b[i], `${where}[${i}]`); if (d) return d; }
    return null;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.join(",") !== kb.join(",")) return `${where}: keys ${ka} vs ${kb}`;
    for (const k of ka) { const d = diff(a[k], b[k], `${where}.${k}`); if (d) return d; }
    return null;
  }
  return close(a, b) ? null : `${where}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`;
}
// print PASS / FAIL for one group of comparisons
function report(name, n, errs) {
  const ok = errs.length === 0;
  if (!ok) failures += errs.length;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: ${n} compared${ok ? "" : `, ${errs.length} different`}`);
  for (const e of errs.slice(0, 5)) console.log("        " + e);
}
// number keys -> string keys, like Python's json
const json = (x) => JSON.parse(JSON.stringify(x));       // number keys -> string keys, like Python's json

// 1. random generator
{
  const c = load("rng.json"), r = new C.Rng(c.seed), errs = [];
  c.values.forEach((v, i) => { const x = r.random(); if (x !== v) errs.push(`#${i}: ${x} vs ${v}`); });
  report("random generator (mulberry32)", c.values.length, errs);
}
// 2. rules engine on random rounds, including invalid actions
{
  const cases = load("rules_random.json"), errs = [];
  cases.forEach((c, i) => { const d = diff(json(C.resolveRound(c.players, c.actions)), c.out, `case ${i}`); if (d) errs.push(d); });
  report("rules engine, random rounds", cases.length, errs);
}
// 3. rules engine on every round of the selected recorded games
{
  const games = load("rules_games.json"), errs = [];
  let n = 0;
  games.forEach((g, gi) => {
    let players = [];
    for (let i = 0; i < g.rec.n; i++) players.push(C.makePlayer(i, "P" + i, i === 0));
    g.rec.rounds.forEach((rd, r) => {
      const out = C.resolveRound(players, C.roundActions(rd));
      const d = diff(json(out), g.outs[r], `game ${gi} round ${r + 1}`);
      if (d) errs.push(d);
      players = out.players; n++;
    });
  });
  report("rules engine, recorded games (rounds)", n, errs);
}
// 4. solved 2-player mixes
{
  const duels = load("duels.json"), errs = [];
  for (const [b1, b2, r1, r2, boost, la, lo, x, y, v] of duels) {
    const [la2, lo2, x2, y2, v2] = C.duelSolution({ bullets: b1, has_reflect: r1 }, { bullets: b2, has_reflect: r2 }, boost);
    const d = diff([la2, lo2, x2, y2, v2], [la, lo, x, y, v], `duel ${b1}v${b2} ${r1}/${r2} boost ${boost}`);
    if (d) errs.push(d);
  }
  report("solved 2-player mixes (position x bullet bonus)", duels.length, errs);
}
// 5. position scores
{
  const cases = load("eval_state.json"), errs = [];
  cases.forEach(([players, me, v], i) => { const d = diff(C.evalState(players, me), v, `case ${i}`); if (d) errs.push(d); });
  report("position scores (eval_state)", cases.length, errs);
}
// 6. bot decisions in the recorded games, memory growing game by game
{
  const data = load("bot_decisions.json"), errs = [];
  const states = {};
  for (let s = 1; s <= C.MAX_SLOT; s++) states[s] = C.newBotState(s);
  let n = 0;
  data.games.forEach((g, gi) => {
    const rec = g.rec, bots = new Map();
    for (let i = 1; i < rec.n; i++) bots.set(i, new C.ThinkingBot(i, g.difficulty, new C.Rng(g.seed_base + i), g.samples, states[i]));
    let players = [];
    for (let i = 0; i < rec.n; i++) players.push(C.makePlayer(i, i === 0 ? "You" : "Bot " + i, i === 0));
    let k = 0;
    rec.rounds.forEach((rd, r) => {
      const acts = C.roundActions(rd);
      const pub = players.map((p) => Object.assign({}, p));
      for (const [pid, b] of bots) {
        if (players[pid].alive) {
          const ch = b.choose(pub, r + 1);
          const exp = g.decisions[k++];
          const d = diff([r + 1, pid, b.last_probs, ch.action, ch.target], exp, `game ${gi} (${g.difficulty}, ${rec.n}p) round ${r + 1} bot ${pid}`);
          if (d) errs.push(d);
          n++;
        }
      }
      const out = C.resolveRound(players, acts);
      for (const b of bots.values()) b.observe(pub, out.actions);
      players = out.players;
    });
    if (k !== g.decisions.length) errs.push(`game ${gi}: ${k} decisions vs ${g.decisions.length}`);
    for (let s = 1; s < rec.n; s++) C.applyRecord(states[s], rec);
  });
  report("bot decisions: probabilities + chosen action/target", n, errs);
  const sErrs = [];
  for (let s = 1; s <= C.MAX_SLOT; s++) {
    const st = states[s], m = st.human;
    const pairs = (row) => Array.from(row, ([k, v]) => [k, v]);
    const mine = { games: st.games, wins: st.wins, losses: st.losses, draws: st.draws,
      human: { c0: pairs(m.c0), c1: Array.from(m.c1, ([k, r]) => [k, pairs(r)]), c2: Array.from(m.c2, ([k, r]) => [k, pairs(r)]),
        cs: Array.from(m.cs, ([k, r]) => [k, pairs(r)]), tgt: pairs(m.tgt), streak: m.streak, last: m.last },
      learn: Array.from(st.learn.t, ([k, acts]) => [k, Array.from(acts, ([a, c]) => [a, c])]) };
    const d = diff(json(mine), data.final_states[String(s)], `bot seat ${s} memory`);
    if (d) sErrs.push(d);
  }
  report("learned bot memory after all games (model of you + lessons)", C.MAX_SLOT, sErrs);
}
// 7. after-game re-tuning
{
  const t = load("tuning.json"), errs = [];
  const sel = load("bot_decisions.json").games.map((g) => g.rec);
  for (const w of C.GRID_W) { const d = diff(C.scorePrediction(sel, w), t.score_prediction[w.toFixed(1)], `score_prediction(${w})`); if (d) errs.push(d); }
  for (const r of C.GRID_R) { const d = diff(C.scorePayoff(sel, r), t.score_payoff[r.toFixed(1)], `score_payoff(${r})`); if (d) errs.push(d); }
  const store = new C.BotStore(new C.MemoryBackend());
  const params = C.updateParams(store, sel, 1);
  const d1 = diff({ STREAK_PRIOR_WEIGHT: params.STREAK_PRIOR_WEIGHT, PATTERN_ROUNDS: params.PATTERN_ROUNDS }, t.update, "update_params");
  if (d1) errs.push(d1);
  const entry = Object.assign({}, params.history[params.history.length - 1]);
  delete entry.time;
  const d2 = diff(json(entry), t.update_entry, "update_params entry");
  if (d2) errs.push(d2);
  report("after-game re-tuning (scores + tuning step)", C.GRID_W.length + C.GRID_R.length + 2, errs);
}
console.log(failures ? `\n${failures} DIFFERENCES` : "\nALL TESTS PASSED: the JavaScript core computes the same as the Python game");
process.exit(failures ? 1 : 0);
