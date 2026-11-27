"""
Export what the website needs from the Python game, plus the test cases that prove the JavaScript
version computes the same things.

    python tools/export_from_python.py          (run from anywhere)

Writes
  js/v2p.js              the 676 solved values of the 2-player game (used by the website)
  tests/cases/*.json     test cases for tests/run_tests.js.  They contain games from bot_data_v10,
                         so they stay on this computer: never upload tests/cases/.

The Python game and bot_data_v10 are only read, never changed.
"""
import importlib.util
import json
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
ROOT = os.path.dirname(SITE)                                   # the 007 folder
GAME = os.path.join(ROOT, "007-premode-pygame-v10.2.py")
RECORDS = os.path.join(ROOT, "bot_data_v10", "game_records.jsonl")
CASES = os.path.join(SITE, "tests", "cases")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
spec = importlib.util.spec_from_file_location("game007", GAME)
g = importlib.util.module_from_spec(spec)
sys.modules["game007"] = g
spec.loader.exec_module(g)
spec_t = importlib.util.spec_from_file_location("train007", os.path.join(ROOT, "007-train-bot.py"))
tb = importlib.util.module_from_spec(spec_t)
spec_t.loader.exec_module(tb)


# ---------------------------------------------------------------- a random generator that JS copies exactly
M32 = 0xFFFFFFFF


def u32(x):
    """Keep the lowest 32 bits (unsigned)."""
    return x & M32


def i32(x):
    """Keep the lowest 32 bits as a signed number, like JavaScript's `| 0`."""
    x &= M32
    return x - (1 << 32) if x >= (1 << 31) else x


def imul(x, y):
    """32-bit multiplication, like JavaScript's Math.imul."""
    return i32(u32(x) * u32(y))


class TestRng:
    """mulberry32: only integer operations, so Python and JavaScript produce exactly the same numbers."""

    def __init__(self, seed):
        """Start the generator from an integer seed."""
        self.a = i32(seed)

    def random(self):
        """Return the next number in [0, 1)."""
        self.a = i32(self.a + 0x6D2B79F5)
        a = self.a
        t = imul(a ^ (u32(a) >> 15), 1 | a)
        t = i32((t + imul(t ^ (u32(t) >> 7), 61 | t)) ^ t)
        return u32(t ^ (u32(t) >> 14)) / 4294967296


# ---------------------------------------------------------------- helpers
def dump(name, obj):
    """Write one file of expected results into tests/cases/."""
    os.makedirs(CASES, exist_ok=True)
    with open(os.path.join(CASES, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"  {name:<24} {os.path.getsize(os.path.join(CASES, name)) / 1024:8.0f} KB")


def mem_dump(m):
    """OppMemory with full precision and its insertion order (lists of pairs)."""
    pairs = lambda row: [[k, v] for k, v in row.items()]
    return {"c0": pairs(m.c0), "c1": [[k, pairs(r)] for k, r in m.c1.items()],
            "c2": [[f"{k[0]}|{k[1]}", pairs(r)] for k, r in m.c2.items()],
            "cs": [[k, pairs(r)] for k, r in m.cs.items()], "tgt": pairs(m.tgt),
            "streak": m.streak, "last": m.last}


def state_dump(st):
    """One bot seat's memory (counts, games, lessons) with full precision."""
    return {"games": st["games"], "wins": st["wins"], "losses": st["losses"], "draws": st["draws"],
            "human": mem_dump(st["human"]),
            "learn": [[k, [[a, c] for a, c in acts.items()]] for k, acts in st["learn"].t.items()]}


def main():
    """Export the solved table and every group of test cases."""
    recs = [json.loads(l) for l in open(RECORDS, encoding="utf-8") if l.strip()]
    by_n = {n: [r for r in recs if r["n"] == n] for n in (2, 3, 5)}
    longest = max(recs, key=lambda r: len(r["rounds"]))
    sel = by_n[2] + by_n[3] + by_n[5][:20]
    if longest not in sel:
        sel.append(longest)
    print(f"{len(recs)} records read; {len(sel)} used for the tests")

    # 1. the solved table, for the website
    V = g.get_v2p()
    with open(os.path.join(SITE, "js", "v2p.js"), "w", encoding="utf-8") as f:
        f.write("// The 676 solved values of the 2-player game, exported from 007-premode-pygame-v10.2.py\n"
                "// by tools/export_from_python.py.  Do not edit by hand.\n")
        f.write("var V2P_TABLE = " + json.dumps(V) + ";\n")
        f.write("if (typeof module !== 'undefined') module.exports = V2P_TABLE;\n")
    print(f"  js/v2p.js: {len(V)} values")

    # 2. random generator check
    r = TestRng(12345)
    dump("rng.json", {"seed": 12345, "values": [r.random() for _ in range(50)]})

    # 3. rules engine: random rounds (also invalid actions) + every round of the selected games
    rng = random.Random(7)
    kinds = ["reload", "defense", "shoot", "bomb", "reflect", "mg", "nonsense", None]
    cases = []
    for _ in range(4000):
        n = rng.choice([2, 3, 4, 5])
        players = [g.make_player(i, f"P{i}", i == 0) for i in range(n)]
        for p in players:
            p["bullets"] = rng.choice([0, 0, 1, 1, 2, 3, 5, 9, 10, 11, 12, 14])
            p["has_reflect"] = rng.random() < 0.6
            p["alive"] = rng.random() < 0.85
        acts = {}
        for p in players:
            if rng.random() < 0.05:
                continue                                   # no action given -> Reload
            k = rng.choice(kinds)
            t = rng.choice([None, 0, 1, 2, 3, 4, 7])
            acts[p["id"]] = {"action": k, "target": t}
        out = g.resolve_round(players, acts)
        cases.append({"players": players, "actions": {str(k): v for k, v in acts.items()}, "out": out})
    dump("rules_random.json", cases)
    games = []
    for rec in sel:
        players = [g.make_player(i, f"P{i}", i == 0) for i in range(rec["n"])]
        outs = []
        for rd in rec["rounds"]:
            out = g.resolve_round(players, {int(k): v for k, v in rd.items()})
            outs.append(out)
            players = out["players"]
        games.append({"rec": rec, "outs": outs})
    dump("rules_games.json", games)

    # 4. the solved 2-player mixes for every position and bullet-bonus level
    duels = []
    for b1 in range(g.CAP + 1):
        for b2 in range(g.CAP + 1):
            for r1 in (False, True):
                for r2 in (False, True):
                    for boost in range(4):
                        la, lo, x, y, v, A = g.duel_solution({"bullets": b1, "has_reflect": r1},
                                                             {"bullets": b2, "has_reflect": r2}, boost)
                        duels.append([b1, b2, r1, r2, boost, la, lo, x, y, v])
    dump("duels.json", duels)

    # 5. position scores for the 3+ player look-ahead
    ev = []
    for _ in range(3000):
        n = rng.choice([2, 3, 4, 5])
        players = [g.make_player(i, f"P{i}", i == 0) for i in range(n)]
        for p in players:
            p["bullets"] = rng.randint(0, 12)
            p["has_reflect"] = rng.random() < 0.5
            p["alive"] = rng.random() < 0.8
        me = rng.randrange(n)
        ev.append([players, me, g.eval_state(players, me)])
    dump("eval_state.json", ev)

    # 6. the bot's decisions in the selected games, with saved memory growing game by game
    states = {s: g.new_bot_state(s) for s in range(1, g.MAX_SLOT + 1)}
    dec_games = []
    for gi, rec in enumerate(sel):
        n = rec["n"]
        diff = ["hard", "normal", "easy"][gi % 3]
        samples = None if n <= 3 else max(60, g.ThinkingBot.LEVELS[diff]["samples"] * 3 // 4)
        bots = {i: g.ThinkingBot(i, diff, TestRng(1000 * gi + i), samples=samples, state=states[i])
                for i in range(1, n)}
        players = [g.make_player(i, "You" if i == 0 else f"Bot {i}", i == 0) for i in range(n)]
        decisions = []
        for rnd, rd in enumerate(rec["rounds"], 1):
            acts = {int(k): v for k, v in rd.items()}
            pub = [dict(p) for p in players]
            for pid, b in bots.items():
                if players[pid]["alive"]:
                    ch = b.choose(pub, rnd)
                    decisions.append([rnd, pid, [list(t) for t in b.last_probs], ch["action"], ch["target"]])
            out = g.resolve_round(players, acts)
            for b in bots.values():
                b.observe(pub, out["actions"])
            players = out["players"]
        for s in range(1, n):
            g.apply_record(states[s], rec)
        dec_games.append({"rec": rec, "difficulty": diff, "samples": samples, "seed_base": 1000 * gi,
                          "decisions": decisions})
    dump("bot_decisions.json", {"games": dec_games,
                                "final_states": {str(s): state_dump(st) for s, st in states.items()}})

    # 7. the after-game re-tuning: every grid score + one tuning step from the defaults
    sc = {str(w): tb.score_prediction(g, sel, w) for w in tb.GRID_W}
    ps = {str(r): tb.score_payoff(g, sel, r) for r in tb.GRID_R}
    with tempfile.TemporaryDirectory() as tmp:
        store = g.BotStore(tmp)
        params = tb.update_params(g, store, sel, max_step=1)
    dump("tuning.json", {"score_prediction": sc, "score_payoff": ps,
                         "update": {k: params[k] for k in ("STREAK_PRIOR_WEIGHT", "PATTERN_ROUNDS")},
                         "update_entry": {k: v for k, v in params["history"][-1].items() if k != "time"}})
    print("done")


if __name__ == "__main__":
    main()
