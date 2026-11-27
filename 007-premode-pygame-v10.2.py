#!/usr/bin/env python3
"""
007 - Only one shall stand   (version 10.2)
===========================================
A multi-round, simultaneous-decision game for 2, 3 or 5 players: you against 1, 2 or 4 bots.
Every round everybody secretly picks one action at the same time; the last player alive wins.

Actions (key)
    Reload (R)        +1 bullet; defenseless this round
    Shoot (S)         -1 bullet; kills the target unless it defends, reflects or shoots back at you
    Defense (D)       free; blocks Shoot and Bomb
    Bomb (B)          -(living players - 1) bullets; a Shoot at every other player at once
    Reflect (E)       once per game; survives Shoot / Bomb and kills those attackers
    Machine Gun (M)   -10 bullets; kills through Defense and Reflect
    Two players shooting each other cancel.  MG against MG: both die.  MG against Shoot: both die
    (the game's hidden rule).

How the bots decide (class ThinkingBot)
    Layer 0  Rules engine.  resolve_round() resolves one round for any number of players.  The bots
             use the same function to look ahead, so they can never disagree with the real rules.
    Layer 1  Two players: the game is solved offline.  Value iteration over all 676 positions
             (bullets 0-12 each, Reflect yes / no each); each position's round is a zero-sum matrix
             game, solved by linear programming (solve_matrix_game).  The resulting values are
             embedded at the end of this file (_V2P_B64); duel_solution() turns them into the
             equilibrium mix of actions for the current position.
    Layer 2  Opponent model (class OppMemory): how often each opponent does what, by bullets, by last
             move and by Defense streak (starting guess: a human defends again 50 / 33 / 15 / 5 % of
             the time after 1 / 2 / 3 / 4+ Defenses in a row).  In a duel the bot exploits this model
             but never gives up more than `tol` of the equilibrium value.
    Layer 3  Three or more players: Monte-Carlo look-ahead.  Sample the other players' moves from
             their models, try every own action with the real rules engine, score the positions with
             eval_state() and pick with a softmax.
    Difficulty (ThinkingBot.LEVELS): Easy = equilibrium + 35 % random mistakes, no exploiting;
             Normal = 3 % mistakes, medium exploiting; Hard = no mistakes, strong exploiting.

Learning between games (folder bot_data_v10/, created next to this file)
    game_records.jsonl   every finished game, one per line (also used by REPLAY)
    botN_memory.json     one per bot seat: long-term model of the human + win / loss lessons
                         (apply_record).  Seat 1 plays in 2-, 3- and 5-player games, seat 2 in 3 and 5,
                         seats 3 and 4 only in 5-player games.
    tuned_params.json    two model parameters, re-fitted by 007-train-bot.py after every game
                         (in a background thread: retune_in_background)

File map (search for these section titles)
    LAYER 0 - RULES ENGINE
    LAYER 1 - TWO-PLAYER GAME
    LAYER 2 - OPPONENT MODEL
    LEARNING FROM RESULTS
    LAYER 1+2+3 - THE BOT
    BASELINE BOTS                        only for --tournament / --selftest
    PERMANENT BOT MEMORY + GAME RECORDS
    HEADLESS TOOLS                       self-test, tournament, solver
    UI                                   everything that needs pygame: table, buttons, human turn,
                                         round animation, match loop, tutorial, setup screen, main()

Command line
    python 007-premode-pygame-v10.2.py                   play
    python 007-premode-pygame-v10.2.py --selftest        checks of the rules, the solver and the bots
    python 007-premode-pygame-v10.2.py --tournament 200  bot strength against baseline bots
    python 007-premode-pygame-v10.2.py --solve           re-solve the 2-player table
    python 007-premode-pygame-v10.2.py --autoplay --headless --games 3 --data-dir DIR
                                                         scripted test run (use a DIR that is not your real data)
    Options: --mode 1|2|4, --difficulty easy|normal|hard, --timer 3|5|10, --skip-tutorial,
             --hotseat (Testing mode: every player is human), --data-dir DIR, --seed N

Rule details
    * bullets and the Reflect charge are spent the moment an action is chosen
    * a Reflect user still kills every Shoot / Bomb attacker even if a Machine Gun kills it in the
      same round (all effects are simultaneous)
    * two Machine Guns at each other: both die; if nobody else is left, the game is a draw
"""
import argparse
import base64
import json
import math
import os
import random
import sys
import time
import zlib
from array import array
from collections import defaultdict

try:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
except ImportError:                       # engine / bots / tournaments work without pygame
    pygame = None

# ============================================================================
#  LAYER 0 - RULES ENGINE (pure functions, no pygame)
# ============================================================================
MG_COST = 10


def make_player(pid, name, human=False):
    """Create a new player with 0 bullets, an unused Reflect and alive status."""
    return {"id": pid, "name": name, "bullets": 0, "has_reflect": True,
            "alive": True, "human": human}


def alive_players(players):
    """Return the players who are still in the game."""
    return [p for p in players if p["alive"]]


def bomb_cost(players):
    """Return the bullet cost of a Bomb, which is one bullet per other living player."""
    return max(1, len(alive_players(players)) - 1)


def legal_actions(p, players):
    """Return every action this player may choose right now."""
    if not p["alive"]:
        return []
    acts = ["reload", "defense"]
    b = p["bullets"]
    if b >= 1:
        acts.append("shoot")
    if len(alive_players(players)) >= 2 and b >= bomb_cost(players):
        acts.append("bomb")
    if p["has_reflect"]:
        acts.append("reflect")
    if b >= MG_COST:
        acts.append("mg")
    return acts


def resolve_round(players, actions):
    """Resolve one round in which all players act at the same time.

    `actions` maps each player id to {"action": str, "target": id or None}.  The result is
    {"players": new list, "deaths": [ids], "events": [...], "actions": cleaned actions}.

    Rules:
      * Bullets and the Reflect charge are spent the moment an action is chosen.
      * A Machine Gun kills its target whatever it does; two players firing MG at each other both die.
      * Two players who Shoot / Bomb each other cancel, so both survive that exchange.
      * Defense blocks Shoot and Bomb.  Reflect also survives them and kills every attacker.
      * Any other target of a Shoot or Bomb dies, including a player who reloads.
      * A Bomb is a Shoot at every other living player.
      * Survivors who chose Reload gain 1 bullet.
    """
    new = [dict(p) for p in players]
    by_id = {p["id"]: p for p in new}
    live_ids = [p["id"] for p in new if p["alive"]]
    bcost = max(1, len(live_ids) - 1)

    acts = {}
    for pid in live_ids:
        p = by_id[pid]
        raw = actions.get(pid) or {}
        act, tgt = raw.get("action", "reload"), raw.get("target")
        if act == "shoot" and p["bullets"] < 1:
            act = "reload"
        elif act == "mg" and p["bullets"] < MG_COST:
            act = "reload"
        elif act == "bomb" and (p["bullets"] < bcost or len(live_ids) < 2):
            act = "reload"
        elif act == "reflect" and not p["has_reflect"]:
            act = "defense"
        elif act not in ("reload", "shoot", "defense", "bomb", "reflect", "mg"):
            act = "reload"
        if act in ("shoot", "mg"):
            if tgt not in live_ids or tgt == pid:
                others = [i for i in live_ids if i != pid]
                tgt = others[0] if others else None
                if tgt is None:
                    act = "reload"
        else:
            tgt = None
        acts[pid] = {"action": act, "target": tgt}

    for pid, a in acts.items():                       # spend resources immediately
        p, act = by_id[pid], a["action"]
        if act == "shoot":
            p["bullets"] -= 1
        elif act == "mg":
            p["bullets"] -= MG_COST
        elif act == "bomb":
            p["bullets"] -= bcost
        elif act == "reflect":
            p["has_reflect"] = False

    attacks, mg_hits = set(), []
    for pid, a in acts.items():
        if a["action"] == "shoot":
            attacks.add((pid, a["target"]))
        elif a["action"] == "bomb":
            attacks.update((pid, t) for t in live_ids if t != pid)
        elif a["action"] == "mg":
            mg_hits.append((pid, a["target"]))

    deaths, events = set(), []
    mg_set = set(mg_hits)
    for (a, t) in sorted(mg_hits):
        if (t, a) in mg_set:
            if a < t:
                events.append(("mg_clash", a, t))
            deaths.update((a, t))
        else:
            events.append(("mg", a, t))
            deaths.add(t)

    seen_cancel = set()
    for (a, t) in sorted(attacks):
        ta = acts[t]["action"]
        if ta == "reflect":
            deaths.add(a)
            events.append(("reflected", a, t))
        elif ta == "defense":
            events.append(("blocked", a, t))
        elif (t, a) in attacks:
            key = frozenset((a, t))
            if key not in seen_cancel:
                seen_cancel.add(key)
                events.append(("cancel", a, t))
        else:
            deaths.add(t)
            events.append(("hit", a, t))

    for pid in live_ids:
        p = by_id[pid]
        if pid in deaths:
            p["alive"] = False
        elif acts[pid]["action"] == "reload":
            p["bullets"] += 1
    return {"players": new, "deaths": sorted(deaths), "events": events, "actions": acts}


# ============================================================================
#  LAYER 1 - TWO-PLAYER GAME: MATRIX-GAME SOLVER + VALUE ITERATION
# ============================================================================
def solve_matrix_game(A):
    """Solve a zero-sum matrix game by linear programming and return both optimal mixes and its value.

    A[i][j] is the payoff to the row player, who maximises.  The function uses a pure-Python simplex
    method with Bland's rule and returns (row mix, column mix, value of the game).
    """
    m, n = len(A), len(A[0])
    lo = min(min(r) for r in A)
    shift = 1.0 - lo                                   # make every entry >= 1
    T = [[A[i][j] + shift for j in range(n)] + [1.0 if k == i else 0.0 for k in range(m)] + [1.0]
         for i in range(m)]
    z = [-1.0] * n + [0.0] * m + [0.0]
    basis = [n + i for i in range(m)]
    eps = 1e-11
    for _ in range(1000):
        col = -1
        for c in range(n + m):
            if z[c] < -eps:
                col = c
                break
        if col < 0:
            break
        row, best = -1, None
        for i in range(m):
            a = T[i][col]
            if a > eps:
                r = T[i][-1] / a
                if best is None or r < best - 1e-12 or (abs(r - best) <= 1e-12 and basis[i] < basis[row]):
                    best, row = r, i
        if row < 0:
            break
        pv = T[row][col]
        T[row] = [v / pv for v in T[row]]
        for i in range(m):
            if i != row:
                f = T[i][col]
                if f != 0.0:
                    Ti, Tr = T[i], T[row]
                    T[i] = [Ti[k] - f * Tr[k] for k in range(n + m + 1)]
        f = z[col]
        z = [z[k] - f * T[row][k] for k in range(n + m + 1)]
        basis[row] = col
    S = z[-1]
    y = [0.0] * n
    for i, b in enumerate(basis):
        if b < n:
            y[b] = T[i][-1]
    x = [max(0.0, z[n + i]) for i in range(m)]
    sx, sy = sum(x) or 1.0, sum(y) or 1.0
    return [v / sx for v in x], [v / sy for v in y], 1.0 / S - shift


CAP = 12            # bullets above this are treated as CAP inside the solved table
GAMMA = 0.985       # tiny discount: prefer winning sooner, makes cycles converge
CANON = ["reload", "defense", "shoot", "reflect", "mg"]     # Bomb == Shoot with 2 players

# Bullet bonus, used only during play: for every round the bot has just defended in a row (at most
# STREAK_K_CAP rounds), a round in which nobody dies is worth an extra
#     STREAK_K * (my bullet change - their bullet change)        (zero-sum, so the game stays solvable)
# The longer the bot hides without hurting anybody, the more it needs ammunition.  The solved table
# itself has no bullet bonus: tests with 400-game matches showed that any bonus there makes the bot
# waste its Reflect early, so the table stays the true-game equilibrium.
STREAK_K = 0.04
STREAK_K_CAP = 3
V2P_SCALE = 2.0          # packed-table scale


def legal2(b, r):
    """Return the actions of one side in a 2-player position.

    Bomb is left out, because with two players it is the same as Shoot.
    """
    acts = ["reload", "defense"]
    if b >= 1:
        acts.append("shoot")
    if r:
        acts.append("reflect")
    if b >= MG_COST:
        acts.append("mg")
    return acts


def s2idx(b1, b2, r1, r2):
    """Convert a 2-player position into its index in the solved value table.

    A position is (my bullets, your bullets, my Reflect, your Reflect).  Bullets above CAP count as CAP.
    """
    return ((min(b1, CAP) * (CAP + 1) + min(b2, CAP)) * 2 + int(bool(r1))) * 2 + int(bool(r2))


def trans2(b1, b2, r1, r2, a1, a2):
    """Play one 2-player round with the real rules and report where it leads.

    It returns ("T", result, 0) when the round ends the game, with +1, -1 or 0 for side 1.  Otherwise
    it returns ("S", index of the next position, change in the bullet difference).
    """
    ps = [make_player(0, "a"), make_player(1, "b")]
    ps[0]["bullets"], ps[1]["bullets"] = b1, b2
    ps[0]["has_reflect"], ps[1]["has_reflect"] = bool(r1), bool(r2)
    out = resolve_round(ps, {0: {"action": a1, "target": 1}, 1: {"action": a2, "target": 0}})
    q = out["players"]
    if not q[0]["alive"] and not q[1]["alive"]:
        return ("T", 0.0, 0.0)
    if not q[0]["alive"]:
        return ("T", -1.0, 0.0)
    if not q[1]["alive"]:
        return ("T", 1.0, 0.0)
    shape = (q[0]["bullets"] - b1) - (q[1]["bullets"] - b2)       # bullet swing this round
    return ("S", s2idx(q[0]["bullets"], q[1]["bullets"], q[0]["has_reflect"], q[1]["has_reflect"]), shape)


def compute_value_table(max_iter=200, tol=2e-4, verbose=False):
    """Solve the whole 2-player game by value iteration.

    V[s] is the value of position s for player 1, from +1 (certain win) to -1 (certain loss).  All values
    start at 0.  For every position the function builds the payoff table of my action against your
    action: +1, -1 or 0 when the round ends the game, otherwise GAMMA times the value of the next
    position.  V[s] becomes the value of that matrix game.  This repeats until no value changes by
    more than `tol`.
    """
    states = [(b1, b2, r1, r2) for b1 in range(CAP + 1) for b2 in range(CAP + 1)
              for r1 in (0, 1) for r2 in (0, 1)]
    plan = []
    for (b1, b2, r1, r2) in states:
        la, lo = legal2(b1, r1), legal2(b2, r2)
        grid = [[trans2(b1, b2, r1, r2, a, o) for o in lo] for a in la]
        plan.append((s2idx(b1, b2, r1, r2), grid))
    V = [0.0] * len(states)
    for it in range(max_iter):
        delta = 0.0
        for idx, grid in plan:
            A = [[(t[1] if t[0] == "T" else GAMMA * V[t[1]]) for t in row] for row in grid]
            v = solve_matrix_game(A)[2]
            delta = max(delta, abs(v - V[idx]))
            V[idx] = v
        if verbose:
            print(f"  iteration {it + 1}: max change {delta:.6f}", flush=True)
        if delta < tol:
            break
    return V


def pack_table(V):
    """Compress the value table into the text that is embedded in this file."""
    arr = array("h", [int(round(max(-V2P_SCALE, min(V2P_SCALE, v)) / V2P_SCALE * 30000)) for v in V])
    return base64.b64encode(zlib.compress(arr.tobytes(), 9)).decode("ascii")


def unpack_table(b64):
    """Turn the embedded text back into the value table."""
    arr = array("h")
    arr.frombytes(zlib.decompress(base64.b64decode(b64)))
    return [v / 30000.0 * V2P_SCALE for v in arr]


_V2P = None
_DUEL_CACHE = {}


def get_v2p():
    """Return the solved 2-player value table, loading it on first use.

    The table is normally unpacked from _V2P_B64.  Only if that text is missing is the game solved
    from scratch, which takes about a minute.
    """
    global _V2P
    if _V2P is None:
        if _V2P_B64:
            _V2P = unpack_table(_V2P_B64)
        else:
            print("[007] no embedded table - solving the 2-player game (one-off, ~1 min)...", flush=True)
            _V2P = compute_value_table(verbose=False)
    return _V2P


def duel_solution(p1, p2, boost=0):
    """Compute the equilibrium mix of actions for the current 2-player position.

    p1 is the row player.  `boost` is the number of rounds p1 has just defended in a row; each one adds
    STREAK_K to the value of gaining bullets, because the longer a player hides, the more it needs
    ammunition.  The result is (p1's actions, p2's actions, p1's mix, p2's mix, value, payoff table).
    """
    b1, b2 = p1["bullets"], p2["bullets"]
    r1, r2 = bool(p1["has_reflect"]), bool(p2["has_reflect"])
    boost = max(0, min(int(boost), STREAK_K_CAP))
    key = (b1, b2, r1, r2, boost)
    hit = _DUEL_CACHE.get(key)
    if hit:
        return hit
    V = get_v2p()
    k_eff = STREAK_K * boost
    la, lo = legal2(b1, r1), legal2(b2, r2)
    A = []
    for a in la:
        row = []
        for o in lo:
            t = trans2(b1, b2, r1, r2, a, o)
            row.append(t[1] if t[0] == "T" else GAMMA * V[t[1]] + k_eff * t[2])
        A.append(row)
    x, y, v = solve_matrix_game(A)
    if len(_DUEL_CACHE) > 20000:
        _DUEL_CACHE.clear()
    sol = (la, lo, x, y, v, A)
    _DUEL_CACHE[key] = sol
    return sol


def duel_value(me, opp):
    """Return how good a 2-player position is for `me` according to the solved table."""
    return get_v2p()[s2idx(me["bullets"], opp["bullets"], me["has_reflect"], opp["has_reflect"])]


# ============================================================================
#  LAYER 2 - OPPONENT MODEL
# ============================================================================
def bucket(b):
    """Group a bullet count into one of 6 bands so that habits can be counted with little data.

    The bands are 0, 1-2, 3-5, 6-8, 9 and 10+.
    """
    return 0 if b == 0 else 1 if b <= 2 else 2 if b <= 5 else 3 if b <= 8 else 4 if b == 9 else 5


STREAK_PRIOR = {1: 0.50, 2: 0.33, 3: 0.15, 4: 0.05}   # P(opponent defends AGAIN | it has defended d rounds in a row; 4 = "4 or more")
STREAK_PRIOR_WEIGHT = 6.0       # the prior is worth this many observations; real data takes over as it accumulates
_UNSET = object()


def streak_bucket(d):
    """Group a Defense streak length into 1, 2, 3 or '4 or more'."""
    return min(int(d), 4)


class OppMemory:
    """Model one opponent's habits from the moves it has made.

    It counts how often the opponent chose each action, by bullet band, by (band, last move) and by
    Defense streak.  The live part fades a little every round, so it follows the current game.  The
    base part holds earlier games; it is attached when a game starts and does not change during it.
    """

    def __init__(self, decay=0.93):
        """Create an empty model of one opponent.

        `decay` is how much every older observation still counts after each new one: 0.93 inside a game,
        1.0 for the saved long-term model, which fades once per game instead.  The tables are c0 (all
        moves), c1 (by bullet band), c2 (by band and last move), cs (by Defense streak) and tgt (who this
        opponent shoots at).
        """
        self.DECAY = decay
        self.c0 = defaultdict(float)
        self.c1 = defaultdict(lambda: defaultdict(float))
        self.c2 = defaultdict(lambda: defaultdict(float))
        self.cs = defaultdict(lambda: defaultdict(float))
        self.tgt = defaultdict(float)
        self.last = None
        self.streak = 0
        self.base = None
        self.base_w = 0.0

    def reset_game(self):
        """Start a new game by forgetting the last move and the current streak, keeping the counts."""
        self.last = None
        self.streak = 0

    def attach_base(self, base):
        """Attach the frozen counts from earlier games as the base of this model.

        Every stored game counts in full, with no cap, so the model improves with every game played.
        """
        self.base = base
        self.base_w = 1.0 if sum(base.c0.values()) > 0 else 0.0

    def update(self, bullets, action, target):
        """Count one observed move and update the Defense streak.

        All earlier counts fade by DECAY before the new move is added.
        """
        bk = bucket(bullets)
        sb = streak_bucket(self.streak)
        for k in self.c0:
            self.c0[k] *= self.DECAY
        self.c0[action] += 1.0
        for table, key in ((self.c1, bk), (self.c2, (bk, self.last)), (self.cs, sb)):
            row = table[key]
            for k in row:
                row[k] *= self.DECAY
            row[action] += 1.0
        if action in ("shoot", "mg") and target is not None:
            for k in list(self.tgt):
                self.tgt[k] *= self.DECAY
            self.tgt[target] += 1.0
        self.streak = self.streak + 1 if action == "defense" else 0
        self.last = action

    def _comb(self, name, key=None):
        """Add the live counts and the frozen base counts of one table row."""
        out = defaultdict(float)
        for src, w in ((self, 1.0), (self.base, self.base_w)):
            if src is None or w <= 0:
                continue
            t = getattr(src, name)
            row = t if key is None else t.get(key)
            if row:
                for k, v in row.items():
                    out[k] += w * v
        return out

    def dist(self, bullets, names, merge_bomb, last=_UNSET):
        """Estimate how likely each action is for this opponent in its current situation.

        Only the actions in `names` are considered.  The estimate starts from all moves, then is refined by
        the bullet band and by the last move, each step weighted by how much data it has.  The result is
        (probabilities or None, number of observations).
        """
        last = self.last if last is _UNSET else last

        def merged(row):
            """Count Bomb as Shoot when the caller asks for that."""
            out = defaultdict(float)
            for k, v in row.items():
                out["shoot" if (merge_bomb and k == "bomb") else k] += v
            return out
        r0 = merged(self._comb("c0"))
        n_all = sum(r0.values())
        if n_all <= 0:
            return None, 0.0
        s0 = sum(r0.get(a, 0.0) for a in names)
        if s0 <= 0:
            return None, 0.0
        p = {a: r0.get(a, 0.0) / s0 for a in names}
        bk = bucket(bullets)
        for row in (self._comb("c1", bk), self._comb("c2", (bk, last))):
            if row:
                r = merged(row)
                n = sum(r.get(a, 0.0) for a in names)
                p = {a: (r.get(a, 0.0) + 2.0 * p[a]) / (n + 2.0) for a in names}
        return p, n_all

    def streak_def_prob(self, d=None):
        """Estimate how likely this opponent is to defend again after its current Defense streak.

        It starts from the prior 50 / 33 / 15 / 5 % for streaks of 1 / 2 / 3 / 4+ and moves toward what has
        really been observed as data comes in.  The result is None when there is no streak.
        """
        d = self.streak if d is None else d
        if d <= 0:
            return None, 0.0
        sb = streak_bucket(d)
        row = self._comb("cs", sb)
        n = sum(row.values())
        pr = STREAK_PRIOR[sb]
        return (row.get("defense", 0.0) + STREAK_PRIOR_WEIGHT * pr) / (n + STREAK_PRIOR_WEIGHT), n

    def target_shares(self):
        """Return how often this opponent has shot at each player."""
        tot = sum(self.tgt.values())
        if tot < 2.0:
            return None, tot
        return {k: v / tot for k, v in self.tgt.items()}, tot

    def scale_inplace(self, f):
        """Multiply every count by f so that older games count less."""
        for row in [self.c0] + list(self.c1.values()) + list(self.c2.values()) + list(self.cs.values()) + [self.tgt]:
            for k in row:
                row[k] *= f

    def to_json(self):
        """Convert the counts to JSON for the memory files."""
        r4 = lambda row: {k: round(v, 4) for k, v in row.items()}
        return {"decay": self.DECAY, "c0": r4(self.c0),
                "c1": {str(k): r4(v) for k, v in self.c1.items()},
                "c2": {f"{k[0]}|{k[1]}": r4(v) for k, v in self.c2.items()},
                "cs": {str(k): r4(v) for k, v in self.cs.items()},
                "tgt": {str(k): round(v, 4) for k, v in self.tgt.items()}}

    @classmethod
    def from_json(cls, d, decay=None):
        """Rebuild a model from its JSON form."""
        m = cls(d.get("decay", 0.93) if decay is None else decay)
        m.c0.update(d.get("c0", {}))
        for k, row in d.get("c1", {}).items():
            m.c1[int(k)].update(row)
        for k, row in d.get("c2", {}).items():
            bk, last = k.split("|", 1)
            m.c2[(int(bk), None if last == "None" else last)].update(row)
        for k, row in d.get("cs", {}).items():
            m.cs[int(k)].update(row)
        for k, v in d.get("tgt", {}).items():
            m.tgt[int(k)] = v
        return m


# What a human does when it is NOT defending: mostly it reloads ("waits, then reloads").
HUMAN_REST = {"reload": 1.0, "shoot": 0.25, "bomb": 0.10, "reflect": 0.10}


def human_rest(names, emp=None, n=0.0):
    """Split the probability of the non-Defense moves the way a human usually plays them.

    It starts from a typical human split (mostly Reload) and blends in what this player has really done.
    """
    base = {a: HUMAN_REST.get(a, 0.0) for a in names if a != "defense"}
    if "mg" in names:                                   # 10+ bullets: it will fire
        base["mg"] = 3.0
        base["reload"] = 0.2
    if emp:
        obs = {a: emp.get(a, 0.0) for a in base}
        so, sb = sum(obs.values()), sum(base.values())
        if so > 1e-9 and sb > 1e-9:
            w = n / (n + 8.0)
            base = {a: (1 - w) * base[a] / sb + w * obs[a] / so for a in base}
    return base


def apply_streak(p, d, mem, rest_hint=None):
    """Adjust an action distribution so that its Defense share follows the Defense-streak model.

    `p` maps actions to probabilities.  The Defense share is set by the streak model and the other
    actions share the rest, in the proportions of `rest_hint` if given.  The result is (new
    distribution, whether it was applied).
    """
    if d <= 0 or "defense" not in p:
        return p, False
    pd, _ = mem.streak_def_prob(d)
    rest = {a: w for a, w in (rest_hint or p).items() if a != "defense" and a in p}
    if not rest:
        return p, False
    s = sum(rest.values())
    if s <= 1e-12:
        rest = {a: 1.0 for a in rest}
        if "reload" in rest:
            rest = {"reload": 1.0}
        s = sum(rest.values())
    out = {a: (1.0 - pd) * w / s for a, w in rest.items()}
    out["defense"] = pd
    return out, True


# ============================================================================
#  LEARNING FROM RESULTS (win / loss feedback per situation)
# ============================================================================
LEARN_BETA = 0.35         # a learned factor never moves an action's probability by more than +-35 %
LEARN_K = 6.0             # shrinkage: a situation needs real evidence before it counts
LEARN_RECENCY = 0.85      # the last decisions of a game get the blame / credit (weight 0.85 ** rounds-from-the-end)
LEARN_GAME_DECAY = 0.98   # old games fade slowly
HUMAN_GAME_DECAY = 0.97


def situation_key(my_bullets, max_opp_bullets, human_streak):
    """Build the situation label that the win / loss lessons are stored under.

    The label is: my bullet band | the richest opponent's bullet band | the human's Defense streak (0-3+).
    """
    return f"{bucket(my_bullets)}|{bucket(max_opp_bullets)}|{min(int(human_streak), 3)}"


class DecisionStats:
    """Store how well each action worked out in each situation in earlier games.

    For each situation and action it keeps how much better or worse than its fair share the bot
    finished after choosing that action.
    """

    def __init__(self):
        """Create empty statistics."""
        self.t = {}           # key -> action -> [sum of advantage x weight, sum of weight]

    def add(self, key, action, adv, w):
        """Record the outcome of one decision."""
        cell = self.t.setdefault(key, {}).setdefault(action, [0.0, 0.0])
        cell[0] += adv * w
        cell[1] += w

    def factor(self, key, action):
        """Return the multiplier that the lessons give this action in this situation.

        The factor lies within 1 +- LEARN_BETA and stays close to 1 while there is little evidence (LEARN_K).
        """
        cell = self.t.get(key, {}).get(action)
        if not cell:
            return 1.0
        m = max(-1.0, min(1.0, cell[0] / (cell[1] + LEARN_K)))
        return 1.0 + LEARN_BETA * m

    def decay(self, f):
        """Make older games count less."""
        for acts in self.t.values():
            for cell in acts.values():
                cell[0] *= f
                cell[1] *= f

    def to_json(self):
        """Convert the statistics to JSON."""
        return {k: {a: [round(c[0], 4), round(c[1], 4)] for a, c in acts.items()} for k, acts in self.t.items()}

    @classmethod
    def from_json(cls, d):
        """Rebuild the statistics from their JSON form."""
        s = cls()
        s.t = {k: {a: list(c) for a, c in acts.items()} for k, acts in d.items()}
        return s


# ============================================================================
#  LAYER 1+2+3 - THE BOT
# ============================================================================
def softmax(vals, tau):
    """Turn scores into probabilities, sharper for a small tau."""
    m = max(vals)
    e = [math.exp((v - m) / tau) for v in vals]
    s = sum(e)
    return [v / s for v in e]


def sample_index(probs, rng):
    """Draw a random index with the given probabilities."""
    r, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def eval_state(players, me_id):
    """Score a position for `me` between -1 (lost) and +1 (won).

    With two players left the value comes exactly from the solved table; with more players it is a
    heuristic based on bullets, Reflect and the number of players.
    """
    me = next(p for p in players if p["id"] == me_id)
    live = [p for p in players if p["alive"]]
    if not live:
        return 0.0
    if not me["alive"]:
        return -1.0
    others = [p for p in live if p["id"] != me_id]
    if not others:
        return 1.0
    if len(others) == 1:
        return GAMMA * duel_value(me, others[0])
    n = len(live)
    v = 0.6 * (2.0 / n - 1.0)
    v += 0.04 * min(me["bullets"], 10) + (0.15 if me["bullets"] >= MG_COST else 0.0)
    if me["has_reflect"]:
        v += 0.08
    for o in others:
        v -= 0.02 * min(o["bullets"], 10)
        if o["has_reflect"]:
            v -= 0.02
        if o["bullets"] >= MG_COST:
            v -= 0.5 / (n - 1)
        elif o["bullets"] == MG_COST - 1:
            v -= 0.2 / (n - 1)
    return max(-0.9, min(0.9, v))


class ThinkingBot:
    """Play as the game's bot, using the solved game, opponent models and look-ahead.

    One bot is created per seat and match.  choose() picks the action for a round and observe() sees
    what everybody did afterwards.  The layers are explained in the file header.
    """
    LEVELS = {
        "easy":   dict(tau=0.45, mistake=0.35, exploit=0.0, tol=0.0,  samples=60),
        "normal": dict(tau=0.12, mistake=0.03, exploit=0.5,  tol=0.12, samples=100),
        "hard":   dict(tau=0.05, mistake=0.0,  exploit=0.85, tol=0.30, samples=160),
    }
    PATTERN_ROUNDS = 8.0    # the 50/33/15/5 streak prior counts as this many observations when exploiting

    def __init__(self, pid, difficulty="normal", rng=None, samples=None, state=None):
        """Create a bot for one seat with the settings of its difficulty level.

        `state` is this seat's saved memory (see BotStore.load); None means the bot starts with no memory.
        """
        self.pid = pid
        self.cfg = dict(self.LEVELS[difficulty])
        if samples:
            self.cfg["samples"] = samples
        self.rng = rng or random.Random()
        self.mem = {}
        self.last_probs = []
        self.my_streak = 0          # how many rounds in a row THIS bot has defended
        self.state = state
        if state is not None:       # earlier games = a frozen base; the new game is analysed on top of it
            self._mem(0).attach_base(state["human"])

    # ---- memory -----------------------------------------------------------
    def _mem(self, pid):
        """Return this bot's in-game model of one opponent, creating it when needed."""
        if pid not in self.mem:
            self.mem[pid] = OppMemory()
        return self.mem[pid]

    def observe(self, players_before, actions):
        """Update the opponent models and the bot's own Defense streak after a round."""
        for p in players_before:
            if p["alive"] and p["id"] != self.pid and p["id"] in actions:
                a = actions[p["id"]]
                self._mem(p["id"]).update(p["bullets"], a["action"], a.get("target"))
        mine = actions.get(self.pid)
        if mine:
            self.my_streak = self.my_streak + 1 if mine["action"] == "defense" else 0

    def _learn_factors(self, me, opps, names):
        """Return the multiplier for each action that the lessons from earlier games give."""
        if self.state is None:
            return {a: 1.0 for a in names}
        hm = self.mem.get(0)
        hs = hm.streak if (hm is not None and any(o["id"] == 0 for o in opps)) else 0
        key = situation_key(me["bullets"], max(o["bullets"] for o in opps), hs)
        return {a: self.state["learn"].factor(key, a) for a in names}

    # ---- public API ---------------------------------------------------------
    def choose(self, players, round_no=0):
        """Choose this bot's action for the round.

        With one opponent left it uses the solved 2-player strategy, otherwise the Monte-Carlo look-ahead.
        Defense and Reflect are not allowed when no opponent has a bullet, because they cannot help.
        """
        me = next(p for p in players if p["id"] == self.pid)
        opps = [p for p in players if p["alive"] and p["id"] != self.pid]
        if not opps:
            return {"action": "reload", "target": None}
        # Secured rule: opponents with 0 bullets cannot attack, so Defense and
        # Reflect are pointless (and Reflect would be wasted). Ban them; the
        # bot still chooses among Reload / Shoot / Bomb / MG as usual.
        ban = ("defense", "reflect") if all(o["bullets"] == 0 for o in opps) else ()
        if len(opps) == 1:
            return self._duel(me, opps[0], ban)
        return self._melee(me, opps, players, ban)

    # ---- 2 players left: solved game + safe exploitation --------------------
    def _duel(self, me, opp, ban=()):
        """Choose an action when only one opponent is left.

        Steps: 1) take the equilibrium mix of the solved position; 2) predict the opponent from its model,
        plus the Defense-streak pattern for the human; 3) lean toward the best answer to that prediction,
        but only as far as the worst case stays within `tol` of the equilibrium value; 4) apply the
        lessons from earlier games; 5) mix in the difficulty's random mistakes.
        """
        la, lo, x, y, v, A = duel_solution(me, opp, self.my_streak)
        mem = self._mem(opp["id"])
        emp, n = mem.dist(opp["bullets"], lo, merge_bomb=True)
        q = list(y)
        if emp:
            lam_m = n / (n + 3.0)
            q = [(1 - lam_m) * y[j] + lam_m * emp[lo[j]] for j in range(len(lo))]
            s = sum(q)
            q = [w / s for w in q]
        pattern = False
        if opp.get("human"):
            # "After d rounds of Defense a human defends again with 50 / 33 / 15 / 5 %":
            # weight every outcome by how likely the human's next move really is.
            hint = human_rest(lo, emp, n)
            qd, pattern = apply_streak(dict(zip(lo, q)), mem.streak, mem, hint)
            q = [qd[a] for a in lo]
        n_eff = n + (self.PATTERN_ROUNDS if pattern else 0.0)
        ev = [sum(q[j] * A[i][j] for j in range(len(lo))) for i in range(len(la))]
        soft = softmax(ev, 0.03)
        lam = self.cfg["exploit"] * (n_eff / (n_eff + 4.0)) if (emp or pattern) else 0.0
        mix = list(x)
        for _ in range(8):
            mix = [(1 - lam) * x[i] + lam * soft[i] for i in range(len(la))]
            worst = min(sum(mix[i] * A[i][j] for i in range(len(la))) for j in range(len(lo)))
            if worst >= v - self.cfg["tol"]:
                break
            lam *= 0.6
        fac = self._learn_factors(me, [opp], la)          # lessons from won / lost games (bounded +-35 %)
        mix = [m * fac[la[i]] for i, m in enumerate(mix)]
        s = sum(mix)
        mix = [m / s for m in mix]
        mk = self.cfg["mistake"]
        mix = [(1 - mk) * m + mk / len(la) for m in mix]
        if ban:
            mix = [0.0 if la[i] in ban else m for i, m in enumerate(mix)]
            if sum(mix) <= 1e-9:
                mix = [0.0 if la[i] in ban else 1.0 for i in range(len(la))]
        s = sum(mix)
        mix = [m / s for m in mix]
        self.last_probs = sorted(zip(la, mix), key=lambda t: -t[1])
        a = la[sample_index(mix, self.rng)]
        return {"action": a, "target": opp["id"] if a in ("shoot", "mg") else None}

    # ---- 3+ players: Monte-Carlo one-ply look-ahead with the real engine -----
    def _opp_profile(self, o, alive, viewer):
        """Predict one opponent's next move for the look-ahead.

        It starts from that opponent's own equilibrium against its most dangerous rival, adjusts it for
        the number of armed players, and blends in what has been observed, plus the Defense-streak pattern
        for the human.  The result is a list of (action, target, probability).
        """
        rivals = [p for p in alive if p["id"] != o["id"]]
        rival = max(rivals, key=lambda p: (p["bullets"], p["id"] == viewer["id"]))
        la, lo, x, y, v, A = duel_solution(o, rival)
        prior = {a: x[i] for i, a in enumerate(la)}
        crowd = max(0, sum(1 for p in rivals if p["bullets"] >= 1) - 1)
        prior["reload"] /= (1.0 + 0.6 * crowd)
        prior["defense"] *= (1.0 + 0.3 * crowd)
        n = len(alive)
        if o["bullets"] >= n - 1 and "shoot" in prior:
            mv = prior["shoot"] * 0.25
            prior["shoot"] -= mv
            prior["bomb"] = mv
        s = sum(prior.values())
        prior = {a: w / s for a, w in prior.items()}
        mem = self.mem.get(o["id"])
        if mem:
            emp, nn = mem.dist(o["bullets"], list(prior), merge_bomb=False)
            if emp:
                lam = nn / (nn + (6.0 if o["human"] else 10.0))
                prior = {a: (1 - lam) * prior[a] + lam * emp[a] for a in prior}
        if mem and o.get("human"):
            names = list(prior)
            e2, n2 = mem.dist(o["bullets"], names, merge_bomb=False)
            prior, _ = apply_streak(prior, mem.streak, mem, human_rest(names, e2, n2))   # defense streak 50/33/15/5 %
        tw = {p["id"]: 1.0 + 0.5 * p["bullets"] + (1.5 if p["bullets"] >= 8 else 0.0) for p in rivals}
        if mem:
            shares, cnt = mem.target_shares()
            if shares:
                mu = cnt / (cnt + 6.0)
                tot = sum(tw.values())
                tw = {t: (1 - mu) * w / tot + mu * shares.get(t, 0.0) for t, w in tw.items()}
        tsum = sum(tw.values())
        out = []
        for a, pa in prior.items():
            if pa <= 0:
                continue
            if a in ("shoot", "mg"):
                for t, w in tw.items():
                    out.append((a, t, pa * w / tsum))
            else:
                out.append((a, None, pa))
        return out

    def _melee(self, me, opps, players, ban=()):
        """Choose an action when two or more opponents are left.

        It draws `samples` guesses of everybody else's moves, plays every own action against each guess
        with the real rules engine, and averages eval_state() of the results.  The final choice uses
        softmax(tau), the lessons from earlier games and the difficulty's mistake rate.
        """
        alive = [p for p in players if p["alive"]]
        rng, cfg = self.rng, self.cfg
        profiles_dist = {o["id"]: self._opp_profile(o, alive, me) for o in opps}
        cum = {}
        for oid, dist in profiles_dist.items():
            c, acc = [], 0.0
            for (a, t, p) in dist:
                acc += p
                c.append(acc)
            cum[oid] = (dist, c, acc)
        profiles = []
        for _ in range(cfg["samples"]):
            prof = {}
            for oid, (dist, c, tot) in cum.items():
                r = rng.random() * tot
                k = 0
                while k < len(c) - 1 and r > c[k]:
                    k += 1
                prof[oid] = {"action": dist[k][0], "target": dist[k][1]}
            profiles.append(prof)
        cands = []
        for a in legal_actions(me, players):
            if a in ban:
                continue
            if a in ("shoot", "mg"):
                cands += [(a, o["id"]) for o in opps]
            else:
                cands.append((a, None))
        ev = []
        for (a, t) in cands:
            tot = 0.0
            for prof in profiles:
                acts = dict(prof)
                acts[me["id"]] = {"action": a, "target": t}
                tot += eval_state(resolve_round(players, acts)["players"], me["id"])
            ev.append(tot / len(profiles))
        probs = softmax(ev, cfg["tau"])
        fac = self._learn_factors(me, opps, {a for a, _ in cands})      # lessons from won / lost games
        probs = [p * fac[a] for (a, _), p in zip(cands, probs)]
        s = sum(probs)
        probs = [p / s for p in probs]
        mk = cfg["mistake"]
        probs = [(1 - mk) * p + mk / len(cands) for p in probs]
        agg = defaultdict(float)
        for (a, t), p in zip(cands, probs):
            agg[a] += p
        self.last_probs = sorted(agg.items(), key=lambda t: -t[1])
        a, t = cands[sample_index(probs, rng)]
        return {"action": a, "target": t}


# ============================================================================
#  BASELINE BOTS (for the tournament / self-test only)
# ============================================================================
class RandomBot:
    """Play a random legal action every round, as a baseline for testing."""
    def __init__(self, pid, rng=None):
        """Create the baseline bot."""
        self.pid, self.rng = pid, rng or random.Random()

    def observe(self, *_):
        """Ignore what happened, because this bot has no memory."""
        pass

    def choose(self, players, round_no=0):
        """Choose a random legal action and a random target."""
        me = next(p for p in players if p["id"] == self.pid)
        a = self.rng.choice(legal_actions(me, players))
        tgt = None
        if a in ("shoot", "mg"):
            tgt = self.rng.choice([p["id"] for p in players if p["alive"] and p["id"] != self.pid])
        return {"action": a, "target": tgt}


class FixedBot:
    """Play the same action every round, as a baseline for testing."""
    def __init__(self, pid, action):
        """Create the baseline bot with its fixed action."""
        self.pid, self.action = pid, action

    def observe(self, *_):
        """Ignore what happened, because this bot has no memory."""
        pass

    def choose(self, players, round_no=0):
        """Return the fixed action."""
        return {"action": self.action, "target": None}


class HeuristicBot:
    """Play simple common-sense rules, as a stand-in for a sensible human in tests.

    It reloads when that is safe, shoots when ahead, and fires the Machine Gun at 10 bullets.
    """
    def __init__(self, pid, rng=None):
        """Create the rule-based bot."""
        self.pid, self.rng = pid, rng or random.Random()

    def observe(self, *_):
        """Ignore what happened, because this bot has no memory."""
        pass

    def choose(self, players, round_no=0):
        """Choose an action by the simple rules."""
        me = next(p for p in players if p["id"] == self.pid)
        opps = [p for p in players if p["alive"] and p["id"] != self.pid]
        top = max(opps, key=lambda p: p["bullets"])
        if me["bullets"] >= MG_COST:
            return {"action": "mg", "target": top["id"]}
        r = self.rng.random()
        if top["bullets"] == 0:
            return {"action": "shoot" if me["bullets"] and r < 0.6 else "reload",
                    "target": top["id"] if me["bullets"] and r < 0.6 else None}
        if r < 0.35:
            return {"action": "reload", "target": None}
        if r < 0.65 and me["bullets"] >= 1:
            return {"action": "shoot", "target": top["id"]}
        if r < 0.72 and me["has_reflect"]:
            return {"action": "reflect", "target": None}
        return {"action": "defense", "target": None}


# ============================================================================
#  PERMANENT BOT MEMORY + GAME RECORDS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "bot_data_v10")
# Which player counts each bot slot takes part in (and therefore learns from):
#   Bot 1 sits in 2-, 3- and 5-player games, Bot 2 in 3- and 5-, Bots 3 and 4 only in 5-player games.
SLOT_MODES = {1: (2, 3, 5), 2: (3, 5), 3: (5,), 4: (5,)}
MAX_SLOT = 4


TUNABLE_DEFAULTS = {"STREAK_PRIOR_WEIGHT": 6.0, "PATTERN_ROUNDS": 8.0}


def apply_params(params):
    """Put the tuned variables into force."""
    global STREAK_PRIOR_WEIGHT
    STREAK_PRIOR_WEIGHT = float(params["STREAK_PRIOR_WEIGHT"])
    ThinkingBot.PATTERN_ROUNDS = float(params["PATTERN_ROUNDS"])


def new_bot_state(slot):
    """Create the empty memory of one bot seat."""
    return {"slot": slot, "modes": list(SLOT_MODES[slot]), "games": 0, "wins": 0, "losses": 0, "draws": 0,
            "human": OppMemory(decay=1.0), "learn": DecisionStats()}


def bot_state_to_json(st):
    """Convert a bot seat's memory to JSON."""
    return {"version": 1, "slot": st["slot"], "modes_players": st["modes"],
            "games": st["games"], "bot_wins": st["wins"], "bot_losses": st["losses"], "draws": st["draws"],
            "human_model": st["human"].to_json(), "decision_stats": st["learn"].to_json()}


def bot_state_from_json(d, slot):
    """Rebuild a bot seat's memory from its JSON form."""
    st = new_bot_state(slot)
    st["games"], st["wins"] = d.get("games", 0), d.get("bot_wins", 0)
    st["losses"], st["draws"] = d.get("bot_losses", 0), d.get("draws", 0)
    st["human"] = OppMemory.from_json(d.get("human_model", {}), decay=1.0)
    st["learn"] = DecisionStats.from_json(d.get("decision_stats", {}))
    return st


def apply_record(st, rec, weight=1.0, sim=False):
    """Let one bot seat learn from one finished game.

    The game is rebuilt from its recorded actions with the real rules engine.  The same function is
    used after every game and by 007-train-bot.py, so both give identical results.
      * The seat's model of the human counts what the human did.
      * After a loss (win) the bot's late decisions become less (more) likely in the same situation,
        measured against the fair share 1/n.
    With sim=True (a practice game against a simulated human) only the lessons are learned, usually
    with a weight below 1.  It returns False when this seat did not take part in the game.
    """
    slot, n = st["slot"], rec["n"]
    if slot >= n:
        return False
    hm, ls = st["human"], st["learn"]
    if not sim:
        hm.scale_inplace(HUMAN_GAME_DECAY)
        ls.decay(LEARN_GAME_DECAY)
        hm.reset_game()
    players = [make_player(i, f"P{i}") for i in range(n)]
    hs = 0                                   # the human's defense streak (for the situation key)
    decisions = []
    for rd in rec["rounds"]:
        acts = {int(k): v for k, v in rd.items()}
        me = players[slot]
        if me["alive"] and slot in acts:
            mo = max((p["bullets"] for p in players if p["alive"] and p["id"] != slot), default=0)
            decisions.append((situation_key(me["bullets"], mo, hs), acts[slot]["action"]))
        h = players[0]
        if h["alive"] and 0 in acts:
            if not sim:
                hm.update(h["bullets"], acts[0]["action"], acts[0].get("target"))
            hs = hs + 1 if acts[0]["action"] == "defense" else 0
        else:
            hs = 0
        players = resolve_round(players, acts)["players"]
    winner = rec.get("winner")
    res = 1.0 if winner == slot else (0.0 if winner is None else -1.0)
    adv = res - (2.0 / n - 1.0)
    for j, (key, action) in enumerate(reversed(decisions)):
        w = weight * (LEARN_RECENCY ** j)
        ls.add(key, action, adv * w, w)
    if not sim:
        st["games"] += 1
        st["wins"] += res > 0
        st["losses"] += res < 0
        st["draws"] += res == 0
    return True


def make_record(rec_id, cfg, n, history, winner):
    """Build the record of a finished game that is saved and can be replayed."""
    return {"v": 1, "id": rec_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n": n, "mode": n - 1,
            "difficulty": cfg.get("difficulty", "normal"), "timer": cfg.get("timer", 10), "winner": winner,
            "rounds": [{str(pid): {"action": a["action"], "target": a.get("target")} for pid, a in r.items()}
                       for r in history]}


class BotStore:
    """Read and write the bots' memory files and the game records.

    The folder holds bot1_memory.json ... bot4_memory.json, game_records.jsonl (one game per line)
    and tuned_params.json.
    """

    def __init__(self, data_dir=None):
        """Use the given data folder, or bot_data_v10 next to this file."""
        self.dir = data_dir or DEFAULT_DATA_DIR
        self.records_path = os.path.join(self.dir, "game_records.jsonl")

    def memory_path(self, slot):
        """Return the path of one seat's memory file."""
        return os.path.join(self.dir, f"bot{slot}_memory.json")

    def load(self, slot):
        """Load one seat's memory, or an empty memory if there is no usable file."""
        try:
            with open(self.memory_path(slot), encoding="utf-8") as f:
                return bot_state_from_json(json.load(f), slot)
        except (OSError, ValueError):
            return new_bot_state(slot)

    def save(self, st):
        """Save one seat's memory safely through a temporary file."""
        os.makedirs(self.dir, exist_ok=True)
        path = self.memory_path(st["slot"])
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(bot_state_to_json(st), f, indent=1)
        os.replace(path + ".tmp", path)

    def records(self):
        """Return every recorded game, skipping damaged lines."""
        out = []
        try:
            with open(self.records_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            pass                      # skip a damaged line, keep the rest
        except OSError:
            pass
        return out

    def next_id(self):
        """Return the number of the next game record."""
        return len(self.records()) + 1

    # ---- tuned variables (written by 007-train-bot.py after every game) ------
    @property
    def params_path(self):
        """Return the path of the tuned-variables file."""
        return os.path.join(self.dir, "tuned_params.json")

    def load_params(self):
        """Load the tuned variables, using the defaults where the file has none."""
        out = dict(TUNABLE_DEFAULTS)
        try:
            with open(self.params_path, encoding="utf-8") as f:
                d = json.load(f)
            for k in TUNABLE_DEFAULTS:
                if isinstance(d.get(k), (int, float)) and d[k] >= 0:
                    out[k] = float(d[k])
            out["history"] = d.get("history", [])
        except (OSError, ValueError, AttributeError):
            pass
        return out

    def save_params(self, params):
        """Save the tuned variables safely through a temporary file."""
        os.makedirs(self.dir, exist_ok=True)
        with open(self.params_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=1)
        os.replace(self.params_path + ".tmp", self.params_path)

    def add_game(self, rec):
        """Save a finished game and let every bot seat that played learn from it."""
        os.makedirs(self.dir, exist_ok=True)
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        for slot in range(1, MAX_SLOT + 1):
            if slot < rec["n"]:
                st = self.load(slot)
                apply_record(st, rec)
                self.save(st)

    @staticmethod
    def summary(records):
        """Count the recorded games, wins, losses and draws."""
        s = {"games": len(records), "wins": 0, "losses": 0, "draws": 0, "by_mode": {}}
        for r in records:
            w = r.get("winner")
            k = "wins" if w == 0 else ("draws" if w is None else "losses")
            s[k] += 1
            m = s["by_mode"].setdefault(r["n"], [0, 0, 0])
            m[0 if w == 0 else (2 if w is None else 1)] += 1
        return s


class CloneBot:
    """Play like the recorded human, for the trainer's practice games.

    It samples moves from a bot's model of the human and is used by 007-train-bot.py.
    """

    def __init__(self, pid, model, rng=None):
        """Create a copy of the human from a bot's model of them."""
        self.pid, self.model = pid, model
        self.rng = rng or random.Random()
        self.last, self.streak = None, 0

    def observe(self, players_before, actions):
        """Track this player's own last move and Defense streak."""
        a = actions.get(self.pid)
        if a:
            self.streak = self.streak + 1 if a["action"] == "defense" else 0
            self.last = a["action"]

    def choose(self, players, round_no=0):
        """Choose a move the way the recorded human tends to."""
        me = next(p for p in players if p["id"] == self.pid)
        acts = legal_actions(me, players)
        emp, n = self.model.dist(me["bullets"], acts, merge_bomb=False, last=self.last)
        p = dict(emp) if emp else {a: 1.0 for a in acts}
        s = sum(p.values())
        p = {a: w / s for a, w in p.items()}
        p, _ = apply_streak(p, self.streak, self.model, human_rest(acts, emp, n))
        a = acts[sample_index([p[x] for x in acts], self.rng)]
        tgt = None
        if a in ("shoot", "mg"):
            opps = [o for o in players if o["alive"] and o["id"] != self.pid]
            ws = [1.0 + 0.5 * o["bullets"] for o in opps]
            tgt = opps[sample_index([w / sum(ws) for w in ws], self.rng)]["id"]
        return {"action": a, "target": tgt}


# ============================================================================
#  HEADLESS TOOLS: self-test, tournament, solver
# ============================================================================
def play_headless(bots, n_players, max_rounds=150):
    """Play one game between bots without a window and return the winner.

    `bots` maps player ids to bots.  It returns the winner's id, or None for a draw or a stalemate.
    """
    players = [make_player(i, f"P{i}") for i in range(n_players)]
    for rnd in range(1, max_rounds + 1):
        if len(alive_players(players)) <= 1:
            break
        pub = [dict(p) for p in players]
        actions = {p["id"]: bots[p["id"]].choose(pub, rnd) for p in players if p["alive"]}
        out = resolve_round(players, actions)
        for b in bots.values():
            b.observe(pub, out["actions"])
        players = out["players"]
    live = alive_players(players)
    return live[0]["id"] if len(live) == 1 else None


def run_selftest():
    """Run the built-in checks of the rules, the solver, the bots and the memory.

    It is started with --selftest, prints PASS or FAIL for every check and works in a temporary folder.
    """
    ok = True

    def check(name, cond):
        """Print PASS or FAIL for one check and remember any failure."""
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok &= bool(cond)

    def mk(*bullets):
        """Create players with the given bullet counts."""
        ps = [make_player(i, f"P{i}") for i in range(len(bullets))]
        for p, b in zip(ps, bullets):
            p["bullets"] = b
        return ps

    def R(ps, **acts):
        """Resolve a round with actions given as keyword arguments."""
        return resolve_round(ps, {int(k[1:]): v for k, v in acts.items()})

    print("Rules engine")
    A = lambda a, t=None: {"action": a, "target": t}
    o = R(mk(3, 3, 3), p0=A("shoot", 1), p1=A("shoot", 0), p2=A("shoot", 0))
    check("A<->B shoot cancel, C also shoots A  -> only A dies", o["deaths"] == [0])
    o = R(mk(10, 10), p0=A("mg", 1), p1=A("mg", 0))
    check("mutual MG -> both die (draw)", o["deaths"] == [0, 1] and not any(p["alive"] for p in o["players"]))
    o = R(mk(10, 10, 0), p0=A("mg", 1), p1=A("mg", 0), p2=A("reload"))
    check("mutual MG with a 3rd player -> third survives", o["deaths"] == [0, 1] and o["players"][2]["alive"])
    o = R(mk(1, 0), p0=A("shoot", 1), p1=A("reflect"))
    check("Reflect kills the shooter, reflector lives", o["deaths"] == [0])
    o = R(mk(1, 0), p0=A("shoot", 1), p1=A("defense"))
    check("Defense blocks Shoot", o["deaths"] == [])
    o = R(mk(1, 0), p0=A("shoot", 1), p1=A("reload"))
    check("Reload is vulnerable and gains nothing when killed", o["deaths"] == [1])
    o = R(mk(10, 0), p0=A("mg", 1), p1=A("defense"))
    check("MG ignores Defense", o["deaths"] == [1])
    o = R(mk(10, 0), p0=A("mg", 1), p1=A("reflect"))
    check("MG ignores Reflect", o["deaths"] == [1])
    o = R(mk(2, 0, 0), p0=A("bomb"), p1=A("reload"), p2=A("defense"))
    check("Bomb: reloader dies, defender lives, cost n-1 = 2", o["deaths"] == [1] and o["players"][0]["bullets"] == 0)
    o = R(mk(2, 1, 0), p0=A("bomb"), p1=A("shoot", 0), p2=A("reload"))
    check("Bomb vs Shoot cancel pair; third reloader dies", o["deaths"] == [2] and o["players"][0]["alive"] and o["players"][1]["alive"])
    o = R(mk(0, 0), p0=A("reload"), p1=A("reload"))
    check("Both reload -> +1 bullet each", [p["bullets"] for p in o["players"]] == [1, 1])
    o = R(mk(1, 10, 0), p0=A("shoot", 1), p1=A("mg", 0), p2=A("reload"))
    check("Shoot at an MG-firing player is NOT cancelled; both die", set(o["deaths"]) == {0, 1})
    o = R(mk(1, 1, 0), p0=A("shoot", 2), p1=A("shoot", 2), p2=A("reflect"))
    check("Reflect kills every attacker", set(o["deaths"]) == {0, 1})
    o = R(mk(3, 3), p0=A("shoot", 1), p1=A("bomb"))
    check("Shoot <-> Bomb cancel", o["deaths"] == [])

    print("Matrix-game solver")
    x, y, v = solve_matrix_game([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    check("rock-paper-scissors value 0, uniform", abs(v) < 1e-9 and all(abs(p - 1 / 3) < 1e-9 for p in x + y))
    x, y, v = solve_matrix_game([[3, -1], [-2, 1]])
    check("2x2 mixed game value 1/7", abs(v - 1 / 7) < 1e-9)
    x, y, v = solve_matrix_game([[1, 2], [0, 3]])
    check("saddle point game value 1", abs(v - 1) < 1e-9 and x[0] > 0.999)

    print("Solved 2-player table")
    V = get_v2p()
    check("start position value ~ 0 (symmetric game)", abs(V[s2idx(0, 0, 1, 1)]) < 0.02)
    check("10 bullets vs 0 is a forced win (~ +1)", V[s2idx(10, 0, 1, 1)] > 0.99)
    check("0 vs 10 bullets is lost (~ -1) when you cannot also fire MG", V[s2idx(0, 10, 1, 1)] < -0.5)
    check("table is antisymmetric", all(abs(V[s2idx(a, b, r, q)] + V[s2idx(b, a, q, r)]) < 0.02
                                        for a in range(0, 12, 3) for b in range(0, 12, 2) for r in (0, 1) for q in (0, 1)))
    me, opp = make_player(0, "m"), make_player(1, "o")
    me["bullets"] = 10
    la, lo, xs, ys, v, _ = duel_solution(me, opp)
    check("bot with 10 bullets vs 0 plays MG 100%", xs[la.index("mg")] > 0.999)
    print("Secured rule: no Defense/Reflect when every opponent has 0 bullets")
    for diff in ("easy", "normal", "hard"):
        bot = ThinkingBot(0, diff, rng=random.Random(1))
        res = {bot.choose(mk(0, 0), 1)["action"] for _ in range(60)}
        check(f"{diff}: round 1 (2p) -> Reload only", res == {"reload"})
        res = {bot.choose(mk(0, 0, 0), 1)["action"] for _ in range(60)}
        check(f"{diff}: nobody has bullets (3p) -> Reload only", res == {"reload"})
        for ps in (mk(3, 0), mk(5, 0, 0), mk(6, 0, 0, 0, 0)):
            res = {bot.choose(ps, 5)["action"] for _ in range(150)}
            check(f"{diff}: {len(ps)}p armed vs empty opponents -> never Defense/Reflect (saw {sorted(res)})",
                  not (res & {"defense", "reflect"}) and "reload" in res)
    bot = ThinkingBot(0, "hard", rng=random.Random(1))
    check("10 bullets vs lone 0-bullet foe -> MG", bot.choose(mk(10, 0), 9)["action"] == "mg")
    print("v10: defense streak pattern")
    m = OppMemory()
    for a, want in (("defense", 1), ("defense", 2), ("reload", 0), ("defense", 1), ("shoot", 0)):
        m.update(3, a, None)
        check(f"streak after {a} = {want}", m.streak == want)
    fresh = OppMemory()
    got = [round(fresh.streak_def_prob(d)[0], 3) for d in (1, 2, 3, 4, 7)]
    check(f"prior P(defend again) after 1,2,3,4,7+ defenses = 50/33/15/5/5 % (got {got})",
          got == [0.5, 0.33, 0.15, 0.05, 0.05])
    check("streak 0 has no prior", fresh.streak_def_prob(0)[0] is None)
    pd, used = apply_streak({"reload": 0.2, "defense": 0.7, "shoot": 0.1}, 3, fresh)
    check("apply_streak keeps a distribution and sets Defense to 15 %",
          used and abs(sum(pd.values()) - 1) < 1e-9 and abs(pd["defense"] - 0.15) < 1e-9
          and abs(pd["reload"] / pd["shoot"] - 2.0) < 1e-9)
    m = OppMemory(decay=1.0)
    for i in range(30):
        m.update(i % 7, ("defense", "reload", "shoot")[i % 3], 1)
    m2 = OppMemory.from_json(json.loads(json.dumps(m.to_json())))
    d1 = m.dist(3, ["reload", "defense", "shoot"], False, last="defense")
    d2 = m2.dist(3, ["reload", "defense", "shoot"], False, last="defense")
    check("memory JSON round trip keeps the model", abs(d1[1] - d2[1]) < 0.01 and all(abs(d1[0][a] - d2[0][a]) < 0.01 for a in d1[0]))

    def bot_vs_human(streak_len):
        """Return the bot's mix after it has seen the human defend a given number of times."""
        bot = ThinkingBot(0, "hard", rng=random.Random(5))
        hm = bot._mem(1)
        for _ in range(streak_len):
            hm.update(2, "defense", None)
        ps = mk(3, 2)
        ps[1]["human"] = True
        bot.choose(ps, 4)
        return dict(bot.last_probs)
    p0, p3 = bot_vs_human(0), bot_vs_human(3)
    check(f"after 3 human defenses the bot shoots more (shoot {p0.get('shoot', 0):.2f} -> {p3.get('shoot', 0):.2f})",
          p3.get("shoot", 0) > p0.get("shoot", 0) + 0.15)

    print("v10: permanent bot memory")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        store = BotStore(tmp)

        def rec(n, winner):
            """Build a short test game record."""
            hist = [{i: {"action": "reload", "target": None} for i in range(n)},
                    {i: {"action": "defense", "target": None} for i in range(n)}]
            return make_record(store.next_id(), {"difficulty": "hard", "timer": 10}, n, hist, winner)
        store.add_game(rec(2, 0))
        store.add_game(rec(3, 1))
        store.add_game(rec(5, 4))
        g = {s: store.load(s)["games"] for s in range(1, 5)}
        check(f"bot 1 learns from 2p+3p+5p, bot 2 from 3p+5p, bots 3,4 from 5p only (games {g})",
              g == {1: 3, 2: 2, 3: 1, 4: 1})
        s1 = store.load(1)
        check("bot 1 file: 1 win and 2 losses recorded", (s1["wins"], s1["losses"]) == (1, 2))
        check("human model counts the recorded rounds", sum(s1["human"].c0.values()) > 5)
        sm = BotStore.summary(store.records())
        check(f"summary counts games: {sm['games']} played, {sm['wins']} human wins", sm["games"] == 3 and sm["wins"] == 1)
        bot = ThinkingBot(1, "normal", rng=random.Random(2), state=store.load(1))
        check("new bot starts with the stored history as a frozen base", bot.mem[0].base is not None and bot.mem[0].base_w > 0)
        big = OppMemory(decay=1.0)
        for i in range(500):
            big.update(i % 7, ("defense", "reload", "shoot")[i % 3], None)
        mm = OppMemory()
        mm.attach_base(big)
        check("stored history is not capped (500 moves count in full)", mm.base_w == 1.0)
        store.save_params({"STREAK_PRIOR_WEIGHT": 11.0, "PATTERN_ROUNDS": 3.0})
        pr = store.load_params()
        check("tuned variables survive a save / load", pr["STREAK_PRIOR_WEIGHT"] == 11.0 and pr["PATTERN_ROUNDS"] == 3.0)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return 0 if ok else 1


def run_tournament(games, samples=60):
    """Measure the bots' strength against the baseline bots."""
    rng = random.Random(7)
    print(f"Tournament: {games} games per match-up (draw/stalemate counted separately)\n")

    def duel(name, make_a, make_b):
        """Play one 1-against-1 match-up, alternating the seats."""
        w = [0, 0, 0]
        for g in range(games):
            flip = g % 2
            bots = {0: make_a(0 if not flip else 1), 1: make_b(1 if not flip else 0)}
            if flip:
                bots = {1: make_a(1), 0: make_b(0)}
            res = play_headless(bots, 2)
            if res is None:
                w[2] += 1
            elif (res == 0) != bool(flip):
                w[0] += 1
            else:
                w[1] += 1
        print(f"  {name:<46} wins {w[0]:>4} | losses {w[1]:>4} | draws {w[2]:>4}   score {(w[0] + 0.5 * w[2]) / games:.3f}")

    print("2 players")
    for lvl in ("hard", "normal", "easy"):
        t0 = time.time()
        mkb = lambda pid, l=lvl: ThinkingBot(pid, l, random.Random(rng.random()))
        duel(f"{lvl:<6} bot vs random", mkb, lambda pid: RandomBot(pid, random.Random(rng.random())))
        duel(f"{lvl:<6} bot vs always-defend", mkb, lambda pid: FixedBot(pid, "defense"))
        duel(f"{lvl:<6} bot vs always-reload", mkb, lambda pid: FixedBot(pid, "reload"))
        duel(f"{lvl:<6} bot vs heuristic 'sensible human'", mkb, lambda pid: HeuristicBot(pid, random.Random(rng.random())))
        print(f"    ({time.time() - t0:.1f}s)")
    duel("hard bot vs hard bot (mirror, expect ~0.5)",
         lambda pid: ThinkingBot(pid, "hard", random.Random(rng.random())),
         lambda pid: ThinkingBot(pid, "hard", random.Random(rng.random())))

    print("\nMulti-player (focal hard bot in seat 0; fair share would be 1/N)")
    for n, g in ((3, max(10, games // 3)), (5, max(6, games // 6))):
        for name, make_opp in (("random", lambda pid: RandomBot(pid, random.Random(rng.random()))),
                               ("heuristic", lambda pid: HeuristicBot(pid, random.Random(rng.random())))):
            wins = dr = 0
            t0 = time.time()
            for gi in range(g):
                seat = gi % n
                bots = {i: make_opp(i) for i in range(n)}
                bots[seat] = ThinkingBot(seat, "hard", random.Random(rng.random()), samples=samples)
                res = play_headless(bots, n, max_rounds=120)
                wins += res == seat
                dr += res is None
            print(f"  {n} players vs {name:<9} bots: focal wins {wins}/{g} ({wins / g:.0%}), draws {dr}   "
                  f"[fair share {1 / n:.0%}]  ({time.time() - t0:.0f}s)")
    return 0


def run_solve():
    """Re-solve the 2-player game and print the text to embed in this file."""
    t0 = time.time()
    V = compute_value_table(verbose=True)
    s = pack_table(V)
    print(f"solved {len(V)} states in {time.time() - t0:.1f}s, embedded string length {len(s)}")
    print(s)
    return 0


# Solved table: (b1,b2,r1,r2) -> value for player 1, produced by `--solve`.
_V2P_B64 = (
    "eNqF0nlM0wcUB/Bf75ueQGvpAW3pAa1TdDhkKkoYI+pUJDodYlGDmcfQ6TYmeIFjm06ZxyJRqch0GkSdGsYIinPoZIiTQu8CpRYK"
    "tpRC7+vXNVn238zeS755/33yXh4ALI/chwIAaO+29bjvuarGmkfhM9nTT82J5gXOvVOjwznDlQ7FpGjwK2Ot/YgtQ39KF//6z/Ff"
    "1FR105jXslrZ2tdhahja+zza4+nfoNz+5L8bSp2Py6StwANATqQz5l0JSP3VEUq4wJvpQYfsgaeu7pmFAaZ/zXSBc4eP4z3mkE8u"
    "87S4wNd3Jz6ekTtlY/ctPEelzTm0YTB/jPpKrhT3LVXJ+ideLHlxsd/Wd6JX1Evtr+qrZoQJRbO64jYhq6D9KAcMAASgAQIAlEhH"
    "KC8aitBDXYHWSElYEGjyZYfrg93egHtzcJn/omvzdIfP5Q44Muw/ujxT5VaGpdrebL2hPaw6aMjWcAYYygGtXPWWsunlqObeQBbr"
    "GXGCdZLUivHDu7E0pAPaDuTCEbHNLoDbYt52MBA2R2tBdzg91BabSaHf/NkRdXCVP9/jCzb64G6kc47v7Mz39l3WXOfUxK/GlZqF"
    "w2e0NzWEgXFDg/pz1SXlbb1BxeTWkDu5ckoW/gkykzCKGocXQLORjbDzEDaAglXEpOvg+lg6QH5kKXACPBIBQoRofXh+qM0nidQF"
    "VvmOzjwLnHaznc0TXFer7cFwndY7Uq+T63erZENnNUgtZqDT+FStTamgfsk7RLsVdw7dShzBGJBdsGJ0GQIPy4cMwFEwCuRh1Ajd"
    "F5PWgjdj6QU/DTcCEpAdKQugo3mhm4HVbnh4vbd/Rm5r8hROzjOf071jqdLXG6vVSaZyLU8vVNUONWh6Bbb4PalvJx4gP8AqKIX4"
    "TMxlBAr3AOWEd0EXoySIQ1AXACDKYFhISfQ7mDomnYrsiV21FmwPoiCSyPFQn6cIzPOzPCb7Aj9parfluH7d2FbDiaG9Gv9Ikc5r"
    "iFfnm2q074uLGZNiLBMa/wNhZYKCuBWfi74cR8DdQWEQbdgSdA9cDc1HlyPNUD9Qj2yHV0ACYB5CHZPSwqrY15REtvnWQRzBLb4D"
    "jrLQpunOsZ8MTeNHjZ+ZSrVkyxf6a4MOdaH5lu62NJdTICtNVjIGycuZNTQU+SN8LxVHCuP+QgeJOwl30AuQBwm7cWuRYdht3HXM"
    "fngzZDa2HdUIvQFeQFMQAKALCJEAsDFocmYA8z1dE9sG79gKhszmn3VYa4nxzHBQs2K0xXAyA5cOmXdNSkpuY1xLmUyqo1dRrcwi"
    "+ifUDuIZOjnBTDThixIKaFJCMVZBa6G8wj5DcalO0iV0PfwY+UrcVURvlEvswQLAjDcOAwCLpmpGEqaLzEYraHxsTxq5+uqh/l3b"
    "I9P5rH9aIX0o0ciU6euER/mHJV2i93hPuHdFu4TLuFh2ubAs9TnrD+aO1GrBaeYNxn7BRX4OozjxQ/5BXnbCKtqFlNbkx6TXcZXs"
    "ZhbwhvrXSZyzNf3RnHjZSFq60CpVSNaK6HxXWqlYIVAmn5bsFJXxLJwqcZ1wXzKafVLUnPotZ4rZIPxawGZ56R+kNvL49EVUKM/C"
    "/j9n39zJdO3catlLKVNUPHtF2jeS1YItUoEkKMzh+dKyxIOCimSdpFSE5bdxbOJK4YaUFhZFvCZ1F/feLKSokL8x6XfaUsESzpuc"
    "vwHcmmnq"
)

# ============================================================================
#  UI - everything below needs pygame
# ============================================================================
W, H = 1280, 720

C_BG = (11, 13, 30)
C_PANEL = (22, 26, 54)
C_PANEL2 = (32, 38, 78)
C_TEXT = (240, 240, 250)
C_MUTED = (150, 158, 190)
C_GOLD = (255, 205, 70)
C_GREEN = (70, 225, 140)
C_RED = (245, 75, 90)
C_BLUE = (95, 175, 255)
C_PURPLE = (190, 125, 255)
C_ORANGE = (255, 150, 60)

ACT_COLOR = {"reload": (120, 205, 255), "shoot": (255, 205, 70), "defense": (90, 225, 150),
             "bomb": (255, 150, 60), "reflect": (190, 125, 255), "mg": (255, 85, 95)}
ACT_NAME = {"reload": "RELOAD", "shoot": "SHOOT", "defense": "DEFENSE", "bomb": "BOMB",
            "reflect": "REFLECT", "mg": "MACHINE GUN"}
ACT_KEY = {"reload": "R", "shoot": "S", "defense": "D", "bomb": "B", "reflect": "E", "mg": "M"}
ACT_ORDER = ["reload", "shoot", "defense", "bomb", "reflect", "mg"]
AVATAR_COLORS = [(70, 175, 255), (255, 145, 70), (185, 115, 255), (70, 205, 195), (255, 105, 165)]

SHOTS_TAKEN = set()


class QuitGame(Exception):
    """Signal that the window was closed."""
    pass


class ReturnToMenu(Exception):
    """Signal that the player left the running match."""


class ReplayStop(Exception):
    """Signal that the player stopped a replay."""


MENU_BTN = pygame.Rect(W - 134, 56, 110, 32) if pygame else None


def speed():
    """Return the animation speed factor."""
    return 4.0 if OPTS.fast else 1.0


def snap(ctx, tag):
    """Save a screenshot of a key screen when testing with --shots."""
    if OPTS.shots and tag not in SHOTS_TAKEN:
        SHOTS_TAKEN.add(tag)
        os.makedirs(OPTS.shots, exist_ok=True)
        pygame.image.save(ctx.screen, os.path.join(OPTS.shots, f"{tag}.png"))


def lerp(a, b, t):
    """Return the point a fraction t of the way from a to b."""
    return a + (b - a) * t


def lerp_col(c1, c2, t):
    """Mix two colours by the fraction t."""
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def ease_out(t):
    """Shape an animation so it starts fast and stops softly."""
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


class Ctx:
    """Hold everything that the screens share."""
    def __init__(self, screen, fonts, clock):
        """Create the shared state for a new window."""
        self.screen, self.fonts, self.clock = screen, fonts, clock
        self.cfg = {}
        self.log = []
        self.insight = {}           # {bot id: [(action, probability), ...]} of the last round
        self.bot_names = {}         # {bot id: name} of the bots in the running match (empty in hot-seat / replay)
        self.show_p = False         # SHOW P / HIDE P button state
        self.pbtn_shown = False     # the SHOW P button was drawn this frame
        self.players = []
        self.t_game = 0.0
        self.in_match = False       # True while a match is running (shows the MENU button)
        self.btn_shown = False      # the MENU button was drawn this frame
        self.paused = 0.0           # seconds spent in the confirm dialog (not charged to the round timer)
        self.store = None           # BotStore: permanent bot memory + game records
        self.stats = BotStore.summary([])    # games played so far (shown on the setup / game-over screens)
        self.replaying = False      # True while a recorded game is replayed (Esc stops it)
        self.secret_rect = None     # clickable spot of the hidden line on the golden-rules page
        self.secret_found = False   # the hidden line was revealed (stays revealed for the session)


def _find_font(bold):
    """Find a font file on this computer."""
    wdir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    names = (["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf", "simhei.ttf"] if bold else
             ["segoeui.ttf", "arial.ttf", "calibri.ttf", "simhei.ttf"])
    for n in names + ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]:
        for d in (wdir, "/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "/System/Library/Fonts/Supplemental"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None


def load_fonts():
    """Load the fonts the game uses in every size."""
    reg, bold = _find_font(False), _find_font(True)

    def f(sz, b=False):
        """Load one font size, falling back to pygame's default font."""
        path = (bold if b else reg) or reg or bold
        try:
            return pygame.font.Font(path, sz) if path else pygame.font.Font(None, int(sz * 1.2))
        except Exception:
            return pygame.font.Font(None, int(sz * 1.2))
    return {"title": f(62, True), "big": f(40, True), "med": f(27, True), "body": f(22), "bodyb": f(22, True),
            "small": f(18), "tiny": f(15), "num": f(30, True)}


def text(ctx, font, s, pos, color=C_TEXT, anchor="topleft", shadow=False, alpha=255):
    """Draw a text at a position given by one point of its box."""
    surf = ctx.fonts[font].render(str(s), True, color)
    if alpha < 255:
        surf.set_alpha(alpha)
    r = surf.get_rect()
    setattr(r, anchor, pos)
    if shadow:
        sh = ctx.fonts[font].render(str(s), True, (0, 0, 0))
        sh.set_alpha(min(alpha, 140))
        ctx.screen.blit(sh, (r.x + 2, r.y + 2))
    ctx.screen.blit(surf, r)
    return r


def wrap(ctx, font, s, width):
    """Split a text into lines that fit a given width."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if ctx.fonts[font].size(t)[0] <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def rrect(surf, rect, color, radius=14, width=0):
    """Draw a rounded rectangle, filled or as an outline."""
    pygame.draw.rect(surf, color, rect, width=width, border_radius=radius)


def glow(surf, pos, radius, color, alpha=90):
    """Draw a soft round glow."""
    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for i in range(6, 0, -1):
        pygame.draw.circle(s, (*color, int(alpha * i / 6 / 2)), (radius, radius), int(radius * i / 6))
    surf.blit(s, (pos[0] - radius, pos[1] - radius))


def draw_icon(surf, kind, cx, cy, r, color):
    """Draw the icon of an action."""
    lw = max(2, int(r * 0.16))
    if kind == "reload":
        rect = pygame.Rect(cx - r * 0.75, cy - r * 0.75, r * 1.5, r * 1.5)
        pygame.draw.arc(surf, color, rect, 0.7, 5.5, lw + 1)
        a = 0.7
        px, py = cx + 0.75 * r * math.cos(a), cy - 0.75 * r * math.sin(a)
        pygame.draw.polygon(surf, color, [(px + r * 0.45, py - r * 0.05), (px - r * 0.1, py - r * 0.4),
                                          (px + r * 0.05, py + r * 0.35)])
        pygame.draw.rect(surf, color, (cx - r * 0.12, cy - r * 0.3, r * 0.24, r * 0.6), border_radius=2)
    elif kind in ("shoot", "mg"):
        n = 3 if kind == "mg" else 1
        for i in range(n):
            oy = cy + (i - (n - 1) / 2) * r * 0.62
            body = pygame.Rect(cx - r * 0.8, oy - r * 0.2, r * 1.0, r * 0.4)
            pygame.draw.rect(surf, color, body, border_radius=3)
            pygame.draw.polygon(surf, color, [(cx + r * 0.2, oy - r * 0.2), (cx + r * 0.85, oy),
                                              (cx + r * 0.2, oy + r * 0.2)])
            pygame.draw.line(surf, (20, 20, 40), (cx - r * 0.45, oy - r * 0.2), (cx - r * 0.45, oy + r * 0.2), 2)
    elif kind == "defense":
        pts = [(cx - r * 0.75, cy - r * 0.75), (cx + r * 0.75, cy - r * 0.75), (cx + r * 0.75, cy + r * 0.1),
               (cx, cy + r * 0.9), (cx - r * 0.75, cy + r * 0.1)]
        pygame.draw.polygon(surf, color, pts)
        pygame.draw.polygon(surf, (20, 20, 40), pts, 2)
        pygame.draw.line(surf, (20, 20, 40), (cx, cy - r * 0.6), (cx, cy + r * 0.6), 2)
    elif kind == "bomb":
        pygame.draw.circle(surf, color, (int(cx - r * 0.05), int(cy + r * 0.2)), int(r * 0.62))
        pygame.draw.line(surf, color, (cx + r * 0.3, cy - r * 0.3), (cx + r * 0.6, cy - r * 0.7), lw)
        for ang in (0, 1.2, 2.4, 3.6, 4.8):
            pygame.draw.line(surf, C_GOLD, (cx + r * 0.62, cy - r * 0.74),
                             (cx + r * 0.62 + r * 0.28 * math.cos(ang), cy - r * 0.74 + r * 0.28 * math.sin(ang)), 2)
    elif kind == "reflect":
        pygame.draw.line(surf, color, (cx + r * 0.35, cy - r * 0.85), (cx + r * 0.35, cy + r * 0.85), lw + 2)
        pygame.draw.line(surf, color, (cx - r * 0.85, cy - r * 0.7), (cx + r * 0.3, cy), lw)
        pygame.draw.line(surf, color, (cx + r * 0.3, cy), (cx - r * 0.7, cy + r * 0.65), lw)
        pygame.draw.polygon(surf, color, [(cx - r * 0.9, cy + r * 0.75), (cx - r * 0.35, cy + r * 0.7),
                                          (cx - r * 0.7, cy + r * 0.25)])


# ---------------------------------------------------------------------------
#  Table (players on screen)
# ---------------------------------------------------------------------------
ARENA_X0, ARENA_X1 = 430, 1230
AV_R = 44


def layout(players, viewer_id):
    """Compute where every player is drawn on the screen.

    The opponents stand in a row at the top and the viewer lower in the middle.
    """
    pos = {}
    others = [p for p in players if p["id"] != viewer_id]
    k = len(others)
    for i, p in enumerate(others):
        pos[p["id"]] = ((ARENA_X0 + ARENA_X1) / 2, 190) if k == 1 else \
            (ARENA_X0 + (ARENA_X1 - ARENA_X0) * (i + 0.5) / k, 190)
    pos[viewer_id] = ((ARENA_X0 + ARENA_X1) / 2, 430)      # a bit lower so its action icon does not cover the bots' bullet row
    return pos


def avatar_color(p):
    """Return the fixed colour of a player's seat."""
    return AVATAR_COLORS[p["id"] % len(AVATAR_COLORS)]


def draw_table(ctx, players, pos, *, alive_vis=None, bubbles=None, rings=None, tnow=0.0, shake=None,
               bullets_vis=None, number_badges=None, dim=False):
    """Draw all players with their bullets, Reflect status and action bubbles.

    alive_vis maps ids to whether the player is still shown alive, bubbles maps ids to
    (action, target name, scale), and rings maps ids to a highlight colour.
    """
    s = ctx.screen
    for p in players:
        pid = p["id"]
        x, y = pos[pid]
        if shake and pid in shake:
            amp = shake[pid]
            x += math.sin(tnow * 70) * amp
            y += math.cos(tnow * 55) * amp * 0.5
        alive = p["alive"] if not alive_vis else alive_vis.get(pid, p["alive"])
        col = avatar_color(p) if alive else (70, 72, 90)
        if alive:
            glow(s, (x, y), AV_R + 26, col, 70)
        pygame.draw.circle(s, col, (int(x), int(y)), AV_R)
        pygame.draw.circle(s, (255, 255, 255) if alive else (110, 112, 130), (int(x), int(y)), AV_R, 3)
        # agent silhouette
        dark = (18, 20, 42) if alive else (40, 42, 58)
        pygame.draw.circle(s, dark, (int(x), int(y - 9)), 13)
        pygame.draw.ellipse(s, dark, (x - 23, y + 5, 46, 34))
        if not alive:
            pygame.draw.line(s, C_RED, (x - 26, y - 26), (x + 26, y + 26), 7)
            pygame.draw.line(s, C_RED, (x + 26, y - 26), (x - 26, y + 26), 7)
        text(ctx, "small", p["name"], (x, y + AV_R + 6), C_TEXT if alive else C_MUTED, "midtop", shadow=True)
        b = p["bullets"] if bullets_vis is None else bullets_vis.get(pid, p["bullets"])
        by = y + AV_R + 34
        bc = C_RED if b >= MG_COST else (C_GOLD if b >= 7 else C_TEXT)
        if alive:
            draw_icon(s, "shoot", int(x - 28), int(by + 9), 13, bc)
            text(ctx, "med", f"{b}", (x - 10, by - 6), bc)
            draw_icon(s, "defense", int(x + 32), int(by + 9), 10, C_PURPLE if p["has_reflect"] else (75, 78, 100))   # Reflect still available?
            if b >= MG_COST:
                text(ctx, "tiny", "MG READY", (x + 50, by + 9), C_RED, "midleft")
        if rings and pid in rings and alive:
            rc = rings[pid]
            rad = AV_R + 10 + 4 * math.sin(tnow * 9)
            pygame.draw.circle(s, rc, (int(x), int(y)), int(rad), 4)
            glow(s, (x, y), int(rad) + 14, rc, 110)
        if number_badges and pid in number_badges:
            pygame.draw.circle(s, (0, 0, 0), (int(x + 34), int(y - 34)), 14)
            text(ctx, "small", number_badges[pid], (x + 34, y - 34), C_TEXT, "center")
        if bubbles and pid in bubbles and alive:
            act, sub, sc = bubbles[pid]
            if sc > 0:
                size = int(34 * sc)
                cx, cy = int(x), int(y - AV_R - 36)
                rect = pygame.Rect(0, 0, size * 2, size * 2)
                rect.center = (cx, cy)
                c = ACT_COLOR[act]
                rrect(s, rect, (24, 28, 60), 14)
                rrect(s, rect, c, 14, 3)
                draw_icon(s, act, cx, cy - 3, int(size * 0.62), c)
                if sc >= 0.95 and sub:
                    text(ctx, "tiny", f"> {sub}", (cx, cy - size - 2), c, "midbottom", shadow=True)


def draw_header(ctx, round_no, right_text=""):
    """Draw the background, the round number, the mode label and the MENU button."""
    s = ctx.screen
    s.fill(C_BG)
    for i in range(0, H, 4):                     # subtle scanlines
        pygame.draw.line(s, (13, 15, 34), (0, i), (W, i))
    text(ctx, "med", f"ROUND {round_no}" if round_no else "", (24, 22), C_GOLD)
    if right_text:
        text(ctx, "small", right_text, (W - 24, 28), C_MUTED, "topright")
    if ctx.in_match:
        hot = MENU_BTN.collidepoint(pygame.mouse.get_pos())
        rrect(s, MENU_BTN, (70, 40, 60) if hot else C_PANEL, 10)
        text(ctx, "small", "< MENU", MENU_BTN.center, C_TEXT if hot else C_MUTED, "center")
        ctx.btn_shown = True


PROB_STRIP = pygame.Rect(16, 432, 380, 38) if pygame else None      # bottom of the left column: label + SHOW P / HIDE P
PROB_BTN = pygame.Rect(16 + 380 - 112, 436, 100, 30) if pygame else None
PROB_ROW_H = 76


def draw_log(ctx):
    """Draw the side log and, when SHOW P is on, the bots' probabilities.

    The log is on top; the bots' probabilities appear in their own panels below it.
    """
    s = ctx.screen
    bottom = 470
    pp = None
    if ctx.bot_names:
        bottom = PROB_STRIP.y - 6
        if ctx.show_p:
            nb = len(ctx.bot_names)
            ph = 38 + ((nb + 1) // 2 if nb > 1 else 1) * PROB_ROW_H
            pp = pygame.Rect(16, PROB_STRIP.y - 6 - ph, 380, ph)
            bottom = pp.y - 6
    panel = pygame.Rect(16, 70, 380, bottom - 70)
    rrect(s, panel, C_PANEL, 16)
    text(ctx, "small", "WHAT JUST HAPPENED", (panel.x + 16, panel.y + 12), C_MUTED)
    lines = []
    for (line, col) in ctx.log[-40:]:
        for ln in wrap(ctx, "tiny", line, panel.w - 32)[:2]:
            lines.append((ln, col))
    max_lines = max(1, (panel.bottom - 12 - (panel.y + 42)) // 19)
    y = panel.y + 42
    for (ln, col) in lines[-max_lines:]:
        text(ctx, "tiny", ln, (panel.x + 16, y), col)
        y += 19
    if not ctx.bot_names:
        return
    if pp:
        rrect(s, pp, C_PANEL, 16)
        text(ctx, "small", "BOT PROBABILITIES (last round)", (pp.x + 16, pp.y + 8), C_MUTED)
        cols = 1 if len(ctx.bot_names) == 1 else 2
        cw = (pp.w - 24) // cols
        for k, (pid, name) in enumerate(ctx.bot_names.items()):
            cx, cy = pp.x + 16 + (k % cols) * cw, pp.y + 32 + (k // cols) * PROB_ROW_H
            text(ctx, "tiny", name, (cx, cy), avatar_color({"id": pid}))
            mix = ctx.insight.get(pid)
            if mix is None:
                text(ctx, "tiny", "eliminated" if ctx.insight else "(after round 1)", (cx + 8, cy + 18), C_MUTED)
                continue
            for j, (a, p) in enumerate(mix[:3]):
                text(ctx, "tiny", f"{ACT_NAME.get(a, a)}  {p * 100:3.0f}%", (cx + 8, cy + 18 + j * 17), ACT_COLOR.get(a, C_TEXT))
    rrect(s, PROB_STRIP, C_PANEL, 16)
    text(ctx, "tiny", "Bot probabilities", (PROB_STRIP.x + 16, PROB_STRIP.centery), C_MUTED, "midleft")
    hot = PROB_BTN.collidepoint(pygame.mouse.get_pos())
    rrect(s, PROB_BTN, C_PANEL2 if hot else (28, 33, 68), 10)
    rrect(s, PROB_BTN, C_PURPLE, 10, 2)
    text(ctx, "small", "HIDE P" if ctx.show_p else "SHOW P", PROB_BTN.center, C_PURPLE, "center")
    ctx.pbtn_shown = True


def elim_text(names):
    """Build the banner text that says who was eliminated.

    Examples: 'You are eliminated', 'Bot 1 eliminated', 'You and Bot 1 are eliminated'.
    """
    if len(names) == 1:
        return f"{names[0]} are eliminated" if names[0] == "You" else f"{names[0]} eliminated"
    return ", ".join(names[:-1]) + f" and {names[-1]} " + ("are eliminated" if "You" in names else "eliminated")


def describe_events(players, outcome):
    """Describe a round in plain words for the side log."""
    names = {p["id"]: p["name"] for p in players}
    lines = []
    acts = outcome["actions"]
    for pid, a in acts.items():
        extra = f" -> {names[a['target']]}" if a.get("target") is not None else ""
        lines.append((f"{names[pid]}: {ACT_NAME[a['action']]}{extra}", C_TEXT))
    for e in outcome["events"]:
        k, a, t = e
        if k == "hit":
            lines.append((f"{names[a]} hit {names[t]}", C_RED))
        elif k == "blocked":
            lines.append((f"{names[t]} blocked {names[a]}", C_GREEN))
        elif k == "reflected":
            lines.append((f"{names[t]} reflected {names[a]}'s bullet - {names[a]} dies", C_PURPLE))
        elif k == "cancel":
            lines.append((f"{names[a]} and {names[t]}: bullets cancelled", C_BLUE))
        elif k == "mg":
            lines.append((f"{names[a]}'s Machine Gun killed {names[t]}", C_RED))
        elif k == "mg_clash":
            lines.append((f"{names[a]} and {names[t]}: MG clash, both die", C_RED))
    for pid in outcome["deaths"]:
        lines.append((elim_text([names[pid]]), C_RED))
    if not outcome["deaths"]:
        lines.append(("Nobody died.", C_GREEN))
    return lines


# ---------------------------------------------------------------------------
#  Buttons
# ---------------------------------------------------------------------------
def button_rects():
    """Return the positions of the six action buttons."""
    w, h, gap = 178, 150, 14
    x0 = (W - (6 * w + 5 * gap)) // 2
    return {a: pygame.Rect(x0 + i * (w + gap), 538, w, h) for i, a in enumerate(ACT_ORDER)}


def button_state(a, me, players):
    """Tell whether an action button can be used and what it costs.

    It returns (enabled, cost label, reason why it is disabled).
    """
    b = me["bullets"]
    bc = bomb_cost(players)
    if a == "reload":
        return True, "+1 bullet", ""
    if a == "defense":
        return True, "free", ""
    if a == "shoot":
        return b >= 1, "1 bullet", "need 1" if b < 1 else ""
    if a == "bomb":
        return b >= bc, f"{bc} bullet{'s' if bc > 1 else ''}", f"need {bc}" if b < bc else ""
    if a == "reflect":
        return me["has_reflect"], "once per game", "used" if not me["has_reflect"] else ""
    return b >= MG_COST, "10 bullets", "need 10" if b < MG_COST else ""


def draw_buttons(ctx, me, players, *, hover=None, selected=None, locked=False):
    """Draw the six action buttons."""
    s = ctx.screen
    for a, r in button_rects().items():
        ok, cost, why = button_state(a, me, players)
        c = ACT_COLOR[a]
        face = (28, 33, 68) if ok else (24, 26, 40)
        if hover == a and ok and not locked:
            face = (44, 52, 100)
        rr = r.copy()
        if selected == a:
            rr.inflate_ip(8, 8)
            glow(s, rr.center, 100, c, 110)
        rrect(s, rr, face, 16)
        rrect(s, rr, c if ok else (62, 64, 84), 16, 4 if selected == a else 2)
        draw_icon(s, a, rr.centerx, rr.y + 58, 32, c if ok else (70, 72, 92))
        text(ctx, "body", ACT_NAME[a], (rr.centerx, rr.y + 102), C_TEXT if ok else (85, 88, 110), "midtop")
        text(ctx, "tiny", why if (why and not ok) else cost, (rr.centerx, rr.y + 128),
             C_MUTED if ok else C_RED, "midtop")
        kb = pygame.Rect(rr.x + 8, rr.y + 8, 24, 24)
        rrect(s, kb, (10, 12, 26), 6)
        text(ctx, "tiny", ACT_KEY[a], kb.center, c if ok else (70, 72, 92), "center")


# ---------------------------------------------------------------------------
#  Human turn
# ---------------------------------------------------------------------------
def confirm_menu(ctx):
    """Ask whether the player really wants to leave the match.

    It returns True to leave the match.  Time spent in this dialog is not charged to the round timer.
    """
    t0 = time.time()
    snapshot = ctx.screen.copy()
    box = pygame.Rect(0, 0, 520, 220)
    box.center = (W // 2, H // 2)
    yes = pygame.Rect(box.x + 40, box.bottom - 80, 200, 52)
    no = pygame.Rect(box.right - 240, box.bottom - 80, 200, 52)
    answer = None
    while answer is None:
        ctx.clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise QuitGame()
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if yes.collidepoint(e.pos):
                    answer = True
                elif no.collidepoint(e.pos):
                    answer = False
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_RETURN, pygame.K_y):
                    answer = True
                elif e.key in (pygame.K_ESCAPE, pygame.K_n):
                    answer = False
        ctx.screen.blit(snapshot, (0, 0))
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        ctx.screen.blit(dim, (0, 0))
        rrect(ctx.screen, box, C_PANEL, 18)
        text(ctx, "med", "End this match?", (box.centerx, box.y + 28), C_GOLD, "midtop")
        text(ctx, "body", "The game is terminated and you return to the game selection screen.",
             (box.centerx, box.y + 80), C_MUTED, "midtop")
        mx = pygame.mouse.get_pos()
        rrect(ctx.screen, yes, (150, 50, 60) if yes.collidepoint(mx) else (110, 40, 50), 12)
        rrect(ctx.screen, no, (60, 90, 70) if no.collidepoint(mx) else (40, 70, 55), 12)
        text(ctx, "body", "Yes, leave (Y)", yes.center, C_TEXT, "center")
        text(ctx, "body", "No, keep playing (N)", no.center, C_TEXT, "center")
        pygame.display.flip()
    ctx.paused += time.time() - t0
    return answer


def pump_events(ctx):
    """Read this frame's window events and handle the ones that apply on every screen.

    It handles closing the window, the MENU and SHOW P buttons and Esc during a replay, and returns
    the remaining events.
    """
    evs = pygame.event.get()
    shown, ctx.btn_shown = ctx.btn_shown, False
    pshown, ctx.pbtn_shown = ctx.pbtn_shown, False
    out = []
    for e in evs:
        if e.type == pygame.QUIT:
            raise QuitGame()
        if (e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and ctx.in_match and pshown
                and PROB_BTN.collidepoint(e.pos)):
            ctx.show_p = not ctx.show_p
            continue                               # swallow the click (must not skip the reveal)
        if (e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and ctx.in_match and shown
                and MENU_BTN.collidepoint(e.pos)):
            if confirm_menu(ctx):
                raise ReturnToMenu()
            continue                               # swallow the click that opened the dialog
        if ctx.replaying and e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            raise ReplayStop()
        out.append(e)
    return out


_SCAN_TO_KEY = None


def key_of(e):
    """Return the key of a key press, even when an input method hides it.

    An active input method (for example Chinese) can swallow a key and report 'unknown key'.  Then the
    physical key position is used for letters, digits, Esc, Space, Enter and the right arrow.
    """
    if e.key:
        return e.key
    global _SCAN_TO_KEY
    if _SCAN_TO_KEY is None:
        _SCAN_TO_KEY = {pygame.KSCAN_ESCAPE: pygame.K_ESCAPE, pygame.KSCAN_SPACE: pygame.K_SPACE,
                        pygame.KSCAN_RETURN: pygame.K_RETURN, pygame.KSCAN_RIGHT: pygame.K_RIGHT}   # Space / Enter too (an IME can swallow them)
        for ch in "abcdefghijklmnopqrstuvwxyz123456789":
            _SCAN_TO_KEY[getattr(pygame, "KSCAN_" + ch.upper())] = getattr(pygame, "K_" + ch)
    return _SCAN_TO_KEY.get(getattr(e, "scancode", None), 0)


def human_turn(ctx, me, players, round_no, hotseat=False):
    """Let the player choose an action, and a target if needed, for one round.

    The player first picks an action with a button or key, then a target for Shoot or Machine Gun.
    When the timer runs out, the choice is Reload, or a Machine Gun at a random opponent with 10 or
    more bullets.
    """
    cfg = ctx.cfg
    limit = cfg["timer"]
    pos = layout(players, me["id"])
    opps = [p for p in players if p["alive"] and p["id"] != me["id"]]
    btns = button_rects()
    pending = None
    t0 = time.time()
    ctx.paused = 0.0
    hover = None
    auto_stage = 0
    rng = random.Random()
    shown_target = False
    mode_label = ctx.cfg["label"]

    def finish(action, target=None):
        """Package an action and its target."""
        return {"action": action, "target": target}

    while True:
        ctx.clock.tick(60)
        now = time.time()
        el = now - t0 - ctx.paused
        left = limit - el
        for e in pump_events(ctx):
            if e.type == pygame.MOUSEMOTION:
                hover = next((a for a, r in btns.items() if r.collidepoint(e.pos)), None)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 3:
                    pending = None
                elif e.button == 1:
                    if pending:
                        for p in opps:
                            px, py = pos[p["id"]]
                            if (e.pos[0] - px) ** 2 + (e.pos[1] - py) ** 2 <= (AV_R + 16) ** 2:
                                return finish(pending, p["id"])
                        pending = None
                    else:
                        for a, r in btns.items():
                            if r.collidepoint(e.pos) and button_state(a, me, players)[0]:
                                if a in ("shoot", "mg"):
                                    pending = a
                                else:
                                    return finish(a)
            elif e.type == pygame.KEYDOWN:
                k = key_of(e)
                if k == pygame.K_ESCAPE:
                    pending = None
                elif pending and pygame.K_1 <= k <= pygame.K_9:
                    i = k - pygame.K_1
                    if i < len(opps):
                        return finish(pending, opps[i]["id"])
                else:
                    for a in ACT_ORDER:
                        if k == getattr(pygame, "K_" + ACT_KEY[a].lower()) and button_state(a, me, players)[0]:
                            if a in ("shoot", "mg"):
                                pending = a
                            else:
                                return finish(a)
        if OPTS.autoplay:
            if auto_stage == 0 and el > 0.35 / speed():
                snap(ctx, "02_choosing")
                legal = legal_actions(me, players)
                wts = {"reload": 3, "defense": 1.5, "shoot": 2.5, "bomb": 1, "reflect": 0.6, "mg": 4}
                a = rng.choices(legal, [wts[x] for x in legal])[0]
                if a in ("shoot", "mg"):
                    pending, auto_stage = a, 1
                else:
                    return finish(a)
            elif auto_stage == 1 and el > 0.7 / speed():
                snap(ctx, "03_targeting_" + pending)
                return finish(pending, rng.choice(opps)["id"])
        if left <= 0:
            if me["bullets"] >= MG_COST:
                return finish("mg", rng.choice(opps)["id"])
            return finish("reload")

        # ---- draw
        draw_header(ctx, round_no, mode_label)
        frac = max(0.0, left / limit)
        barcol = lerp_col(C_RED, C_GREEN, min(1.0, frac * 1.6)) if frac < 0.625 else C_GREEN
        pygame.draw.rect(ctx.screen, (30, 34, 66), (0, 0, W, 10))
        pygame.draw.rect(ctx.screen, barcol, (0, 0, int(W * frac), 10))
        shake = 3 if left < 3 else 0
        text(ctx, "title", f"{max(0, math.ceil(left))}", (W // 2 + (random.randint(-shake, shake)), 50),
             C_RED if left < 3 else C_TEXT, "midtop", shadow=True)
        rings = {p["id"]: ACT_COLOR[pending] if pending == "mg" else C_BLUE for p in opps} if pending else None
        badges = {p["id"]: str(i + 1) for i, p in enumerate(opps)} if pending else None
        draw_table(ctx, players, pos, rings=rings, tnow=now, number_badges=badges)
        draw_log(ctx)
        if hotseat:
            text(ctx, "med", f"{me['name']}'s turn", (W // 2, 448), C_GOLD, "midtop")
        if pending:
            col = C_RED if pending == "mg" else C_BLUE
            text(ctx, "body", f"Pick a target for {ACT_NAME[pending]}: click a blinking player (or press 1-{len(opps)}).  Esc cancels.",
                 (745, 515), col, "midright")
        else:
            text(ctx, "body", "Choose your action  -  your choice is final once you click.",
                 (745, 515), C_MUTED, "midright")
        draw_buttons(ctx, me, players, hover=hover, selected=pending)
        pygame.display.flip()


def pass_screen(ctx, name, color):
    """Show the 'pass the device' screen between two players in Testing mode."""
    t0 = time.time()
    ctx.btn_shown = False                          # this screen has no MENU button
    while True:
        ctx.clock.tick(60)
        go = False
        for e in pump_events(ctx):
            if e.type == pygame.MOUSEBUTTONDOWN or (e.type == pygame.KEYDOWN and key_of(e) in (pygame.K_RETURN, pygame.K_SPACE)):
                go = True
        if OPTS.autoplay and time.time() - t0 > 0.2 / speed():
            go = True
        if go:
            return
        ctx.screen.fill(C_BG)
        glow(ctx.screen, (W // 2, H // 2 - 40), 220, color, 120)
        text(ctx, "med", "Pass the device to", (W // 2, 230), C_MUTED, "midtop")
        text(ctx, "title", name, (W // 2, 270), color, "midtop", shadow=True)
        text(ctx, "body", "Make sure the other players are not looking.  Click or press Enter when ready.",
             (W // 2, 400), C_TEXT, "midtop")
        pygame.display.flip()


# ---------------------------------------------------------------------------
#  Reveal animation
# ---------------------------------------------------------------------------
def build_schedule(pos, acts, outcome):
    """Plan the animation of a round as a list of timed pieces.

    The pieces are bullets, shields, bursts and beams.  It returns (pieces, {id: time of death},
    end time).
    """
    T0, TR = 1.05, 0.55
    sched, death = [], {}
    k = 0

    def col_of(pid):
        """Return the colour of a player's chosen action."""
        return ACT_COLOR[acts[pid]["action"]]
    for pid, a in acts.items():
        if a["action"] == "bomb":
            sched.append({"k": "ring", "t0": T0 - 0.15, "dur": 0.8, "p": pos[pid], "col": ACT_COLOR["bomb"]})
    for (kind, a, t) in outcome["events"]:
        t0 = T0 + 0.07 * k
        k += 1
        if kind == "hit":
            sched.append({"k": "bullet", "t0": t0, "dur": TR, "p0": pos[a], "p1": pos[t], "col": col_of(a)})
            sched.append({"k": "burst", "t0": t0 + TR, "dur": 0.6, "p": pos[t], "col": C_RED})
            death[t] = min(death.get(t, 99), t0 + TR)
        elif kind == "blocked":
            sched.append({"k": "bullet", "t0": t0, "dur": TR, "p0": pos[a], "p1": pos[t], "col": col_of(a)})
            sched.append({"k": "shield", "t0": t0 + TR, "dur": 0.55, "p": pos[t], "col": ACT_COLOR["defense"]})
        elif kind == "reflected":
            sched.append({"k": "bullet", "t0": t0, "dur": TR, "p0": pos[a], "p1": pos[t], "col": col_of(a)})
            sched.append({"k": "shield", "t0": t0 + TR, "dur": 0.55, "p": pos[t], "col": ACT_COLOR["reflect"]})
            sched.append({"k": "bullet", "t0": t0 + TR + 0.12, "dur": TR, "p0": pos[t], "p1": pos[a], "col": ACT_COLOR["reflect"]})
            sched.append({"k": "burst", "t0": t0 + 2 * TR + 0.12, "dur": 0.6, "p": pos[a], "col": C_PURPLE})
            death[a] = min(death.get(a, 99), t0 + 2 * TR + 0.12)
        elif kind == "cancel":
            mid = ((pos[a][0] + pos[t][0]) / 2, (pos[a][1] + pos[t][1]) / 2)
            sched.append({"k": "bullet", "t0": t0, "dur": TR, "p0": pos[a], "p1": mid, "col": col_of(a)})
            sched.append({"k": "bullet", "t0": t0, "dur": TR, "p0": pos[t], "p1": mid, "col": col_of(t)})
            sched.append({"k": "burst", "t0": t0 + TR, "dur": 0.7, "p": mid, "col": C_BLUE})
        elif kind == "mg":
            sched.append({"k": "beam", "t0": t0, "dur": 0.45, "p0": pos[a], "p1": pos[t], "col": ACT_COLOR["mg"]})
            sched.append({"k": "burst", "t0": t0 + 0.2, "dur": 0.7, "p": pos[t], "col": C_RED})
            death[t] = min(death.get(t, 99), t0 + 0.25)
        elif kind == "mg_clash":
            sched.append({"k": "beam", "t0": t0, "dur": 0.45, "p0": pos[a], "p1": pos[t], "col": ACT_COLOR["mg"]})
            sched.append({"k": "beam", "t0": t0, "dur": 0.45, "p0": pos[t], "p1": pos[a], "col": ACT_COLOR["mg"]})
            mid = ((pos[a][0] + pos[t][0]) / 2, (pos[a][1] + pos[t][1]) / 2)
            sched.append({"k": "burst", "t0": t0 + 0.2, "dur": 0.8, "p": mid, "col": C_ORANGE})
            death[a] = death[t] = t0 + 0.25
    end = max([s["t0"] + s["dur"] for s in sched] + [T0 + 0.4])
    return sched, death, end


def draw_sched(ctx, sched, t):
    """Draw the animation pieces that are visible at time t."""
    s = ctx.screen
    for it in sched:
        lt = t - it["t0"]
        if lt < 0 or lt > it["dur"]:
            continue
        f = lt / it["dur"]
        k = it["k"]
        if k == "bullet":
            x = lerp(it["p0"][0], it["p1"][0], f)
            y = lerp(it["p0"][1], it["p1"][1], f)
            for i in range(1, 6):                                   # trail
                tf = max(0.0, f - i * 0.04)
                tx, ty = lerp(it["p0"][0], it["p1"][0], tf), lerp(it["p0"][1], it["p1"][1], tf)
                pygame.draw.circle(s, it["col"], (int(tx), int(ty)), max(2, 9 - i))
            glow(s, (x, y), 24, it["col"], 160)
            pygame.draw.circle(s, (255, 255, 255), (int(x), int(y)), 7)
        elif k == "beam":
            a = 1 - abs(2 * f - 1) * 0.6
            w = int(18 * a) + 4
            pygame.draw.line(s, it["col"], it["p0"], it["p1"], w)
            pygame.draw.line(s, (255, 255, 255), it["p0"], it["p1"], max(2, w // 3))
            glow(s, it["p1"], 50, it["col"], 160)
        elif k == "burst":
            for i in range(14):
                ang = i / 14 * math.tau + 0.3
                d = 12 + 60 * ease_out(f)
                px, py = it["p"][0] + math.cos(ang) * d, it["p"][1] + math.sin(ang) * d
                pygame.draw.circle(s, it["col"], (int(px), int(py)), max(1, int(7 * (1 - f))))
            pygame.draw.circle(s, (255, 255, 255), (int(it["p"][0]), int(it["p"][1])),
                               int(40 * (1 - f)) + 2, 3)
            glow(s, it["p"], 70, it["col"], int(150 * (1 - f)))
        elif k == "shield":
            rad = AV_R + 14 + 10 * f
            pygame.draw.circle(s, it["col"], (int(it["p"][0]), int(it["p"][1])), int(rad), 5)
            glow(s, it["p"], int(rad) + 20, it["col"], int(160 * (1 - f)))
        elif k == "ring":
            rad = 20 + 160 * ease_out(f)
            pygame.draw.circle(s, it["col"], (int(it["p"][0]), int(it["p"][1])), int(rad), max(2, int(8 * (1 - f))))


def play_reveal(ctx, players_before, outcome, viewer_id, wait_click):
    """Animate the result of one round.

    It shows everybody's action, the shots and their results, then the result banner.  Space or Enter
    jumps to the end; with wait_click it then waits for one more press.
    """
    pos = layout(players_before, viewer_id)
    acts = outcome["actions"]
    names = {p["id"]: p["name"] for p in players_before}
    sched, death, end = build_schedule(pos, acts, outcome)
    after = {p["id"]: p for p in outcome["players"]}
    lines = describe_events(players_before, outcome)
    t_real0 = time.time()
    banner_t = end + 0.15
    total = banner_t + 0.9
    logged = False
    skip = False
    while True:
        ctx.clock.tick(60)
        t = (time.time() - t_real0) * speed()
        for e in pump_events(ctx):
            if e.type == pygame.KEYDOWN and key_of(e) in (pygame.K_SPACE, pygame.K_RETURN):
                skip = True
            if e.type == pygame.MOUSEBUTTONDOWN and t > banner_t:
                skip = True
        if skip and t < total:
            t_real0 -= (total - t) / speed()
            t = total
            skip = False
        if t >= banner_t and not logged:
            ctx.log = ctx.log + [(f"-- Round {ctx.round_no} --", C_GOLD)] + lines
            logged = True
        alive_vis = {pid: (death.get(pid, 99) > t and players_before[i]["alive"])
                     for i, pid in enumerate(p["id"] for p in players_before)}
        bubbles = {}
        for pid, a in acts.items():
            sc = ease_out(t / 0.5)
            tn = names[a["target"]] if a.get("target") is not None else ""
            bubbles[pid] = (a["action"], tn, sc)
        bullets_vis = {pid: (after[pid]["bullets"] if t >= 0.9 else
                             next(p for p in players_before if p["id"] == pid)["bullets"]) for pid in after}
        shake = {pid: 6 for pid, dt in death.items() if 0 <= dt - t < 0.3}
        draw_header(ctx, ctx.round_no, ctx.cfg["label"])
        draw_table(ctx, players_before, pos, alive_vis=alive_vis, bubbles=bubbles, tnow=t,
                   shake=shake, bullets_vis=bullets_vis)
        draw_sched(ctx, sched, t)
        draw_log(ctx)
        if t >= banner_t:
            died = [names[p] for p in outcome["deaths"]]
            msg = elim_text(died) if died else "Nobody died"
            col = C_RED if died else C_GREEN
            a = int(255 * min(1.0, (t - banner_t) / 0.25))
            text(ctx, "big", msg, (W // 2, 580), col, "center", shadow=True, alpha=a)
        if t < 1.4:
            snap_tag = "04_reveal_actions" if t > 0.7 else None
            if snap_tag:
                pygame.display.flip()
                snap(ctx, snap_tag)
        elif t > 1.05 + 0.45 + 0.07:
            pygame.display.flip()
            snap(ctx, "05_reveal_impact")
        if t >= total:
            if wait_click:
                text(ctx, "body", "Click or press Space to continue", (W // 2, 640), C_GOLD, "midtop")
                pygame.display.flip()
                if OPTS.autoplay and t > total + 0.2:
                    return
                for e in pump_events(ctx):
                    if e.type == pygame.MOUSEBUTTONDOWN or (e.type == pygame.KEYDOWN and key_of(e) in (pygame.K_SPACE, pygame.K_RETURN)):
                        return
                continue
            return
        pygame.display.flip()


# ---------------------------------------------------------------------------
#  Match loop
# ---------------------------------------------------------------------------
def run_match(ctx):
    """Play one whole match and return its result and history.

    It creates the bots with their saved memory.  Every round the bots choose, the human(s) choose, the
    round is resolved and animated, and the bots observe the result.  It returns (players, rounds,
    action history).
    """
    cfg = ctx.cfg
    n = cfg["opponents"] + 1
    hot = cfg["hotseat"]
    players = []
    for i in range(n):
        if hot:
            players.append(make_player(i, f"Player {i + 1}", True))
        else:
            players.append(make_player(i, "You" if i == 0 else f"Bot {i}", i == 0))
    bots = {}
    if not hot:
        rng_seed = OPTS.seed
        for p in players[1:]:
            bots[p["id"]] = ThinkingBot(p["id"], cfg["difficulty"],
                                        random.Random(None if rng_seed is None else rng_seed + p["id"]),
                                        samples=None if n <= 3 else max(60, ThinkingBot.LEVELS[cfg["difficulty"]]["samples"] * 3 // 4),
                                        state=ctx.store.load(p["id"]))     # this bot's permanent memory (bot_data_v10/botN_memory.json)
    ctx.log = [("Everyone starts with 0 bullets and one Reflect charge.", C_MUTED),
               ("Reload is risky: anyone who attacks you kills you.", C_MUTED)]
    ctx.insight = {}
    ctx.bot_names = {pid: players[pid]["name"] for pid in bots}      # drives the SHOW P / HIDE P panel
    round_no = 1
    history = []                                   # every round's actions, for the game record / replay
    ctx.in_match = True
    while len(alive_players(players)) > 1:
        ctx.round_no = round_no
        public = [dict(p) for p in players]
        actions = {pid: b.choose(public, round_no) for pid, b in bots.items() if players[pid]["alive"]}
        if bots:
            ctx.next_insight = {pid: list(b.last_probs) for pid, b in bots.items() if pid in actions}
        for h in [p for p in players if p["alive"] and p["human"]]:
            if hot:
                pass_screen(ctx, h["name"], avatar_color(h))
            actions[h["id"]] = human_turn(ctx, h, players, round_no, hotseat=hot)
        outcome = resolve_round(players, actions)
        history.append(outcome["actions"])
        play_reveal(ctx, players, outcome, 0 if not hot else min(p["id"] for p in players if p["alive"]), hot)
        if bots:
            ctx.insight = ctx.next_insight
        for b in bots.values():
            b.observe(public, outcome["actions"])
        players = outcome["players"]
        round_no += 1
    ctx.in_match = False
    ctx.bot_names = {}
    return players, round_no - 1, history


def _wait_next(ctx):
    """Wait for one key press or click at the end of a replayed round."""
    text(ctx, "body", "Click or press Space for the next round", (W // 2, 640), C_GOLD, "midtop")
    t0 = time.time()
    while True:
        ctx.clock.tick(60)
        pygame.display.flip()
        if OPTS.autoplay and time.time() - t0 > 0.2:
            return
        for e in pump_events(ctx):
            if e.type == pygame.MOUSEBUTTONDOWN or (e.type == pygame.KEYDOWN and key_of(e) in (pygame.K_SPACE, pygame.K_RETURN)):
                return


def replay_game(ctx, rec):
    """Replay a recorded game round by round.

    A press during the animation goes straight to the next round; after the animation the round waits
    for one press.  Esc stops the replay.
    """
    n = rec["n"]
    players = [make_player(i, "You" if i == 0 else f"Bot {i}", i == 0) for i in range(n)]
    old_cfg, old_in = ctx.cfg, ctx.in_match
    ctx.cfg = dict(old_cfg)
    ctx.cfg["label"] = f"REPLAY of game #{rec['id']}  |  Space = next round  |  Esc = stop"
    ctx.in_match, ctx.replaying = False, True
    ctx.log = [(f"Replay of game #{rec['id']} ({rec['time']}).", C_GOLD)]
    ctx.insight, ctx.bot_names = {}, {}
    try:
        for r, rd in enumerate(rec["rounds"], 1):
            ctx.round_no = r
            outcome = resolve_round(players, {int(k): v for k, v in rd.items()})
            _, _, end = build_schedule(layout(players, 0), outcome["actions"], outcome)
            full = (end + 0.15 + 0.9) / speed()           # how long play_reveal runs if nobody presses anything
            t0 = time.time()
            play_reveal(ctx, players, outcome, 0, False)   # False = return at the end / right after a press
            if time.time() - t0 >= full - 0.05:            # watched to the end (no press yet): wait for one
                _wait_next(ctx)
            players = outcome["players"]
    except ReplayStop:
        pass
    finally:
        ctx.cfg, ctx.in_match, ctx.replaying = old_cfg, old_in, False


def stats_line(stats):
    """Summarise the recorded games in one line for the setup screen."""
    if not stats["games"]:
        return "No games recorded yet - your first one is about to be saved."
    return (f"Games played: {stats['games']}   |   you won {stats['wins']}   "
            f"bots won {stats['losses']}   draws {stats['draws']}")


def game_over(ctx, players, rounds, rec=None):
    """Show the result screen and return the player's next choice.

    It returns 'replay', 'again', 'menu' or 'quit'.
    """
    live = alive_players(players)
    if len(live) == 1:
        w = live[0]
        msg = f"{w['name']} WON!" if w["name"] != "You" else "YOU WIN!"
        col = C_GREEN if (w["human"]) else C_RED
        if not ctx.cfg["hotseat"] and not w["human"]:
            msg = f"{w['name']} wins - you lose"
    else:
        msg, col = "DRAW - nobody survives", C_GOLD
    live_n = len(players)
    pos = {p["id"]: (W // 2 + (i - (live_n - 1) / 2) * 200, 150) for i, p in enumerate(players)}
    btn_replay = pygame.Rect(W // 2 - 410, 560, 190, 70)
    btn_again = pygame.Rect(W // 2 - 200, 560, 190, 70)
    btn_menu = pygame.Rect(W // 2 + 10, 560, 190, 70)
    btn_quit = pygame.Rect(W // 2 + 220, 560, 190, 70)
    t0 = time.time()
    while True:
        ctx.clock.tick(60)
        t = time.time() - t0
        for e in pump_events(ctx):
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if rec and btn_replay.collidepoint(e.pos):
                    return "replay"
                if btn_again.collidepoint(e.pos):
                    return "again"
                if btn_menu.collidepoint(e.pos):
                    return "menu"
                if btn_quit.collidepoint(e.pos):
                    return "quit"
            if e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
                return "again"
            if e.type == pygame.KEYDOWN and key_of(e) == pygame.K_r and rec:
                return "replay"
        if OPTS.autoplay and t > 0.8 / speed():
            return "again"
        draw_header(ctx, 0, ctx.cfg["label"])
        text(ctx, "med", "MATCH OVER", (24, 22), C_GOLD)
        draw_table(ctx, players, pos, tnow=t)
        sc = ease_out(t / 0.6)
        glow(ctx.screen, (W // 2, 380), int(240 * sc) + 10, col, 70)
        text(ctx, "title", msg, (W // 2, 380), col, "center", shadow=True, alpha=int(255 * sc))
        text(ctx, "body", f"Match lasted {rounds} round{'s' if rounds != 1 else ''}.", (W // 2, 450), C_MUTED, "center")
        if not ctx.cfg["hotseat"]:
            text(ctx, "small", f"Saved as game #{rec['id']} - the bots have learned from it." if rec else
                 "This game could not be saved.", (W // 2, 482), C_MUTED, "center")
            text(ctx, "small", stats_line(ctx.stats), (W // 2, 510), C_GOLD, "center")
        for r, label, c, show in ((btn_replay, "REPLAY (R)", C_PURPLE, bool(rec)), (btn_again, "AGAIN", C_GREEN, True),
                                  (btn_menu, "MENU", C_BLUE, True), (btn_quit, "QUIT", C_RED, True)):
            if not show:
                continue
            rrect(ctx.screen, r, (28, 33, 68), 16)
            rrect(ctx.screen, r, c, 16, 3)
            text(ctx, "med", label, r.center, c, "center")
        pygame.display.flip()
        snap(ctx, "06_game_over")


# ---------------------------------------------------------------------------
#  Tutorial and menu
# ---------------------------------------------------------------------------
TUTORIAL = [                                    # (title, lines): page 1 = title page, page 2 = Instruction
    ("007", ["Multi-round simultaneous decision games, see who's the last to survive."]),
    ("Instruction", None),
]

# page 2: (action, description) in two columns, left column top to bottom, then the right column
INSTRUCTION_COLS = [
    [("reload", "+1 bullet, defenseless"),
     ("defense", "free, defend bullets"),
     ("reflect", "only once, reflect bullets & kill attacker")],
    [("shoot", "-1 bullet, attack another player"),
     ("bomb", "-(living players - 1) bullets, shoot everyone else"),
     ("mg", "-10 bullets, neglect defense and reflect")],
]

# Hover demo on the Instruction page: a tiny looping round played by the real rules engine and the real
# round animation.  (bullets per player, {player: (action, target)}, caption)
DEMO_SCENES = {
    "reload": ([0, 1], {0: ("reload", None), 1: ("defense", None)}, "Reload: +1 bullet, but any attack would kill you"),
    "shoot": ([1, 0], {0: ("shoot", 1), 1: ("reload", None)}, "Hit!  Shoot beats Reload"),
    "defense": ([0, 1], {0: ("defense", None), 1: ("shoot", 0)}, "Blocked!  Defense beats Shoot"),
    "bomb": ([2, 0, 0], {0: ("bomb", None), 1: ("reload", None), 2: ("reload", None)}, "Bomb hits everyone else at once"),
    "reflect": ([0, 1], {0: ("reflect", None), 1: ("shoot", 0)}, "Reflected: the attacker dies"),
    "mg": ([10, 1], {0: ("mg", 1), 1: ("defense", None)}, "MG goes through Defense"),
}
_DEMO_CACHE = {}


def _demo_round(a):
    """Compute the demo round of one action with the real rules, once.

    It returns (players before the round, outcome).
    """
    if a not in _DEMO_CACHE:
        bullets, acts, _ = DEMO_SCENES[a]
        players = [make_player(i, "You" if i == 0 else f"Bot {i}", i == 0) for i in range(len(bullets))]
        for p, b in zip(players, bullets):
            p["bullets"] = b
        out = resolve_round(players, {pid: {"action": k, "target": t} for pid, (k, t) in acts.items()})
        _DEMO_CACHE[a] = (players, out)
    return _DEMO_CACHE[a]


def draw_action_demo(ctx, a, panel, t):
    """Draw the looping demo of one action in a popup panel.

    `t` is the time in seconds since the mouse arrived on the card; the demo loops.
    """
    players, out = _demo_round(a)
    n = len(players)
    cy = panel.y + 118
    if n == 2:
        pos = {0: (panel.x + 110, cy), 1: (panel.right - 110, cy)}
    else:
        pos = {0: (panel.x + 110, cy - 8), 1: (panel.right - 110, cy - 56), 2: (panel.right - 110, cy + 36)}
    sched, death, end = build_schedule(pos, out["actions"], out)
    t = t % (end + 1.3)                                  # loop: the round, then a short pause
    rrect(ctx.screen, panel, (14, 17, 38), 16)
    rrect(ctx.screen, panel, ACT_COLOR[a], 16, 2)
    text(ctx, "med", ACT_NAME[a].title(), (panel.centerx, panel.y + 12), ACT_COLOR[a], "midtop")
    after = {p["id"]: p for p in out["players"]}
    for p in players:
        x, y = pos[p["id"]]
        alive = death.get(p["id"], 99) > t
        col = AVATAR_COLORS[p["id"] % 5] if alive else (70, 72, 84)
        r = 24 if n == 3 else 28
        pygame.draw.circle(ctx.screen, col, (int(x), int(y)), r)
        pygame.draw.circle(ctx.screen, (230, 232, 240) if alive else (110, 112, 124), (int(x), int(y)), r, 2)
        b = after[p["id"]]["bullets"] if t >= 0.9 else p["bullets"]
        text(ctx, "tiny", f"{p['name']}  ({b})", (x, y + r + 4), C_TEXT if alive else C_MUTED, "midtop")
        if t >= 0.15:                                    # action bubble beside the player
            k = out["actions"][p["id"]]["action"]
            bx = x + (r + 34) * (1 if p["id"] == 0 else -1)
            draw_icon(ctx.screen, k, int(bx), int(y), 13, ACT_COLOR[k])
    draw_sched(ctx, sched, t)
    if t >= end - 0.2:
        text(ctx, "body", DEMO_SCENES[a][2], (panel.centerx, panel.bottom - 36), C_GOLD, "midtop")


ACTION_HELP = {
    "reload": "+1 bullet. You are defenseless while reloading.",
    "shoot": "1 bullet. Hit one chosen player.",
    "defense": "Free, unlimited. Blocks Shoot and Bomb. Useless against MG.",
    "bomb": "(living players - 1) bullets. Shoots everyone else at once.",
    "reflect": "Once per game. Survive Shoot/Bomb, kill the attackers.",
    "mg": "10 bullets. Unstoppable. MG vs MG: both die.",
}


# ---------------------------------------------------------------------------
#  Title page background: forest_bg_v10.2.png (pine forest at night, white tree contours).
#  Optional: if the file is missing, the title page uses the plain background.
# ---------------------------------------------------------------------------
FOREST_FILE = os.path.join(SCRIPT_DIR, "forest_bg_v10.2.png")
_FOREST = None


def forest_bg():
    """Return the title page background image, loading it on first use.

    It returns None when the image file is missing or cannot be read.
    """
    global _FOREST
    if _FOREST is None:
        try:
            img = pygame.image.load(FOREST_FILE).convert()
            _FOREST = img if img.get_size() == (W, H) else pygame.transform.smoothscale(img, (W, H))
        except (pygame.error, FileNotFoundError, OSError):
            _FOREST = False
    return _FOREST or None


def tutorial(ctx):
    """Show the two introduction pages."""
    page = 0
    btn_next = pygame.Rect(W // 2 + 20, H - 100, 260, 64)
    btn_back = pygame.Rect(W // 2 - 280, H - 100, 260, 64)
    btn_skip = pygame.Rect(W - 170, 20, 150, 44)
    t0 = time.time()
    last = time.time()
    demo_a, demo_t0 = None, 0.0                       # hover demo on the Instruction page
    while True:
        ctx.clock.tick(60)
        for e in pump_events(ctx):
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if btn_skip.collidepoint(e.pos):
                    return
                if TUTORIAL[page][0] == "Instruction" and ctx.secret_rect and ctx.secret_rect.collidepoint(e.pos):
                    ctx.secret_found = True
                if btn_next.collidepoint(e.pos):
                    if page == len(TUTORIAL) - 1:
                        return
                    page += 1
                if btn_back.collidepoint(e.pos) and page > 0:
                    page -= 1
            if e.type == pygame.KEYDOWN and key_of(e) in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_RIGHT):
                if page == len(TUTORIAL) - 1:
                    return
                page += 1
        if OPTS.autoplay and time.time() - last > 0.25 / speed():
            last = time.time()
            snap(ctx, f"00_tutorial_{page}")
            if page == len(TUTORIAL) - 1:
                return
            page += 1
        title, lines = TUTORIAL[page]
        bg = forest_bg() if page == 0 else None
        if bg is not None:
            ctx.screen.blit(bg, (0, 0))                  # forest behind the title page
        else:
            ctx.screen.fill(C_BG)
        text(ctx, "title" if page == 0 else "big", title, (W // 2, 18 if page else 80), C_GOLD, "midtop", shadow=True)
        if page:                                     # subtitle of the Instruction page
            text(ctx, "body", "Put your mouse on each action to see the animation", (W // 2, 76), C_MUTED, "midtop")
        ctx.secret_rect = None
        if lines:                                    # page 1: tagline + one sentence
            text(ctx, "big", "How's your luck? Wanna Gamble?", (W // 2, 220), C_RED, "midtop", shadow=True)
            y = 310
            for ln in lines:
                for wl in wrap(ctx, "body", ln, 980):
                    text(ctx, "body", wl, (W // 2, y), C_TEXT, "midtop", shadow=True)
                    y += 32
        else:                                        # page 2: the 6 actions, key in front of every name
            cards = {}
            for c, col in enumerate(INSTRUCTION_COLS):
                for i, (a, desc) in enumerate(col):
                    r = pygame.Rect(50 + c * 600, 125 + i * 118, 580, 104)
                    cards[a] = r
                    rrect(ctx.screen, r, C_PANEL, 14)
                    rrect(ctx.screen, r, (60, 66, 110), 14, 2)
                    draw_icon(ctx.screen, a, r.x + 46, r.centery, 24, ACT_COLOR[a])
                    key = pygame.Rect(r.x + 92, r.y + 16, 34, 34)
                    rrect(ctx.screen, key, (28, 33, 68), 8)
                    rrect(ctx.screen, key, ACT_COLOR[a], 8, 2)
                    text(ctx, "body", ACT_KEY[a], key.center, ACT_COLOR[a], "center")
                    text(ctx, "med", ACT_NAME[a].title(), (key.right + 12, key.centery), ACT_COLOR[a], "midleft")
                    for j, wl in enumerate(wrap(ctx, "body", desc, r.right - (r.x + 92) - 16)):
                        text(ctx, "body", wl, (r.x + 92, r.y + 58 + j * 26), C_TEXT)
            text(ctx, "body", "Note: bullet cancellation!      MG-MG: TIE!", (W // 2, 488), C_GOLD, "midtop")
            ctx.secret_rect = pygame.Rect(W // 2 - 300, 524, 600, 36)     # hidden note: appears when clicked
            text(ctx, "body", "Press here ->", (W // 2 - 150, 528), C_MUTED, "topright")   # points at the hidden note
            if ctx.secret_found:
                text(ctx, "body", "Machine gun vs. Shoot: TIE!", (W // 2, 528), C_GOLD, "midtop")
            mx, my = pygame.mouse.get_pos()                # hover demo: the card under the mouse
            hov = next((a for a, r in cards.items() if r.collidepoint(mx, my)), None)
            if hov != demo_a:
                demo_a, demo_t0 = hov, time.time()
            if demo_a:
                r = cards[demo_a]
                panel = pygame.Rect(0, 0, 520, 250)
                panel.top = min(max(r.y - 20, 100), H - 120 - 250)
                panel.left = 50 + 600 + 30 if r.x < W // 2 else 50 + 30   # over the other column
                draw_action_demo(ctx, demo_a, panel, time.time() - demo_t0)
        text(ctx, "tiny", f"{page + 1} / {len(TUTORIAL)}", (W // 2, H - 24), C_MUTED, "midtop")
        for r, label, c, show in ((btn_next, "Instruction" if page == 0 else "Next", C_GOLD, True),
                                  (btn_back, "BACK", C_BLUE, page > 0), (btn_skip, "SKIP", C_MUTED, True)):
            if not show:
                continue
            rrect(ctx.screen, r, (28, 33, 68), 16)
            rrect(ctx.screen, r, c, 16, 3)
            text(ctx, "med", label, r.center, c, "center")
        pygame.display.flip()


def menu(ctx, cfg):
    """Show the game setup screen and return the chosen settings."""
    cfg["opponents"] = 4                              # Game Setup always opens on 1 vs 4 Bots
    cards = [(1, "1 vs 1 Bot", "1H1B - a duel against the solved-game bot"),
             (2, "1 vs 2 Bots", "1H2B - three-way survival"),
             (4, "1 vs 4 Bots", "1H4B - five-player free-for-all")]
    card_rects = [pygame.Rect(70 + i * 380, 130, 360, 150) for i in range(3)]
    diffs = ["easy", "normal", "hard"]
    diff_rects = [pygame.Rect(70 + i * 170, 360, 150, 54) for i in range(3)]
    timers = [3, 5, 10]
    timer_rects = [pygame.Rect(720 + i * 130, 360, 110, 54) for i in range(3)]
    hot_rect = pygame.Rect(70, 480, 230, 56)          # Testing mode button; its colour shows ON (red) / OFF
    start_rect = pygame.Rect(W // 2 - 140, 600, 280, 76)
    help_rect = pygame.Rect(70, 610, 200, 56)
    diff_desc = {"easy": "Makes deliberate mistakes. A good first opponent.",
                 "normal": "Plays near-optimal and slowly learns your habits.",
                 "hard": "Near-optimal and ruthlessly exploits your habits."}
    t0 = time.time()
    while True:
        ctx.clock.tick(60)
        for e in pump_events(ctx):
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for r, (n, _, _) in zip(card_rects, cards):
                    if r.collidepoint(e.pos):
                        cfg["opponents"] = n
                for r, d in zip(diff_rects, diffs):
                    if r.collidepoint(e.pos):
                        cfg["difficulty"] = d
                for r, tm in zip(timer_rects, timers):
                    if r.collidepoint(e.pos):
                        cfg["timer"] = tm
                if hot_rect.collidepoint(e.pos):
                    cfg["hotseat"] = not cfg["hotseat"]
                if help_rect.collidepoint(e.pos):
                    tutorial(ctx)
                if start_rect.collidepoint(e.pos):
                    return cfg
            if e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
                return cfg
        if OPTS.autoplay and time.time() - t0 > 0.3 / speed():
            snap(ctx, "01_menu")
            return cfg
        ctx.screen.fill(C_BG)
        text(ctx, "big", "GAME SETUP", (W // 2, 40), C_GOLD, "midtop", shadow=True)
        for r, (n, name, desc) in zip(card_rects, cards):
            sel = cfg["opponents"] == n
            rrect(ctx.screen, r, C_PANEL2 if sel else C_PANEL, 18)
            rrect(ctx.screen, r, C_GOLD if sel else (60, 66, 110), 18, 4 if sel else 2)
            for i in range(n + 1):
                cx = r.centerx + (i - n / 2) * 46
                pygame.draw.circle(ctx.screen, AVATAR_COLORS[i % 5] if i else (70, 175, 255), (int(cx), r.y + 44), 17)
            text(ctx, "med", name, (r.centerx, r.y + 78), C_TEXT, "midtop")
            for j, ln in enumerate(wrap(ctx, "tiny", desc, r.w - 30)):
                text(ctx, "tiny", ln, (r.centerx, r.y + 112 + j * 18), C_MUTED, "midtop")
        text(ctx, "body", "Bot difficulty", (70, 322), C_MUTED)
        for r, d in zip(diff_rects, diffs):
            sel = cfg["difficulty"] == d
            rrect(ctx.screen, r, C_PANEL2 if sel else C_PANEL, 14)
            rrect(ctx.screen, r, C_GREEN if sel else (60, 66, 110), 14, 3 if sel else 2)
            text(ctx, "body", d.upper(), r.center, C_TEXT, "center")
        text(ctx, "small", diff_desc[cfg["difficulty"]], (70, 428), C_MUTED)
        text(ctx, "body", "Round timer", (720, 322), C_MUTED)
        for r, tm in zip(timer_rects, timers):
            sel = cfg["timer"] == tm
            rrect(ctx.screen, r, C_PANEL2 if sel else C_PANEL, 14)
            rrect(ctx.screen, r, C_GREEN if sel else (60, 66, 110), 14, 3 if sel else 2)
            text(ctx, "body", f"{tm} s", r.center, C_TEXT, "center")
        on = cfg["hotseat"]
        rrect(ctx.screen, hot_rect, (90, 36, 46) if on else C_PANEL, 14)
        rrect(ctx.screen, hot_rect, C_RED if on else (60, 66, 110), 14, 3)
        text(ctx, "body", "Testing mode", hot_rect.center, C_TEXT, "center")      # red = ON
        text(ctx, "bodyb", "press SPACE to skip the animation of the round", (hot_rect.right + 28, hot_rect.centery),
             C_TEXT, "midleft")
        text(ctx, "body", stats_line(ctx.stats), (W // 2, 552), C_GOLD, "midtop")
        rrect(ctx.screen, help_rect, C_PANEL, 14)
        rrect(ctx.screen, help_rect, C_BLUE, 14, 2)
        text(ctx, "body", "HOW TO PLAY", help_rect.center, C_BLUE, "center")
        rrect(ctx.screen, start_rect, (40, 60, 40), 18)
        rrect(ctx.screen, start_rect, C_GREEN, 18, 4)
        text(ctx, "big", "START", start_rect.center, C_GREEN, "center")
        pygame.display.flip()


# ---------------------------------------------------------------------------
#  Entry points
# ---------------------------------------------------------------------------
def parse_args(argv):
    """Read the command-line options."""
    ap = argparse.ArgumentParser(description="007 - only one shall stand")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tournament", type=int, metavar="GAMES")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--autoplay", action="store_true", help="scripted random 'human' (for testing)")
    ap.add_argument("--headless", action="store_true", help="use SDL's dummy video driver")
    ap.add_argument("--shots", metavar="DIR", help="save screenshots of key screens")
    ap.add_argument("--games", type=int, default=0, help="exit after this many games (testing)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mode", type=int, choices=(1, 2, 4), default=4, help="number of opponents")
    ap.add_argument("--difficulty", choices=("easy", "normal", "hard"), default="normal")
    ap.add_argument("--timer", type=int, choices=(3, 5, 10), default=10)
    ap.add_argument("--hotseat", action="store_true")
    ap.add_argument("--skip-tutorial", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--data-dir", metavar="DIR", help="where bot memory + game records live (default: bot_data_v10 next to this file)")
    return ap.parse_args(argv)


# The options in force.  Until main() reads the real command line these are the defaults, taken from
# parse_args() itself so that there is only one list of options and default values.
OPTS = parse_args([])


def retune_after_game(store):
    """Re-fit the tuned variables on all recorded games with 007-train-bot.py.

    Each call moves the values by at most one small step, so a single unusual game cannot throw them
    around.
    """
    path = os.path.join(SCRIPT_DIR, "007-train-bot.py")
    if not os.path.exists(path):
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("train007", path)
    tb = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(tb)
    finally:
        sys.dont_write_bytecode = old
    tb.update_params(sys.modules[__name__], store, store.records(), max_step=1)


_RETUNE = {"thread": None, "done": False}


def retune_in_background(store_dir):
    """Start the after-game re-tuning in a background thread.

    The re-tuning re-scores every recorded game and takes several seconds, which would otherwise delay
    the game-over screen.  The thread works on a separate copy of this module, because the tuning
    temporarily changes module variables and a game started meanwhile must not be disturbed.  The new
    values are applied when the next game starts.  It is not a daemon thread, so quitting waits until
    the result is saved.
    """
    import importlib.util
    import threading
    old = _RETUNE["thread"]
    if old is not None and old.is_alive():
        old.join()

    def work():
        """Do the re-tuning on a separate copy of this module."""
        try:
            name = "game007_retune"
            spec = importlib.util.spec_from_file_location(name, os.path.abspath(__file__))
            g = importlib.util.module_from_spec(spec)
            sys.modules[name] = g
            spec.loader.exec_module(g)
            g.retune_after_game(g.BotStore(store_dir))
            _RETUNE["done"] = True
        except Exception as err:                         # tuning must never break the game
            print(f"[007] parameter update skipped: {err}", file=sys.stderr)

    _RETUNE["thread"] = threading.Thread(target=work, name="retune", daemon=False)
    _RETUNE["thread"].start()


def main(argv=None):
    """Start the game, or one of the headless tools."""
    global OPTS
    args = parse_args(argv if argv is not None else sys.argv[1:])
    OPTS = args
    if args.seed is not None:
        random.seed(args.seed)
    if args.selftest:
        return run_selftest()
    if args.tournament:
        return run_tournament(args.tournament)
    if args.solve:
        return run_solve()
    if pygame is None:
        print("pygame is required to play:  pip install pygame")
        return 1
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("007 - Only one shall stand")
    try:
        pygame.key.stop_text_input()                # keep a Chinese / other input method from swallowing the hotkeys
    except Exception:
        pass
    ctx = Ctx(screen, load_fonts(), pygame.time.Clock())
    ctx.store = BotStore(args.data_dir)
    apply_params(ctx.store.load_params())              # tuned variables from the last training / game
    ctx.stats = BotStore.summary(ctx.store.records())
    cfg = {"opponents": args.mode, "difficulty": args.difficulty, "timer": args.timer,
           "hotseat": args.hotseat}
    games = 0
    try:
        get_v2p()                                   # make sure the solved table is ready
        if not args.skip_tutorial:
            tutorial(ctx)
        while True:
            cfg = menu(ctx, cfg)
            while True:
                ctx.cfg = dict(cfg)
                ctx.cfg["label"] = (f"{'Testing mode' if cfg['hotseat'] else '1 vs ' + str(cfg['opponents']) + ' Bot' + ('s' if cfg['opponents'] > 1 else '')}"
                                    f"  |  {cfg['difficulty'].title()}  |  {cfg['timer']}s"
                                    + ("" if cfg["hotseat"] else f"  |  game #{ctx.stats['games'] + 1}")
                                    + "  |  Space = skip")
                if _RETUNE["done"]:                          # new values from the background retune
                    _RETUNE["done"] = False
                    apply_params(ctx.store.load_params())
                try:
                    players, rounds, history = run_match(ctx)
                except ReturnToMenu:
                    ctx.in_match = False
                    ctx.bot_names = {}
                    break
                games += 1
                rec = None
                if not cfg["hotseat"]:                       # record it and let every bot that played learn from it
                    live = alive_players(players)
                    rec = make_record(ctx.store.next_id(), cfg, len(players), history, live[0]["id"] if len(live) == 1 else None)
                    try:
                        ctx.store.add_game(rec)
                    except OSError as err:
                        print(f"[007] could not save the game record / bot memory: {err}", file=sys.stderr)
                        rec = None
                    retune_in_background(ctx.store.dir)  # takes seconds: runs in the background
                    ctx.stats = BotStore.summary(ctx.store.records())
                if args.games and games >= args.games:
                    return 0
                res = game_over(ctx, players, rounds, rec)
                while res == "replay":
                    replay_game(ctx, rec)
                    res = game_over(ctx, players, rounds, rec)
                if res == "quit":
                    return 0
                if res == "menu":
                    break
    except QuitGame:
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
