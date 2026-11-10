#!/usr/bin/env python3
"""
007 - bot trainer (for 007-premode-pygame-v10.py)
=================================================
The game saves two things in bot_data_v10/ :
    game_records.jsonl        every game you played (one line per game, all actions of all players)
    bot1_memory.json ...      bot4_memory.json: what each bot has learned
        bot 1 learns from 2-, 3- and 5-player games, bot 2 from 3- and 5-player games,
        bots 3 and 4 from 5-player games only.

This script (re)builds those bot files from the records and can practise the bots against a
simulated copy of you.  The game loads bot1..bot4_memory.json every time it starts a match.

    python 007-train-bot.py                      rebuild all four bot files from the game records
    python 007-train-bot.py --status             show what the bots know (and how YOU defend)
    python 007-train-bot.py --selfplay 200       rebuild, then practise 200 games per mode against a clone of you
    python 007-train-bot.py --eval 100           measure: bots with memory vs. bots without, against the clone
    options: --data-dir DIR   --difficulty easy|normal|hard   --modes 2,3,5   --seed N

What "training" means here (honest version)
  1. REBUILD (needs only your recorded games): replays every game with the real rules engine and
     (a) counts what you do in each situation (bullets, last move, how many rounds you have been
         defending) -> the human model each bot starts a new game with;
     (b) records, per situation and per move, whether the bot finished better or worse than the fair
         share after choosing it.  The bot nudges its probabilities by at most +-35 % in that direction.
  2. SELF-PLAY (optional): the bots play against a CLONE that samples moves from the model of you.
     This only exercises (b); it cannot teach anything you have not shown the model.  Simulated
     games count 30 % of a real game.  Treat it as a sanity check, not as new knowledge.
  3. EVAL: win rate against the clone with and without the stored memory.  Because the clone IS the
     model, a good number here shows the bot uses the model - not that it beats you.
More recorded games = better bots.  Run the game and this script one after the other, not at the same time.
"""
import argparse
import importlib.util
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_WEIGHT = 0.3
MAX_ROUNDS = 120


def load_game():
    sys.dont_write_bytecode = True
    path = os.path.join(HERE, "007-premode-pygame-v10.py")
    spec = importlib.util.spec_from_file_location("game007", path)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def rebuild(g, store, records):
    states = {s: g.new_bot_state(s) for s in range(1, g.MAX_SLOT + 1)}
    for rec in records:
        for st in states.values():
            g.apply_record(st, rec)
    for st in states.values():
        store.save(st)
    return states


def status(g, store, records):
    sm = g.BotStore.summary(records)
    print(f"data dir : {store.dir}")
    print(f"games    : {sm['games']} recorded  (you won {sm['wins']}, bots won {sm['losses']}, draws {sm['draws']})")
    for n, (w, l, d) in sorted(sm["by_mode"].items()):
        print(f"           {n}-player: you {w} - bots {l} - draws {d}")
    for slot in range(1, g.MAX_SLOT + 1):
        st = store.load(slot)
        obs = sum(st["human"].c0.values())
        print(f"bot {slot}    : {st['games']:>3} games ({'/'.join(map(str, st['modes']))}-player)   "
              f"bot won {st['wins']}, lost {st['losses']}, drew {st['draws']}   "
              f"observed {obs:.0f} of your moves, {sum(len(a) for a in st['learn'].t.values())} learned (situation, move) cells")
    hm = store.load(1)["human"]
    print("\nHow you defend (bot 1's model of you)  -  P(you defend AGAIN | defended d rounds in a row):")
    print("   d    prior   observed   rounds seen")
    for d in range(1, 5):
        p, n = hm.streak_def_prob(d)
        print(f"   {d}{'+' if d == 4 else ' '}   {g.STREAK_PRIOR[d]:>5.0%}   {p:>7.0%}   {n:>8.0f}")
    print("   (observed = prior blended with your real moves, prior worth "
          f"{g.STREAK_PRIOR_WEIGHT:.0f} rounds)")
    p = store.load_params()
    print(f"\nTuned variables in use: STREAK_PRIOR_WEIGHT = {p['STREAK_PRIOR_WEIGHT']:.0f}, "
          f"PATTERN_ROUNDS = {p['PATTERN_ROUNDS']:.0f}   ({len(p.get('history', []))} updates; file {store.params_path})")
    if p.get("history"):
        h = p["history"][-1]
        print(f"   last update {h['time']} on {h['games']} games: {h.get('moves', 0)} of your moves scored"
              + (f", log-loss {h['pred_loss']:.3f} vs {h['uniform_loss']:.3f} for a uniform guess" if h.get("pred_loss") else
                 f" (fewer than {MIN_MOVES}: values unchanged)"))


# ---------------------------------------------------------------------------
#  Tuning of the variables on YOUR recorded games only (no simulated games)
#     STREAK_PRIOR_WEIGHT : how many observations the 50/33/15/5 % prior is worth
#                           -> picked by how well the model predicts your next move
#     PATTERN_ROUNDS      : how strongly the bot exploits the streak pattern
#                           -> picked by the bot's expected payoff against your real 1-v-1 moves
#  Both scores are "prequential": game k is judged with a model built from games 1..k-1 only
#  (plus the rounds already played in game k), exactly as the game uses its memory.
# ---------------------------------------------------------------------------
GRID_W = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
GRID_R = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
MIN_MOVES = 100          # your moves needed before the prediction score may change STREAK_PRIOR_WEIGHT
MIN_DUELS = 50           # 1-v-1 decisions needed before the payoff score may change PATTERN_ROUNDS
MIN_GAIN_LOSS = 0.0005   # log-loss improvement needed to move away from the current value
MIN_GAIN_PAYOFF = 0.002  # payoff improvement needed to move away from the current value


def predict_human(g, mem, bullets, legal):
    """What the bot's model of you says about your next move (same recipe as CloneBot)."""
    emp, n = mem.dist(bullets, legal, merge_bomb=False)
    p = dict(emp) if emp else {a: 1.0 for a in legal}
    s = sum(p.values())
    p = {a: w / s for a, w in p.items()}
    p, _ = g.apply_streak(p, mem.streak, mem, g.human_rest(legal, emp, n))
    return p


def score_prediction(g, records, w):
    """Mean log-loss (lower = better) of the predicted next move of the human; also the uniform guess for reference."""
    old = g.STREAK_PRIOR_WEIGHT
    g.STREAK_PRIOR_WEIGHT = w
    try:
        st = g.new_bot_state(1)
        loss = uni = 0.0
        cnt = 0
        for rec in records:
            mem = g.OppMemory()
            mem.attach_base(st["human"])
            players = [g.make_player(i, f"P{i}", i == 0) for i in range(rec["n"])]
            for rd in rec["rounds"]:
                acts = {int(k): v for k, v in rd.items()}
                h = players[0]
                if h["alive"] and 0 in acts:
                    legal = g.legal_actions(h, players)
                    a = acts[0]["action"]
                    p = predict_human(g, mem, h["bullets"], legal)
                    if a in p:
                        loss -= math.log(max(p[a], 1e-4))
                        uni += math.log(len(legal))
                        cnt += 1
                    mem.update(h["bullets"], a, acts[0].get("target"))
                players = g.resolve_round(players, acts)["players"]
            g.apply_record(st, rec)
        return (loss / cnt, uni / cnt, cnt) if cnt else (None, None, 0)
    finally:
        g.STREAK_PRIOR_WEIGHT = old


def score_payoff(g, records, pr):
    """Mean payoff (higher = better) of the Hard bot's mix against your REAL move in every 1-v-1 decision."""
    old = g.ThinkingBot.PATTERN_ROUNDS
    g.ThinkingBot.PATTERN_ROUNDS = pr
    try:
        states = {s: g.new_bot_state(s) for s in range(1, g.MAX_SLOT + 1)}
        tot, cnt = 0.0, 0
        for rec in records:
            n = rec["n"]
            bots = {i: g.ThinkingBot(i, "hard", random.Random(1), state=states[i]) for i in range(1, n)}
            players = [g.make_player(i, f"P{i}", i == 0) for i in range(n)]
            for rnd, rd in enumerate(rec["rounds"], 1):
                acts = {int(k): v for k, v in rd.items()}
                pub = [dict(p) for p in players]
                live = [p for p in players if p["alive"]]
                if len(live) == 2 and players[0]["alive"] and 0 in acts:
                    pid = next(p["id"] for p in live if p["id"] != 0)
                    b = bots[pid]
                    b.choose(pub, rnd)
                    mix = dict(b.last_probs)
                    la, lo, _, _, _, A = g.duel_solution(pub[pid], pub[0], b.my_streak)
                    ha = acts[0]["action"]
                    ha = "shoot" if ha == "bomb" else ha
                    if ha in lo:
                        j = lo.index(ha)
                        tot += sum(mix.get(a, 0.0) * A[i][j] for i, a in enumerate(la))
                        cnt += 1
                out = g.resolve_round(players, acts)
                for b in bots.values():
                    b.observe(pub, out["actions"])
                players = out["players"]
            for s in range(1, n):
                g.apply_record(states[s], rec)
        return (tot / cnt, cnt) if cnt else (None, 0)
    finally:
        g.ThinkingBot.PATTERN_ROUNDS = old


def _move(grid, cur, best, max_step):
    """Jump to `best`, or (max_step given) move at most that many grid steps from `cur` towards it."""
    ic = min(range(len(grid)), key=lambda i: abs(grid[i] - cur))
    ib = grid.index(best)
    if max_step is not None:
        ib = ic + max(-max_step, min(max_step, ib - ic))
    return grid[ib]


def update_params(g, store, records, max_step=None, verbose=False):
    """
    Re-fit STREAK_PRIOR_WEIGHT and PATTERN_ROUNDS on all recorded games and save them to
    bot_data_v10/tuned_params.json.  The game calls this after every game with max_step=1 (one grid step
    at most, so a single odd game cannot throw the values around); the trainer calls it with a full jump.
    Too little data (see MIN_MOVES / MIN_DUELS) or no clear gain = the value stays as it is.
    """
    params = store.load_params()
    cur_w, cur_r = params["STREAK_PRIOR_WEIGHT"], params["PATTERN_ROUNDS"]
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "games": len(records)}
    new_w, new_r = cur_w, cur_r
    gw = sorted(set(GRID_W + [cur_w]))
    sc = {w: score_prediction(g, records, w) for w in gw}
    moves = sc[cur_w][2]
    entry.update(moves=moves, uniform_loss=sc[cur_w][1])
    if moves >= MIN_MOVES:
        best = min(gw, key=lambda w: sc[w][0])
        if sc[cur_w][0] - sc[best][0] >= MIN_GAIN_LOSS:
            new_w = _move(gw, cur_w, best, max_step)
        entry["pred_loss"] = sc[new_w][0]
        if verbose:
            print("  STREAK_PRIOR_WEIGHT   log-loss of your next move (lower = better; uniform guess "
                  f"{sc[cur_w][1]:.3f}, {moves} moves)")
            for w in gw:
                print(f"     {w:>5.0f}   {sc[w][0]:.4f}{'   <- best' if w == best else ''}{'   <- in use' if w == cur_w else ''}")
    g.STREAK_PRIOR_WEIGHT = new_w
    try:
        gr = sorted(set(GRID_R + [cur_r]))
        ps = {r: score_payoff(g, records, r) for r in gr}
        duels = ps[cur_r][1]
        entry["duels"] = duels
        if duels >= MIN_DUELS:
            best = max(gr, key=lambda r: ps[r][0])
            if ps[best][0] - ps[cur_r][0] >= MIN_GAIN_PAYOFF:
                new_r = _move(gr, cur_r, best, max_step)
            entry["payoff"] = ps[new_r][0]
            if verbose:
                print(f"  PATTERN_ROUNDS        bot payoff against your real 1-v-1 moves (higher = better, {duels} decisions)")
                for r in gr:
                    print(f"     {r:>5.0f}   {ps[r][0]:+.4f}{'   <- best' if r == best else ''}{'   <- in use' if r == cur_r else ''}")
    finally:
        g.STREAK_PRIOR_WEIGHT = cur_w
    entry.update(STREAK_PRIOR_WEIGHT=new_w, PATTERN_ROUNDS=new_r)
    params.update(STREAK_PRIOR_WEIGHT=new_w, PATTERN_ROUNDS=new_r)
    params["history"] = (params.get("history", []) + [entry])[-50:]
    store.save_params(params)
    return params


def play_one(g, states, n, difficulty, rng, samples, use_memory=True):
    """One simulated game: seat 0 = clone of the human, seats 1.. = trained bots.  Returns (record, winner, stalled)."""
    model = states[1]["human"]
    players = [g.make_player(i, "You" if i == 0 else f"Bot {i}", i == 0) for i in range(n)]
    bots = {0: g.CloneBot(0, model, random.Random(rng.random()))}
    for i in range(1, n):
        bots[i] = g.ThinkingBot(i, difficulty, random.Random(rng.random()), samples=samples,
                                state=states[i] if use_memory else None)
    history = []
    for rnd in range(1, MAX_ROUNDS + 1):
        if len(g.alive_players(players)) <= 1:
            break
        pub = [dict(p) for p in players]
        acts = {p["id"]: bots[p["id"]].choose(pub, rnd) for p in players if p["alive"]}
        out = g.resolve_round(players, acts)
        for b in bots.values():
            b.observe(pub, out["actions"])
        history.append(out["actions"])
        players = out["players"]
    live = g.alive_players(players)
    stalled = len(live) > 1
    winner = live[0]["id"] if len(live) == 1 else None
    rec = g.make_record(0, {"difficulty": difficulty}, n, history, winner)
    return rec, winner, stalled


def selfplay(g, store, states, games, modes, difficulty, rng, samples):
    for n in modes:
        t0, done, stalled, bot_wins = time.time(), 0, 0, 0
        for _ in range(games):
            rec, winner, st = play_one(g, states, n, difficulty, rng, samples)
            if st:
                stalled += 1                      # nobody could finish: nothing to learn from
                continue
            for slot in range(1, n):
                g.apply_record(states[slot], rec, weight=SIM_WEIGHT, sim=True)
            done += 1
            bot_wins += winner not in (None, 0)
        print(f"  {n}-player self-play: {done} games used ({stalled} stalled and skipped), bots won {bot_wins}/{max(done, 1)}"
              f"   ({time.time() - t0:.0f}s)")
    for st in states.values():
        store.save(st)


def evaluate(g, states, modes, difficulty, games, rng, samples):
    print(f"\nEvaluation against the clone of you ({games} games per line, difficulty {difficulty})")
    print("  mode      bots WITHOUT memory        bots WITH stored memory")
    for n in modes:
        row = []
        for use in (False, True):
            bw = hw = dr = sl = 0
            for _ in range(games):
                _, winner, stalled = play_one(g, states, n, difficulty, rng, samples, use_memory=use)
                if stalled:
                    sl += 1
                elif winner is None:
                    dr += 1
                elif winner == 0:
                    hw += 1
                else:
                    bw += 1
            row.append(f"bots {bw:>3} | clone {hw:>3} | draw {dr:>2} | stall {sl:>2}")
        print(f"  {n}-player   {row[0]}     {row[1]}")


def main():
    ap = argparse.ArgumentParser(description="train / inspect the 007 bots")
    ap.add_argument("--data-dir")
    ap.add_argument("--status", action="store_true", help="only show what the bots know")
    ap.add_argument("--selfplay", type=int, default=0, metavar="GAMES", help="practice games per mode against the clone")
    ap.add_argument("--eval", type=int, default=0, metavar="GAMES", help="evaluation games per mode and per brain")
    ap.add_argument("--difficulty", choices=("easy", "normal", "hard"), default="hard")
    ap.add_argument("--modes", default="2,3", help="player counts to simulate, e.g. 2,3,5 (5 is slow)")
    ap.add_argument("--samples", type=int, default=40, help="look-ahead samples for 3+ player simulation (speed)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    g = load_game()
    store = g.BotStore(args.data_dir)
    g.apply_params(store.load_params())
    records = store.records()
    rng = random.Random(args.seed)
    modes = [int(m) for m in args.modes.split(",") if m.strip()]
    if args.status:
        status(g, store, records)
        return 0
    print(f"Rebuilding the 4 bot files from {len(records)} recorded game(s) ...")
    states = rebuild(g, store, records)
    if records:
        print("Tuning on your recorded games (nothing simulated):")
        g.apply_params(update_params(g, store, records, verbose=True))
    if args.selfplay:
        print("Self-play against a clone of you:")
        selfplay(g, store, states, args.selfplay, modes, args.difficulty, rng, args.samples)
    print("Saved:")
    for slot in range(1, g.MAX_SLOT + 1):
        print(f"  {store.memory_path(slot)}")
    if args.eval:
        evaluate(g, states, modes, args.difficulty, args.eval, rng, args.samples)
    print()
    status(g, store, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
