# 007 v10.2 — website version

The same game as `007-premode-pygame-v10.2.py`, rewritten in JavaScript so it runs in any browser:
no installation, no Python download, no server. Open `index.html` (double-click it) to play.

## How it works

Everything runs in the visitor's browser:

```
index.html  ->  js/core.js   rules, solved 2-player game, opponent model, bot, learning, re-tuning
                js/ui.js     the screens (canvas, 1280 x 720), same layout and controls as the pygame game
                js/v2p.js    the 676 solved values of the 2-player game, exported from the Python game
```

- The bot calculates in the browser (a Hard decision in a 5-player game takes a few milliseconds).
- Game records and the bots' memory are saved in the browser's `localStorage`, so the bots learn each
  visitor's habits over their games. After every game the re-tuning runs too, as in the desktop game.
- Keys are read by their physical position, so an input method (for example Chinese) cannot swallow them.
- The only difference: QUIT on the game-over screen returns to the title page (a web page cannot close itself).

## Same results as the Python game

`js/core.js` is a line-by-line port of the Python game logic. The tests prove it computes the same:

```
python tools/export_from_python.py     # writes js/v2p.js and the expected results (tests/cases/)
node tests/run_tests.js                # compares the JavaScript core with them
```

They compare the rules engine (random rounds and every round of recorded games), the solved 2-player mixes
for every position, the position scores, every bot decision in recorded games (probabilities and the chosen
move, with the same test random generator in both languages), the learned memory and the re-tuning.
Numbers must agree to 1e-9.

`tests/cases/` contains real game records from `bot_data_v10`: keep it private (it is in `.gitignore`).

## Test options

Add them to the address, e.g. `index.html?autoplay=1&fast=1`: `autoplay=1` (a scripted player),
`fast=1`, `skiptutorial=1`, `storage=memory` (save nothing), `mode=1|2|4`, `difficulty=easy|normal|hard`,
`timer=3|5|10`, `seed=N`, `page=2`, `mouse=x,y`, `timerframes=1` and `hold=gameover` (for headless screenshots).
