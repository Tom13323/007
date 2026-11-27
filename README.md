# 007: Only one shall stand
Author: Tom13323

## ▶ [Play it in your browser: tom13323.github.io/007](https://tom13323.github.io/007/)

No installation: the website version runs the same game and the same bots in JavaScript.

*How's your luck? Wanna gamble?*

A multi-round, simultaneous-decision game: you against 1, 2 or 4 bots. Every round, everyone secretly
picks one action at the same time; the last player alive wins.

## How it works

At its core 007 is rock-paper-scissors with bullets. Shoot beats Reload (a reloading player is
defenseless), Defense beats Shoot (the shot is blocked), Reload beats Defense (you gain a bullet, they gain
nothing). You must Reload before you can Shoot, and everyone's bullets are public, so the others can guess
what you can afford. Bomb, Reflect and Machine Gun are special moves bought with bullets.

The bots work in layers on top of one rules engine:

```
rules engine  ->  solved 2-player game  ->  model of your habits  ->  Monte-Carlo look-ahead (3+ players)
                  (value iteration + LP)    (saved after every game)
```

- **2 players:** the duel is solved in advance (value iteration with a linear-programming matrix-game solver),
  so the bot starts from a near-optimal mixed strategy.
- **Your habits:** each bot keeps a model of how you play, for example how likely you are to defend again
  after defending several rounds in a row, and blends exploiting it with the safe strategy.
- **3 and 5 players:** Monte-Carlo look-ahead with the real rules engine and the solved 2-player values.
- **Learning:** every finished game is saved on your computer. The bots update their memory of you, and
  `007-train-bot.py` re-tunes two parameters after each game, in the background.

| Difficulty | Deliberate mistakes | How much it exploits your habits |
|---|---|---|
| Easy | 35 % | none |
| Normal | 3 % | medium |
| Hard | none | strong |

## Rules

| Key | Action | Cost and effect |
|---|---|---|
| R | Reload | +1 bullet, defenseless |
| S | Shoot | -1 bullet, attack one player |
| D | Defense | free, blocks Shoot and Bomb |
| B | Bomb | -(living players - 1) bullets, shoots everyone else |
| E | Reflect | once per game; Shoot and Bomb bounce back and the attackers die |
| M | Machine Gun | -10 bullets, goes through Defense and Reflect |

Two shots at each other cancel. Machine Gun against Machine Gun: both die. One more rule is hidden in the
game.

## Run it

**In the browser:** open the link above.

**On your computer** (needs Python 3 and pygame 2):

```
pip install -r requirements-v10.txt              # only pygame
python 007-premode-pygame-v10.2.py               # play
python 007-premode-pygame-v10.2.py --selftest    # rules and solver tests
```

In the game: Space skips the animation of a round; on the Instruction page, put the mouse on an action to
see it played. Options: `--mode 1|2|4`, `--difficulty easy|normal|hard`, `--timer 3|5|10`,
`--skip-tutorial`, `--data-dir DIR`.

The bot trainer:

```
python 007-train-bot.py              # rebuild the bot files from your recorded games
python 007-train-bot.py --status     # what the bots know about you
```

## Where things are

| Path | What |
|---|---|
| `007-premode-pygame-v10.2.py` | The game: rules engine, bots, interface (current version) |
| `007-train-bot.py` | Bot trainer; the game also calls it after every match |
| `007-premode-pygame-v10.py` | Previous version |
| `docs/` | The website version (JavaScript): `js/core.js` = rules and bots, `js/ui.js` = screens. `docs/tests/run_tests.js` checks that it computes exactly what the Python game computes |
| `bot_data_v10/` | Created on first run: your game records and the bots' memory. Stays on your computer |

## Use of AI

Parts of the code and documentation were written with help from Claude (Anthropic), running in Claude Code,
under my direction.

## Credits

- Built with [pygame](https://www.pygame.org) (LGPL).

## Licence

MIT License. See `LICENSE`.
