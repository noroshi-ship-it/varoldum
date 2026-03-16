# varoldum

neural agents in a grid. they eat, die, reproduce. sometimes they build stuff.

there's a bottleneck encoder so each agent compresses the world into a few neurons. what those neurons "mean" is up to evolution. there's also a world model, proto-language, hypothesis system, etc. most of it probably doesn't work yet.

## run

```bash
pip install numpy
python main.py --pop 500 --ticks 100000
```

gpu + bigger pop:

```bash
pip install torch
python main.py --pop 2000 --ticks 10000000
```

web dashboard (flask):

```bash
pip install flask
python web/server.py --output output
# localhost:8420
```

## what's in here

- agents with evolved neural nets (bottleneck encoder, GRU policy, world model)
- structures (nests, farms, walls, traps)
- proto-language (4-float signals, no predefined meaning)
- concept hypotheses (agents make IF-THEN guesses about their own neurons)
- composable rules, meta-cognition, culture, teaching, imitation
- hall of fame for dead agents who did something notable
- web dashboard with live stats, gpu temp, population chart

## will it produce consciousness?

no.

## license

MIT
