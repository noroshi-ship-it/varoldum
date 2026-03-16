# varoldum

artificial life simulation. agents with evolved neural networks live in a procedurally generated world. they perceive, decide, eat, build, communicate, reproduce, and die. what survives gets to define what matters.

## run

```bash
pip install -r requirements.txt
python main.py
```

quick test:
```bash
python main.py --no-gpu --pop 100 --width 96 --height 96 --ticks 1000
```

resume from checkpoint:
```bash
python main.py --resume
```

| flag | default | what it does |
|------|---------|--------------|
| `--ticks` | 50000 | simulation length |
| `--seed` | 42 | random seed |
| `--pop` | 200 | initial population |
| `--width` | 192 | world width |
| `--height` | 192 | world height |
| `--no-gpu` | off | cpu-only mode |
| `--output` | `output` | output directory |
| `--resume` | - | resume from last checkpoint |
| `--quiet` | off | suppress console output |

## dashboard

```bash
python dashboard.py output
```

opens `http://localhost:8420`. population dynamics, intelligence metrics, ecology, social behavior, trait evolution, discovered rules, event timeline, hall of fame.

## what's in here

**agents**
- bottleneck encoder brain with GRU. architecture is genome-controlled and evolvable
- world model that learns to predict next environment state
- self-model that predicts own internal state. prediction error drives curiosity
- hypothesis system: agents form and test IF-THEN rules (40 features, 20 outcomes)
- symbol codebook: continuous concepts discretized into composable symbols via VQ
- discrete token language grounded in concept space. meanings drift with experience
- mortality model: agents learn to predict own survival from concepts + body state
- inscriptions: agents write token sequences onto structures. knowledge outlives the writer

**world**
- ecology: 4 plant species, 3 fauna types (herbivore, predator, decomposer)
- chemistry: 16 substances with discoverable reactions and medicine recipes
- terrain: tectonic plates, earthquakes, erosion, water flow
- hidden physics: radiation, magnetic fields, underground resources, soil pH
- structures: walls, farms, traps, nests, storage, markers
- social: trade, combat, cooperation, teaching, imitation

## output

| file | content |
|------|---------|
| `population.csv` | per-tick stats |
| `traits.csv` | genome trait distributions over time |
| `consciousness.csv` | self-model accuracy, information integration |
| `discovered_rules.csv` | IF-THEN rules agents learned |
| `composable_rules.csv` | multi-condition rules |
| `events.json` | milestones, extinctions, booms |
| `hall_of_fame.json` | notable agents |
| `checkpoint_*.pkl` | full state for resuming |

## design principle

physics (energy, damage, death, temperature) is hardcoded. everything above it — what counts as danger, food, a safe place, an ally — is learned. symbols, language meanings, mortality concepts, behavioral rules: all emerge from experience and evolution.

## license

MIT
