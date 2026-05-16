# Building a Massive Chess Dataset for Modular Rule Learning

One of the central goals of this project is to train a model that does not just memorize how standard chess works, but can actually represent piece movement rules in a modular and editable way. That means the data pipeline has to do more than generate strong ordinary chess games. It has to create pressure for the model to learn the rules themselves.

Over the last week, we finished the large orthodox-chess corpus that serves as the base dataset for the project. At the same time, we built the engine and generation infrastructure needed for the next phase: a much larger variant-rules corpus designed specifically to stress compositional rule learning.

## The Standard Chess Corpus Is Done

The first major generation run produced a full orthodox-chess corpus of `20,000,000` games. This dataset is the baseline for the project: it gives us a large body of ordinary chess play to train on before introducing structured rule changes.

The completed run used:

- a `C11` chess core as the legality, replay, and adjudication authority
- Stockfish as the move-selection and evaluation oracle
- a compiled opening-book pipeline based on the Lichess openings dataset
- a multiprocessing worker system with resumable orchestration

In total, the run produced roughly:

- `20,000,000` games
- about `2.03B` plies
- about `2.03B` Stockfish queries
- a total runtime of around `131` hours

The dataset was generated with a mixed-strength policy setup so the resulting games were not all at one fixed level of play. Each side in each game independently sampled from four generation groups:

- `A`: `35%`, `512` nodes, temperature `1.20`
- `B`: `30%`, `1024` nodes, temperature `0.75`
- `C`: `25%`, `2048` nodes, temperature `0.30`
- `D`: `10%`, `4096` nodes, greedy

That produced a corpus with a useful range of move quality and prevents an overrepresentation of drawn games while still staying grounded in a strong engine signal.

The opening distribution was also intentionally mixed:

- about `84.98%` opening-seeded
- about `15.02%` pure self-play

(To be clear this was random sampling with a goal of about 15% self play)

Final outcome statistics for the assembled standard corpus came out to approximately:

- white wins: `49.37%`
- black wins: `47.70%`
- draws: `2.93%`

At this point, the orthodox base dataset is complete and stable.

## The Infrastructure Had to Be Built First

Before any large run was possible, the project needed a generation stack that could survive long runtimes, recover from interruptions, and produce artifacts that were easy to validate and resume.

That work is now in place.

On the engine side, the project has:

- a native `C` chess core with FEN support, legal move generation, apply/undo, hashing, adjudication, and replay
- validation via perft, unit tests, and regression coverage

On the data side, it has:

- an opening-book formatting pipeline that stages raw opening data and compiles a deterministic runtime artifact
- a Stockfish UCI integration layer
- a multiprocessing worker model
- a resumable orchestrator with detached execution, progress reporting, control files, and graceful pause/stop/resume behavior
- generated shard output under the project data directory
- persistent manifests, metrics, heartbeats, and operational logs

## Why Standard Chess Alone Is Not Enough

If the only training data were ordinary chess, a model could perform well simply by absorbing standard movement behavior into shared layers. That is exactly what this project is trying to avoid.

The broader research goal is to encourage a model architecture where piece-rule information lives in a modular form that can later be swapped, retrained, or adapted with minimal interference. To pressuree the model into storing representations this the data will include systematic rule changes.

That leads to the second major generation effort: a structured variant-rules corpus.

## The Variant Ruleset Library

The next dataset is built around a finalized library of `30` atomic piece-rule variants:

- `5` pawn variants
- `5` knight variants
- `5` bishop variants
- `5` rook variants
- `5` queen variants
- `5` king variants

These variants are designed to be interpretable, piece-local, and composable. The goal is not to create arbitrary fantasy chess for its own sake. The goal is to create controlled movement-rule changes that force the model to keep rule information active.

Of these `30` atomic variants, `6` are held out for later adaptation and evaluation:

- `P5`
- `N5`
- `B5`
- `R5`
- `Q5`
- `K5`

The remaining `24` atomic variants form the main seen-variant training library.

## The Variant Corpus Will Be Much Larger

The current target for the variant dataset is `48,000,000` games.

Instead of concentrating mostly on small local edits, the dataset is being balanced by the number of piece types that are changed in a ruleset. The idea is to prevent the model seeing too many standard rulesets allowing it to learn to store some data in shared layers.

The planned split is:

- `8M` games with exactly `1` changed piece type
- `8M` games with exactly `2` changed piece types
- `8M` games with exactly `3` changed piece types
- `8M` games with exactly `4` changed piece types
- `8M` games with exactly `5` changed piece types
- `8M` games with exactly `6` changed piece types

This makes the data-generation objective different from the standard run. For the variant corpus, the priority is not to play these rulesets as strongly as possible but instead to make the model repeatedly confront and compose movement rules.

## The C Engine Now Supports Variant Rulesets

To make this possible, the chess core had to be extended from a standard-only engine into a compiled per-game ruleset engine.

That work is now complete enough for the active generation phase.

The engine now supports:

- a real public `ruleset` construction API
- per-piece variant ids for all six piece types
- compiled movement and capture specifications
- composed multi-piece rulesets
- variant-aware legal move generation
- variant-aware attack detection
- variant-aware castling gating
- variant-aware en passant compatibility handling
- orthodox promotion retained across pawn variants
- conservative variant insufficient-material handling

This is a major shift from the original standard-only core. The project now has a native legality engine that can actually represent and enforce the approved atomic variant library.

## Fairy-Stockfish Is the Variant Oracle Path

For the variant run, the project is moving away from the heavy orthodox Stockfish settings used in the standard run. The goal now is faster generation with reasonably coherent play, not maximum engine fidelity.

The chosen path is Fairy-Stockfish.

That required new support code as well:

- a Python-side registry of the atomic variant library
- support classification for which variants are clean Fairy-Stockfish candidates and which still require manual handling
- rendering of custom `variants.ini` definitions
- a compatibility harness that can:
  - build the matching C-core ruleset
  - write Fairy-Stockfish custom-variant config
  - run engine checks
  - search from `startpos`
  - verify that returned moves are legal under the C core
  - emit a structured compatibility report

This compatibility step is important because the project is not assuming that every atomic variant will map cleanly into Fairy-Stockfish without adjustment. Some variants are straightforward, while others are still marked for manual review.

## The Variant Run Is Now Underway

At this point, the project has moved out of basic infrastructure development and into active variant-corpus generation.

The current operational state is:

- the standard orthodox corpus is complete
- the variant ruleset library is finalized
- the C variant engine is implemented
- the Fairy-Stockfish compatibility path is in place
- the variant generation program is underway

The current working expectation is that the variant corpus run will hopefully finish in roughly six days.

## Why This Matters

The completed standard corpus gives the project a strong base distribution of ordinary chess play. The variant corpus is what makes the modularity objective real.

Together, the two datasets support the full training program:

- standard baseline training
- structured seen-variant training
- held-out atomic variant adaptation
- interference and modularity evaluation

That is the real purpose of all of this engineering effort. The project is not just generating chess games at scale. It is building the data foundation for testing whether a model can learn movement rules in a modular, editable, and compositional way.

## Current Status

In short:

- the orthodox `20M`-game dataset is finished
- the supporting generation stack is built and stable
- the variant ruleset library is finalized
- the C engine now supports compiled variant rulesets
- Fairy-Stockfish compatibility tooling is in place
- the variant `48M`-game generation run is in progress
- expected remaining runtime is about five days

If everything proceeds as expected, the next major milestone will be having both the standard and variant corpora in hand, ready for the next training phase of the project.
