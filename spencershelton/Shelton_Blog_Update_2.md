# Blog Post 2

## Overview

Since the first update, the main work has been finishing the variant data-generation phase. Given that standard chess corpus is done, the remaining goal for this week was to generate the variant-rules corpus that will be used to put pressure on the model for the modular rule-learning experiments.

That run is now finished. The final dataset has:

- `20,000,000` standard chess games from the first run
- `48,000,000` variant training games

The variant corpus took longer than expected as the first production attempt had to be stopped after about three days. The issue was not the orchestrator or worker system but that some of the Fairy-Stockfish variant definitions were not actually equivalent to the C-core rulesets in some edge cases. This became clear on doing some inspection of the generated shards from the in progress run as a part of beginning to think about the dictionary for the eventual model training run.

Due to the issue the partial run could not be used as dataset would contain move labels from Fairy under one rule interpretation while the C core stored legality information under a different rule interpretation. Given the purpose of the dataset is to teach rule behavior, that would make the data unreliable, so I stopped the run and discarded the partial variant output.

## Final variant run

After I fixed the defenitions and greatly tightened validation, I relaunched the variant generation run.

The variant corpus contains `48,000,000` games. It is balanced by the number of changed piece types:

| Changed piece types | Games |
|---|---:|
| 1 | 8M |
| 2 | 8M |
| 3 | 8M |
| 4 | 8M |
| 5 | 8M |
| 6 | 8M |

This bucket structure was important to keep as if most of the data only changed one piece type at a time, the model could learn something closer to standard chess plus isolated exceptions. The multi-piece buckets are intended to force the model to deal with composed rule changes to prevent it from being able to learn in shared portions of the model.

The variant run used different generation settings from the standard chess run as I did not use the standard opening book, because ordinary chess openings are not reliable under changed movement rules meaning the run used pure self-play from the initial position.

The engine policy was also far cheaper than the standard run as the point is to pressure the model to store information in specific model segments, not produce excellent performance in a given ruleset:

- `A`: 50%, 64 nodes, high temperature
- `B`: 30%, 128 nodes, medium temperature
- `C`: 15%, 256 nodes, low temperature
- `D`: 5%, 512 nodes, greedy

As in the standard run, White and Black sampled their policy groups independently. The goal was not to produce strongest-play games but to generate enough coherent play under many rule systems to train a model on rule-conditioned behavior.

## Dataset metadata

The variant shards keep the same basic structure as the standard chess shards, but include additional rule metadata.

Each game stores:

- the full six-piece variant vector
- the ruleset id
- the number of changed piece types
- the changed-piece-count bucket
- whether the game belongs to the held-out split

This is needed as downstream training has to know which rule system produced each move sequence and it also makes it possible to filter by ruleset, bucket, or held-out status during evaluation.

## Current project state

The data side of the project is now pretty much complete, tokens for training do need to be generated from this data but from prior experience with training my CornerCaseLM model this should probably take on the order of a few hours and not days.

Completed:

- standard `20M` game corpus
- variant-capable C ruleset engine
- Fairy-Stockfish compatibility and parity validation
- variant `48M` game training corpus

## Next steps

The next stage is building and training the model architecture.

The first implementation step is to architect and build the the transformer. I will need to navigate keeping the FFNs of my shared portions of the model smaller in comparison to the portions I will swap out based on ruleset. I will also need to put together a system to allow for this swapping out componets of the model at training time

After the base model and its componets are trained, as a basic first test we will ablate one of the swapped out componets gradually and use stockfish or Fairy stockfish to check wether the model's performance with the piece and ruleset the module is supposed to correspond to drops accordingly.

After that, the main evaluation is held-out rule adaptation:

- freeze the shared layers
- freeze existing non-target cores
- insert or train a small new core for a held-out piece rule
- measure whether the new rule is learned
- measure whether unchanged rulesets degrade

The primary experiment is still to see if rapidly switching the format or ruleset of a problem for a modular transformer can force the transformer to store information about the changing ruleset in discrete, severable portions of the model.
