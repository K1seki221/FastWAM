# cc_memo — Claude Code memory for the FastWAM repo

Distilled codebase knowledge for future sessions. Read `00_overview.md` first, then drill into the topic files as needed. `06_gotchas.md` is the consolidated foot-gun list — skim it before editing or running anything.

| File | Contents |
|---|---|
| `00_overview.md` | What the repo is (paper, provenance), directory layout, end-to-end workflow, env vars |
| `01_model_architecture.md` | MoT design, FastWAM/Joint/IDM variants, ActionDiT, Wan2.2 backbone, VAE/T5/scheduler, checkpoint format |
| `02_training.md` | Launch chain, `Wan22Trainer` internals, run-dir layout, resume semantics, preprocessing scripts, utils |
| `03_data.md` | Dataset stack (LeRobot v2.1), training batch contract, LIBERO vs RoboTwin, normalization, text-embed cache |
| `04_configs.md` | Hydra composition graph, the 6 task configs, key config keys, custom resolvers |
| `05_evaluation.md` | LIBERO tmux-parallel eval, RoboTwin eval via vendored harness, policy interface, output layouts |
| `06_gotchas.md` | Cross-cutting foot-guns, deduplicated and ranked |
| `07_fuyao_cluster.md` | Running FastWAM on the fuyao cluster (paths, PytorchJob wrappers, what to port from Xiaopeng Zhang's fork in `former/FastWAM`) |
| `08_repro_fidelity.md` | Verified paper recipe (arXiv:2603.16666), target LIBERO numbers, upstream-identity + wrapper audit results |

Facts were verified against the code as of commit `45d8e14` (2026-04). Line numbers are approximate anchors — re-grep if code has moved.
