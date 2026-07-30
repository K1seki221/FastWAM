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
| `09_ironvla_libero.md` | IronVLA (`former/models`) interface contract, LIBERO eval-adapter recipe, and the list of files still needed from the main iron_vla repo |
| `10_router_host_survey.md` | Web survey (2026-07): top VLM+action-DiT frameworks on LIBERO and how the condition router fits each — GR00T N1.7 / π0.5 / FLOWER |
| `11_groot_router_design.md` | Isaac-GR00T N1.7 mapped (vendored at `FastWAM/Isaac-GR00T`, upstream 9c7e746): incumbent VLM→DiT wiring, v1/v2 router insertion design with file:line anchors, official LIBERO recipe, experiment sequence |
| `12_groot_fuyao_runbook.md` | GR00T×router on fuyao: current state, submit chain, provisioned paths, and the container/venv/cache lessons from getting the baseline running |

Facts were verified against the code as of commit `45d8e14` (2026-04). Line numbers are approximate anchors — re-grep if code has moved.
