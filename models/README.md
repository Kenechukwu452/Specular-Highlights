# Models

This folder contains a collection of candidate highlight removal models for benchmarking and ablation studies, alongside our own proprietary architectures developed for this project.

Each model is self-contained in its own subdirectory, including its training and inference code, configuration files, and (optionally) experiment scripts. Models in this folder may be evaluated independently, compared under a consistent protocol, and combined into hybrid or ensemble pipelines where beneficial.

## Conventions

- **One model per folder**: `models/<model_name>/`
- **Self-contained**: model-specific code and configs should not modify shared pipeline components.
- **Shared dependencies**: common dataset loaders, rendering outputs, transforms, and metrics should be imported from the repository-level `pipeline/` and `helpers/` modules.
- **Evaluation consistency**: all models should use the same dataset splits and reporting metrics unless a deviation is explicitly documented in the model’s `README.md`.

## Typical contents of a model folder

```text
models/<model_name>/
├─ src/              # architecture and training logic
├─ scripts/          # train/eval/infer entry points
├─ configs/          # experiment configs
├─ checkpoints/      # optional, typically gitignored
└─ README.md         # model-specific setup and usage and information