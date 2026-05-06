# RAGEN

This repository contains our RAGEN experiments for puzzle generation and lookahead training.

## Layout

- `generate_puzzles.py`: puzzle generation entry point.
- `train_lookahead_baseline.sh`: baseline lookahead training script.
- `train_lookahead_treatment.sh`: treatment lookahead training script.
- `config/base.yaml`: shared experiment configuration.
- `ragen/`: RAGEN agent and trainer code.
- `verl/`: vendored verl source kept as a normal folder, not a Git submodule.

## Notes

Large local artifacts and crash dumps should stay out of Git. In particular, `verl/core` is a local crash dump and is ignored.

## Basic Usage

```bash
python generate_puzzles.py
bash train_lookahead_baseline.sh
bash train_lookahead_treatment.sh
```
