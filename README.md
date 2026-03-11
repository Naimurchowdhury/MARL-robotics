# Mukti Production Floor RL

This project simulates a Mukti warehouse robot on a production floor. The robot collects products from inbound stations and dispatches them to outbound bays while avoiding obstacles and managing battery constraints.

## Files

- `src/mukti_env.py`: custom environment and renderer
- `src/train.py`: Q-learning training entrypoint
- `src/render_simulation.py`: renders a trained episode to GIF/PNG
- `src/improve_training.py`: mutates the training config, retrains, and commits only if performance improves
- `config/training_config.json`: training hyperparameters used by the trainer

## Run

```bash
python3 src/train.py
python3 src/render_simulation.py
python3 src/improve_training.py
```

Training outputs are written to `artifacts/`.
