# Nirnay — Belief-Carrying Passport for Predictive Aircraft Maintenance

> Every part carries a belief about its health that travels with it; evidence
> from sensors, records and depot teardowns updates that belief; every repair
> makes the next prediction better; and the commander gets a calibrated
> readiness number per mission.

## Status

| Component | Status |
|-----------|--------|
| Data contracts | `[REAL]` |
| Evidence store & merge | `[PLANNED]` |
| Simulator (referee) | `[PLANNED]` |
| Belief engine (HMM) | `[PLANNED]` |
| Detection (C-MAPSS) | `[PLANNED]` |
| Laya evidence compiler | `[PLANNED]` |
| Mission planner | `[PLANNED]` |
| Demo modules | `[PLANNED]` |
| UI (React + Vite + Plotly) | `[PLANNED]` |
| Sensor rig | `[PLANNED]` |
| Twin-world experiment | `[NOT RUN]` |

## Claims Policy

- Techniques are established; the claim is the composition and the tests.
- No result is reported before it is measured.
- No number is invented.
- Synthetic and public data only.
- The system advises; a human decides.

## Stack

Python 3.11 · FastAPI · SQLite · NumPy/SciPy · scikit-learn · LightGBM ·
PyTorch · OR-Tools CP-SAT · FAISS · NetworkX · PyNaCl · ONNX Runtime ·
React + Vite + Plotly · pytest

## Repository Structure

```
nirnay/
  store/          # evidence items, hash chains, signatures, merge
  belief/         # HMM belief engine, forward-backward, EM
  detect/         # sibling baselines, trust gate, e-detector, alert budget
  laya/           # question registry, synthetic data, fine-tune, calibration
  plan/           # solver, allocator, shadow prices, risk control
  demo/           # nff, disturbance graph, lot contagion, truth score
  api/            # FastAPI service and WebSocket events
simulator/        # frozen generator, never imported by nirnay/ (enforced)
ui/               # fleet board, aircraft twin, passport viewer, etc.
rig/              # firmware and host scripts for the sensor rig
experiments/      # A, B, C, D, E, F, G, H, I, J, K
reports/          # generated charts and tables
tests/            # pytest suite
```

## Quick Start

```bash
# Create and activate the venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
make install

# Run tests
make test
```

## See Also

- [LIMITATIONS.md](LIMITATIONS.md) — what is NOT built or measured yet
- [Build Plan](docs/build_plan.md) — staged execution plan
