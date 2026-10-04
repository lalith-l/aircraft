"""
Generate synthetic C-MAPSS-like data for testing when the NASA download is unavailable.

This creates FD001-style data with:
- Multiple engine trajectories running to failure
- 3 operational settings
- 21 sensor channels
- Degradation trends in key sensors

Status: [SIMULATED]
"""

import os
import numpy as np


def generate_synthetic_cmapss(output_dir: str, subset: str = "FD001",
                               n_engines_train: int = 100,
                               n_engines_test: int = 100,
                               seed: int = 42):
    """Generate synthetic C-MAPSS-like data files."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    def _generate_engine(engine_id: int, max_cycle: int):
        rows = []
        for cycle in range(1, max_cycle + 1):
            # Operating settings (3)
            op1 = rng.choice([-0.0015, -0.0028, 0.0042, 0.001, 0.0])
            op2 = rng.choice([0.0003, -0.0003, 0.0001])
            op3 = rng.uniform(90, 110)

            # 21 sensors with degradation trends
            base = np.array([
                518.67, 642.15, 1589.70, 1400.60, 14.62,
                21.61, 553.89, 2387.72, 9045.29, 1.30,
                47.20, 521.72, 2387.72, 8139.48, 8.4195,
                0.03, 393.0, 2387.91, 100.0, 38.95, 23.3190
            ])

            # Degradation: some sensors trend with cycle/max_cycle ratio
            progress = cycle / max_cycle
            degradation = np.zeros(21)
            # Key sensors that degrade: 2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21
            degrad_indices = [1, 2, 3, 6, 7, 10, 11, 12, 14, 16, 19, 20]
            for idx in degrad_indices:
                degradation[idx] = progress * rng.uniform(5, 30) * (1 if rng.random() > 0.3 else -1)

            noise = rng.normal(0, 0.5, 21)
            sensors = base + degradation + noise

            row = [engine_id, cycle, op1, op2, op3] + sensors.tolist()
            rows.append(row)

        return rows

    # Training data (run to failure)
    train_rows = []
    for eid in range(1, n_engines_train + 1):
        max_life = rng.randint(128, 362)
        train_rows.extend(_generate_engine(eid, max_life))

    train_arr = np.array(train_rows)
    np.savetxt(os.path.join(output_dir, f"train_{subset}.txt"),
               train_arr, fmt="%.4f", delimiter=" ")

    # Test data (cut before failure) + RUL
    test_rows = []
    ruls = []
    for eid in range(1, n_engines_test + 1):
        max_life = rng.randint(128, 362)
        rul = rng.randint(1, max_life // 2)
        cut_at = max_life - rul
        test_rows.extend(_generate_engine(eid, cut_at))
        ruls.append(rul)

    test_arr = np.array(test_rows)
    np.savetxt(os.path.join(output_dir, f"test_{subset}.txt"),
               test_arr, fmt="%.4f", delimiter=" ")
    np.savetxt(os.path.join(output_dir, f"RUL_{subset}.txt"),
               np.array(ruls).reshape(-1, 1), fmt="%d")

    return output_dir


if __name__ == "__main__":
    generate_synthetic_cmapss("data/cmapss", subset="FD001")
    generate_synthetic_cmapss("data/cmapss", subset="FD002", seed=43)
    print("Synthetic C-MAPSS data generated in data/cmapss/")
