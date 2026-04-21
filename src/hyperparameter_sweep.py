# src/hyperparameter_sweep.py

import itertools
import subprocess

n_estimators_list = [100, 300, 500, 700, 900, 1100, 1300, 1500]
max_depth_list = [2, 3, 4, 5, 6, 7, 8]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

grid = list(itertools.product(
    n_estimators_list,
    max_depth_list,
    learning_rate_list
))

print(f"Total experiments: {len(grid)}")

# Queue all runs
for i, (n_est, depth, lr) in enumerate(grid, start=1):


    cmd = [
        "dvc", "exp", "run",
        "--queue",
        "-S", f"model.n_estimators={n_est}",
        "-S", f"model.max_depth={depth}",
        "-S", f"model.learning_rate={lr}"
    ]

    subprocess.run(cmd, check=True)

# Execute all queued runs
print("Running queued experiments...")
subprocess.run(
    ["dvc", "exp", "run", "--run-all"],
    check=True
)

print("Hyperparameter sweep complete.")