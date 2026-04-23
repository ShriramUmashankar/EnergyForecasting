# src/hyperparameter_sweep.py

import itertools
import subprocess

n_estimators_list = [100, 1500]
max_depth_list = [2,  8]
learning_rate_list = [0.02]

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