
import csv
from pathlib import Path

from experiments import plot_results

results_dir = Path("results")

with (results_dir / "raw_results.csv").open(
    newline="", encoding="utf-8"
) as f:
    raw_rows = list(csv.DictReader(f))

with (results_dir / "epoch_history.csv").open(
    newline="", encoding="utf-8"
) as f:
    history_rows = list(csv.DictReader(f))

# Convert numeric CSV columns back to numbers.
for row in raw_rows:
    for key in ("seed", "training_time", "test_loss", "test_accuracy"):
        row[key] = float(row[key])

for row in history_rows:
    for key in (
        "seed", "epoch", "train_loss",
        "train_accuracy", "val_loss", "val_accuracy"
    ):
        row[key] = float(row[key])

plot_results(raw_rows, history_rows, results_dir)

print("All comparison plots regenerated successfully.")
