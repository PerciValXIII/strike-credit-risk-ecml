from __future__ import annotations

import os
import subprocess
import sys
import re
from pathlib import Path
import statistics


N_TRIALS = 5
ROW_SAMPLE_SIZE = 50000
ROW_SAMPLE_SEED = 123

THIS_FILE = Path(__file__).resolve()
ABL_DIR = THIS_FILE.parent
LOG_DIR = ABL_DIR / "logs"

RUN_SCRIPT = ABL_DIR / "run_ablation_grouping_logic.py"
PROJECT_ROOT = THIS_FILE.parents[3]   # strike-credit-risk/


META_AUC_PATTERN = re.compile(r"(?:Meta-model CV AUC:\s*mean=|\[META\]\s*Logistic Regression AUC \(CV\):\s*)([0-9.]+)")


def extract_meta_auc(log_file: Path) -> float | None:
    text = log_file.read_text()
    matches = META_AUC_PATTERN.findall(text)
    if matches:
        return float(matches[-1])
    return None


def main():
    print("\n🚀 Running RANDOM GROUPING ABLATION TRIALS\n")

    results = []

    for seed in range(N_TRIALS):
        print("\n===============================")
        print(f"Trial {seed}")
        print("===============================\n")

        env = os.environ.copy()
        env["GROUPING_STRATEGY"] = "random"
        env["GROUPING_SEED"] = str(seed)
        env["ROW_SAMPLE_SIZE"] = str(ROW_SAMPLE_SIZE)
        env["ROW_SAMPLE_SEED"] = str(ROW_SAMPLE_SEED)

        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not existing_pythonpath
            else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
        )

        proc = subprocess.run(
            [sys.executable, str(RUN_SCRIPT)],
            env=env,
            cwd=str(PROJECT_ROOT),
        )

        if proc.returncode != 0:
            print(f"❌ Trial {seed} failed")
            continue

        logs = sorted(LOG_DIR.glob(f"strike_log_grouping_random_seed{seed}_*.log"))
        if not logs:
            print(f"❌ No log file found for seed {seed}")
            continue

        latest_log = logs[-1]
        auc = extract_meta_auc(latest_log)

        if auc is None:
            print(f"⚠ Could not extract AUC from {latest_log.name}")
            continue

        results.append((seed, auc))
        print(f"✅ Seed {seed} → META AUC = {auc:.4f}")

    print("\n\n===============================")
    print("FINAL RESULTS")
    print("===============================\n")

    for seed, auc in results:
        print(f"Random Seed {seed}: {auc:.4f}")

    aucs = [x[1] for x in results]
    if aucs:
        print(f"\nMean: {statistics.mean(aucs):.4f}")
        if len(aucs) > 1:
            print(f"Std : {statistics.stdev(aucs):.4f}")


if __name__ == "__main__":
    main()