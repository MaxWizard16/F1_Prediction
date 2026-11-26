from pathlib import Path
import subprocess
import sys

# Keep this file inside the project's src folder.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# Main 2026 prediction workflow, executed in dependency order.
PIPELINE = [
    ("Build historical dataset", "training_dataset.py"),
    ("Generate historical features", "feature_engineering.py"),
    ("Train the 2026 winner prediction model", "train_win_model.py"),
    ("Generate circuit-specific 2026 predictions", "predict_2026.py"),
]


def main():
    print("=" * 65, flush=True)
    print("F1 RACE PREDICTION - FULL PIPELINE", flush=True)
    print("=" * 65, flush=True)

    for number, (description, filename) in enumerate(PIPELINE, start=1):
        script_path = SRC_DIR / filename

        if not script_path.is_file():
            print(f"\nERROR: Cannot find {filename}", flush=True)
            print(f"Expected location: {script_path}", flush=True)
            print("Make sure all four scripts are inside src.", flush=True)
            sys.exit(1)

        print(f"\n[{number}/{len(PIPELINE)}] {description}", flush=True)
        print(f"Running {filename}...", flush=True)
        print("-" * 65, flush=True)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_DIR,
        )

        if result.returncode != 0:
            print(f"\nPIPELINE STOPPED: {filename} returned an error.", flush=True)
            print("Fix the error, then run run_pipeline.py again.", flush=True)
            sys.exit(result.returncode)

        print(f"\nCompleted: {filename}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
