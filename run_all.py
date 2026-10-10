"""End-to-end run: generate data -> features -> cross-validate -> final model -> figures.

Usage:
    python run_all.py                       # use simulated data
    python run_all.py --data my_real.csv    # use your own CSV with columns:
                                            # plant_id, label, t, voltage
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data_generation import build_dataset
from src.features import build_feature_table, feature_dataframe
from src.models import cross_validate_by_plant, get_models
from src.visualize import (plot_confusion, plot_feature_importance,
                           plot_model_comparison, plot_signals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="optional CSV with plant_id,label,t,voltage")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.data:
        df = pd.read_csv(args.data)
        source = f"user data: {args.data}"
    else:
        df = build_dataset()
        df.to_csv(config.DATA_DIR / "synthetic_plant_data.csv", index=False)
        source = "SIMULATED data"
    print(f"Data source: {source} ({len(df):,} rows)")

    plot_signals(df, config.FIGURES_DIR / "signals_by_class.png")

    X, y, groups = build_feature_table(df)
    print(f"Windows: {len(X)}, features: {X.shape[1]}, plants: {len(np.unique(groups))}")
    feature_dataframe(X, y, groups).to_csv(config.DATA_DIR / "features.csv", index=False)

    # 1) Honest evaluation: every fold tests on plants the model never saw.
    cv, n_splits = cross_validate_by_plant(X, y, groups)
    summary = {}
    print(f"\nLeave-plants-out cross-validation ({n_splits} folds)")
    for name, r in cv.items():
        summary[name] = {
            "accuracy_mean": float(np.mean(r["acc"])), "accuracy_std": float(np.std(r["acc"])),
            "macro_f1_mean": float(np.mean(r["f1"])), "macro_f1_std": float(np.std(r["f1"])),
        }
        print(f"  {name:20s} acc {summary[name]['accuracy_mean']:.3f} ± {summary[name]['accuracy_std']:.3f}"
              f"   macro-F1 {summary[name]['macro_f1_mean']:.3f}")
    plot_model_comparison(cv, config.FIGURES_DIR / "model_comparison.png")

    # 2) Detailed report for the best model on one held-out set of plants.
    best = max(summary, key=lambda n: summary[n]["accuracy_mean"])
    plants = np.unique(groups)
    test_plants = plants[-2:]
    test_mask = np.isin(groups, test_plants)
    model = get_models()[best]
    model.fit(X[~test_mask], y[~test_mask])
    pred = model.predict(X[test_mask])
    labels = [c for c in config.CLASSES if c in set(y)]
    cm = confusion_matrix(y[test_mask], pred, labels=labels)
    report = classification_report(y[test_mask], pred, labels=labels, output_dict=True)
    print(f"\nBest model: {best}. Held-out plants {[int(p) for p in test_plants]}")
    print(classification_report(y[test_mask], pred, labels=labels))
    plot_confusion(cm, labels, config.FIGURES_DIR / "confusion_matrix.png",
                   f"{best} - held-out plants")

    # 3) Feature importance from a Random Forest trained on the same split.
    rf = get_models()["Random Forest"]
    rf.fit(X[~test_mask], y[~test_mask])
    importances = rf[-1].feature_importances_
    plot_feature_importance(config.FEATURE_NAMES, importances,
                            config.FIGURES_DIR / "feature_importance.png")

    out = {
        "data_source": source, "n_windows": int(len(X)), "n_plants": int(len(plants)),
        "cv_folds": n_splits, "cv_summary": summary, "best_model": best,
        "holdout_plants": [int(p) for p in test_plants],
        "holdout_report": report,
        "feature_importance": dict(zip(config.FEATURE_NAMES, map(float, importances))),
    }
    with open(config.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics and figures to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
