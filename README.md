# Plant Bioelectric Stress Classifier

A machine-learning pipeline that classifies a plant's condition (**healthy, dehydrated, overwatered, cold-shocked, touched**) from the electrical signal measured between two electrodes.

> **Data disclosure:** All results in this repository come from **simulated signals**, generated to follow behaviours described in plant electrophysiology research. No real electrode recordings were used yet. The goal of this stage is to build and validate the full pipeline (filtering, windowing, features, models, honest evaluation) so that real recordings can be dropped in later. Accuracy on simulated data should **not** be read as accuracy on real plants.

## Why this problem

Plants show electrical responses to stress (water deficit, temperature shock, touch) before visible symptoms like wilting. Published studies have classified plant stress from such signals with classical ML and CNNs. This project reproduces the core pipeline end to end and is designed to extend to real hardware (ESP32/Arduino + electrodes) and deep learning.

## Pipeline

```
raw voltage -> band-pass filter -> 60 s windows (50% overlap) -> 10 features/window
            -> StandardScaler -> classifier -> leave-plants-out evaluation
```

| Stage | File | What it does |
|---|---|---|
| Data | `src/data_generation.py` | Simulates 8 plants x 5 conditions x 1 h at 1 Hz, with per-plant offset and gain |
| Cleaning | `src/preprocessing.py` | Order-3 zero-phase Butterworth band-pass, sliding windows |
| Features | `src/features.py` | mean, variance, std, zero crossings, dominant frequency (FFT), spectral energy, spike count, max abs, range, slope |
| Models | `src/models.py` | Logistic Regression, KNN, SVM (RBF), Random Forest, Gradient Boosting |
| Evaluation | `src/models.py` | `GroupKFold` by plant, so every test fold contains plants never seen in training |
| Figures | `src/visualize.py` | Signals, confusion matrix, model comparison, feature importance |

## Results (simulated data, 4-fold leave-plants-out CV)

| Model | Accuracy (mean ± std) | Macro-F1 |
|---|---|---|
| Gradient Boosting | 0.839 ± 0.042 | 0.839 |
| Random Forest | 0.838 ± 0.036 | 0.837 |
| SVM (RBF) | 0.826 ± 0.044 | 0.825 |
| Logistic Regression | 0.782 ± 0.045 | 0.781 |
| KNN (k=5) | 0.720 ± 0.028 | 0.720 |

Chance level is 0.20 (5 balanced classes). Detailed numbers are in `results/metrics.json`; figures are in `results/figures/`.

Observations on one held-out pair of plants (Gradient Boosting, 78% accuracy):
- `overwatered` (F1 0.90) and `cold_shock` (F1 0.88) are easiest: they have distinctive oscillation / transient signatures.
- `healthy` is hardest (F1 0.60, recall 0.48): its low-amplitude signal overlaps with the early part of a dehydration drift and with quiet stretches of other classes inside a 60 s window.

## Design decisions

- **Split by plant, not by row.** Windows from the same plant share baseline and noise. Random row splits let a model recognise the plant instead of the condition and inflate accuracy. `GroupKFold` prevents this.
- **Scaler inside the pipeline.** `StandardScaler` is fitted on training folds only.
- **Overlapping windows** give more training samples; because evaluation is grouped by plant, overlap cannot leak into the test fold.
- **Several models compared** to show a linear baseline, an instance-based model, a kernel model, and two tree ensembles on the same folds.

## Run it

```bash
git clone <your-repo-url>
cd plant-bioelectric-classifier
pip install -r requirements.txt
python run_all.py                  # simulate data, evaluate, write results/ and figures
python -m pytest -q                # run tests
```

Using real data: provide a CSV with columns `plant_id,label,t,voltage` (label must be one of the names in `src/config.py`, or edit `CLASSES`) and run:

```bash
python run_all.py --data my_recordings.csv
```

## Limitations

- Simulated data is far cleaner than real electrode recordings; real noise, motion artefacts, and sensor drift will lower accuracy.
- Only 8 simulated plants and one species model.
- Classical ML on hand-crafted features only; no deep learning yet.

## Next steps

1. Record real signals (ESP32 or Arduino + instrumentation amplifier + Ag/AgCl electrodes) with labelled conditions.
2. Benchmark on public plant electrophysiology data (e.g. HIPB-MM on Hugging Face).
3. Add a 1D-CNN / LSTM on raw windows and compare with the feature-based models.
4. Live dashboard that shows the current prediction.

## Project layout

```
run_all.py          end-to-end entry point
src/                config, data generation, preprocessing, features, models, plots
tests/              unit tests
results/            metrics.json and figures (generated)
data/               generated CSVs (git-ignored except features)
```

## License

MIT
