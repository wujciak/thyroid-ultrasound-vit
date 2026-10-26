# Thyroid Nodule Classification from Ultrasound

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-UI-61DAFB?logo=react&logoColor=black)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9)
![License](https://img.shields.io/badge/License-MIT-blue)

Benign vs malignant classification of thyroid nodules from ultrasound frames, comparing Vision
Transformers and CNNs with **patient-level** evaluation on histopathology-labelled data, plus a
small web app that serves the best model.

This started as my engineering thesis. Revisiting it showed the original evaluation could not be
trusted: images of the same patient appeared in both training and validation, one dataset's test
folder was a copy of its training folder, and the CNN metrics were computed on misaligned labels.
This repository is the thesis redone properly. The original notebooks are kept unchanged in
[`thesis/`](thesis/), with a list of what was wrong in them.

<!-- TODO after training: headline result (best model, test AUROC image / patient) and ROC figure -->

## Results

<!-- TODO after training: paste results/summary.md, ROC curves and Grad-CAM from notebooks/02_results.ipynb -->

*Training in progress.*

## Data

[Figshare 26067475](https://doi.org/10.6084/m9.figshare.26067475) (CC BY 4.0), with two batches
of ultrasound frames, each patient labelled by **histopathology**. After cleaning there are
**7813 frames from 785 patients**; 64% of the patients are malignant.

| | Patients | Frames |
| --- | --: | --: |
| Benign | 282 | 2608 |
| Malignant | 503 | 5205 |

Cleaning (details and counts in [`notebooks/01_data.ipynb`](notebooks/01_data.ipynb)):

- only the frames listed in the batch CSVs are kept, which drops 773 scanned pathology reports;
- patients are re-identified as `b1-XXXX` / `b2-XXXX` and no names are written anywhere;
- near-duplicates are removed (dHash); the same image under two patients is dropped entirely;
- AUITD, the second dataset used in the thesis, is excluded. Its labels are TI-RADS categories
  rather than pathology, its `test/` folder duplicates `train/`, and it has no patient IDs.

## Method

| Model | Init | Input |
| --- | --- | --- |
| `vit_thesis` | from scratch (the thesis ViT, unchanged) | 256 px |
| `cnn_thesis` | from scratch (the thesis CNN, ported from Keras) | 256 px |
| `resnet50` | ImageNet | 224 px |
| `vit_b16` | ImageNet-21k → 1k | 224 px |

- **Splits:** 157 test patients (20%) are held out. The remaining 628 patients form 5 folds
  (`StratifiedGroupKFold` on source × label), so no patient is ever in two splits. The split is
  committed in [`splits/`](splits/).
- **Training:** each model is trained 5 times, once per fold. Fold *k* is validation, used for
  early stopping (image AUROC) and for the decision threshold (Youden's J). The other folds are
  training data. AdamW, cosine schedule with warmup, class-weighted loss, light augmentation
  (crop, flip, ±10° rotation, brightness/contrast), bf16.
- **Evaluation:** every fold model is scored on the untouched test set. The report gives
  mean ± std over the 5 models and a 5-model ensemble, at image level and at patient level
  (mean score over a patient's frames). Metrics: AUROC, AUPRC, sensitivity, specificity, balanced
  accuracy, and majority-class accuracy as a reference.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU for training.

```bash
uv sync
./scripts/download_data.sh                 # 1.9 GB from Figshare, MD5-checked
uv run python -m thyroid.data.prepare      # -> data/processed/, splits/manifest.csv
uv run python -m thyroid.data.splits       # -> splits/patients.csv (already committed)

./scripts/train_all.sh                     # 4 models x 5 folds, resumable -> runs/, results/
uv run python -m thyroid.export --model vit_b16   # -> models/vit_b16.pt for the app
```

Single runs: `uv run python -m thyroid.train --model resnet50 --fold 0`. Pipeline check on CPU in
seconds: add `--smoke`.

## Web app

FastAPI backend and React frontend. The backend loads an exported model and returns the
malignancy probability together with the validation-chosen threshold.

```bash
cd app
MODEL=vit_b16 docker compose up --build     # UI on http://localhost:3000, API on :8000
```

<!-- TODO after training: screenshot -->

The app is a research demo. It is not a medical device and must not be used for diagnosis.

## Project structure

```
src/thyroid/
  data/prepare.py      raw archives -> anonymised, deduplicated frames + manifest
  data/splits.py       patient-level test set and CV folds
  data/dataset.py      PyTorch dataset and transforms
  models/              registry; vit_thesis.py and cnn_thesis.py are the thesis architectures
  train.py             one model x one fold: training, early stopping, test predictions
  metrics.py           image- and patient-level metrics, thresholds
  summarize.py         fold aggregation and ensemble -> results/
  export.py            trained fold -> self-contained file for serving
notebooks/             01_data (dataset and cleaning), 02_results (after training)
app/                   FastAPI backend, React frontend, docker-compose
thesis/                original thesis notebooks (archive) and their known issues
splits/                manifest and patient split (no personal data)
```

## Limitations

- The frames carry burned-in scanner UI, including calipers and colour-Doppler boxes that
  sonographers add to suspicious nodules. A model can exploit them; Grad-CAM in `02_results`
  checks where the models look.
- All data comes from one published dataset. There is no external test set from another hospital, so
  the numbers say nothing about generalisation to other scanners or populations.
- Labels are per patient; individual frames may not show the diagnosed nodule.

## License

[MIT](LICENSE)
