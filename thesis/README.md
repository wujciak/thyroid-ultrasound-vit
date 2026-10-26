# Engineering thesis notebooks (archive)

These are the notebooks of my engineering thesis on thyroid cancer detection from ultrasound
with Vision Transformers (Wrocław University of Science and Technology, 2024). They are kept
**as defended**: code and markdown are in Polish and unchanged. The only edit is
clearing the outputs of `data_merge_thesis.ipynb`, which printed patient names from the source
CSV files.

| Notebook | Content |
| --- | --- |
| `data_merge_thesis.ipynb` | anonymising and merging the Chinese dataset (batch 1 and 2) with AUITD |
| `transformers_thesis.ipynb` | Vision Transformer implemented from scratch in PyTorch |
| `cnn_thesis.ipynb` | baseline CNN in Keras |

Reported results on the validation split: ViT accuracy 0.68 and CNN accuracy 0.51.

## Known issues

Revisiting the code found problems that make the reported numbers unreliable. The repository
root contains the corrected pipeline.

| Issue | Effect | Fixed in the new pipeline by |
| --- | --- | --- |
| The CNN report compares `val_generator.classes` (directory order) with `model.predict(val_generator)`, but the generator uses Keras's default `shuffle=True` | labels and predictions are misaligned, so the 0.51 is meaningless | ordered evaluation loaders; predictions stored next to their image IDs |
| Split by image, not by patient | images of the same patient fall into both train and validation | `StratifiedGroupKFold` on patient IDs |
| AUITD `test/` is a copy of `train/`, `train/benign` contains manual copies (`- Copy (2).jpg`), and 1026 of its 2289 files are exact duplicates | identical images in train and validation | AUITD excluded; near-duplicate detection (dHash) over the remaining data |
| AUITD labels are TI-RADS categories mapped to benign (2–3) and malignant (4A–5) | most 4A/4B nodules are benign, so much of the "malignant" class is label noise; the Chinese data uses histopathology | only histopathology-labelled data |
| No test set: early stopping and model choice use the same split that is reported | optimistic estimate | held-out test patients that no decision depends on |
| `data_merge` copies every file in a batch-2 patient folder instead of the frames listed in `batch2_image.csv` | report scans (A4 pages) can enter the image set | the CSV is the source of truth |
| 1500 malignant images were removed at random outside the code | not reproducible | class weights in the loss, no deletion |
| ViT-Base (86M parameters) trained from scratch on ~9k images; CNN with lr 1e-5 and no augmentation | the CNN validation accuracy falls from epoch 1; the ViT predicts the majority class for 5 epochs | pretrained baselines next to the scratch models, augmentation, tuned learning rates |

The ViT class from `transformers_thesis.ipynb` is reused unchanged as `src/thyroid/models/vit_thesis.py`,
and the CNN is ported layer for layer to `src/thyroid/models/cnn_thesis.py`, so the thesis
architectures are part of the new comparison.
