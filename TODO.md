You already have signs worth investigating (very high train F1 ≈ 0.998 vs lower val/test performance and the confusion matrix showing specific, repeated confusions). Below I give:

1. a quick *interpreted* breakdown of your confusion matrix (so you know which classes to inspect first),
2. a prioritized checklist of what to do **before** stage-2 (so stage-2 has the best chance of helping), and
3. a small, ready-to-run analysis script + terminal commands that will produce per-class metrics, extract misclassified examples, plot confusion heatmaps, and save Grad-CAM images for the worst confusions.

---

# 1) Quick interpretation of your confusion matrix

I assumed the class index order matches your earlier list:

0 — Comminuted
1 — Greenstick
2 — Healthy
3 — Oblique
4 — Oblique Displaced
5 — Spiral
6 — Transverse
7 — Transverse Displaced

Using your confusion matrix I computed per-class precision / recall / F1 (macro F1 of those rows). Rounded percentages:

| Class (idx)              | Support (true count) | Precision | Recall |     F1 |
| ------------------------ | -------------------: | --------: | -----: | -----: |
| Comminuted (0)           |                   17 |     63.6% | 82.35% |  71.8% |
| Greenstick (1)           |                   13 |     52.2% | 92.31% |  66.7% |
| Healthy (2)              |                   10 |     38.5% | 100.0% |  55.6% |
| Oblique (3)              |                   17 |     50.0% | 17.65% |  26.1% |
| Oblique Displaced (4)    |                    9 |     83.3% | 55.56% |  66.7% |
| Spiral (5)               |                   12 |    100.0% | 100.0% | 100.0% |
| Transverse (6)           |                   17 |     83.3% | 29.41% |  43.5% |
| Transverse Displaced (7) |                   17 |    100.0% | 64.71% |  78.6% |

Key takeaways:

* Some classes (e.g., **Oblique (3)** and **Transverse (6)**) have *very low recall* — the model misses many true Oblique/Transverse examples (they are being predicted as other classes).
* **Healthy (2)** has 100% recall but low precision — many other classes are predicted as Healthy (false positives). Look for systematic over-prediction into Healthy (possibly because margins are subtle).
* **Spiral (5)** looks perfect on test (but check sample count and possible leakage/overfitting).
* Overall per-row F1 average from this confusion matrix ≈ **0.636** — which is lower than the WandB `val_macro_f1` you reported; double-check that the confusion_matrix.txt corresponds to the same metric calculation & split (sometimes metrics in the script vs WandB may differ, or the confusion matrix you saved was for a particular checkpoint/epoch).

---

# 2) Prioritized checklist — what to do **before** stage-2 Grad-CAM cropping

Do these in roughly the order below (do not jump directly to stage-2 cropping until you complete at least the first 4):

**A. Sanity checks (mandatory)**

* Confirm `confusion_matrix.txt` is for the same checkpoint / same split that WandB `val_macro_f1` came from. (Possible mix-up: epoch checkpoint vs best.pth.)
* Verify test / val CSVs have correct `image_path` and true `label` — check for duplicate file paths appearing in train and val/test (data leakage).
* Confirm class index → label mapping used at inference matches training mapping.

**B. Inspect misclassified examples (high priority)**

* For classes with low recall (Oblique, Transverse), inspect 20–50 misclassified images each. Look for:

  * Label noise (wrong labels),
  * Very small fractures not visible at current input resize,
  * Different image orientations / cropping,
  * Consistent image artifacts (markers, casts) causing confusion.
* Run Grad-CAM for a sample of correct vs misclassified images to see where model attends.

**C. Learning dynamics & overfitting mitigation**

* You have near-perfect training F1 (0.998) and lower val (~0.92). That indicates overfitting. Try:

  * stronger augmentations (intensity transforms, CLAHE, random contrast / gamma),
  * dropout in classification head (0.2–0.5),
  * higher weight decay (e.g., 1e-2 → 5e-2), or
  * lower model capacity (use `swin_small` → `convnext_tiny` or DenseNet),
  * early stopping based on val macro-F1,
  * label smoothing (0.05) or focal loss if there are hard classes.
* Track and plot train vs val curves (loss and macro-F1) — if val stops improving early, reduce epochs.

**D. Data fixes (if you find errors)**

* Fix mislabeled images, remove duplicates, or reassign ambiguous cases to an “uncertain” label for later adjudication.
* If certain fracture types are extremely small, consider per-image windowing / cropping (but only *after* checking errors above).

**E. Model-level fixes to try before stage-2**

* Class-weighted CE (weights = 1 / sqrt(freq) or inverse freq) or focal loss to force attention to low-recall classes.
* Oversample underperforming classes or use stratified batch sampler so each batch contains examples of hard classes.
* Ensembling (DenseNet + Swin) usually improves recall & reduces specific confusions.

**F. Only after you’ve fixed data issues and attempted regularization, try stage-2 (Grad-CAM cropping)**

* Stage-2 is helpful if the model is already reasonably good and mistakes are due to small ROIs. But if mistakes are due to label noise or overfitting, stage-2 will amplify errors.

---

# 3) Small analysis script (ready to run)

Save the snippet below as `analyze_results.py` next to your repo. It:

* loads `best.pth`,
* runs inference on your test set,
* writes `misclassified.csv`,
* prints per-class precision/recall/f1,
* saves a confusion matrix heatmap and a small folder with example misclassified images,
* optionally generates Grad-CAM images for top misclassifications.

**Terminal**: `python analyze_results.py --checkpoint outputs/swin_mps/best.pth --test-csv data/test.csv --img-root . --class-names "Comminuted,Greenstick,Healthy,Oblique,Oblique Displaced,Spiral,Transverse,Transverse Displaced" --out-dir outputs/analysis`

Run (example):

```bash
python analyze_results.py \
  --checkpoint outputs/swin_mps/best.pth \
  --test-csv data/test.csv \
  --img-root . \
  --model swin \
  --class-names "Comminuted,Greenstick,Healthy,Oblique,Oblique Displaced,Spiral,Transverse,Transverse Displaced" \
  --out-dir outputs/analysis
```

After it runs:

* `outputs/analysis/confusion_matrix.png` — nice heatmap with absolute counts,
* `outputs/analysis/misclassified.csv` — a list you can open in Excel,
* `outputs/analysis/examples/*` — a folder with top confusion pairs (10 example images each).

If you want Grad-CAM images for those misclassified examples I can give a short script to overlay cams and save them — tell me and I’ll provide it.

---

# 4) Suggested immediate experiments (ordered)

1. **Run the analysis script** above and inspect the top 3 confusion pairs and 30 misclassified images total. If >10% of those are labeling errors, fix labels and retrain.
2. If labels look correct: **increase regularization** — add dropout 0.3 to classification head, increase weight decay (to 5e-2), and stronger intensity augmentations (random gamma, CLAHE). Re-run for 5–10 epochs and watch val F1.
3. **Stratified sampling / oversample** poor recall classes (Oblique, Transverse) to give the model more signal for those classes.
4. Try **class-weighted cross entropy** or **focal loss** (gamma=1.0) focusing on hard classes.
5. If high confusion is localized to small fractures: only then attempt **stage-2 Grad-CAM crops** (but after steps 1–4). Stage-2 works best after data and regularization issues are resolved.
6. Use a **light ensemble** (DenseNet + ConvNeXt or DenseNet + Swin) to improve recall on hard classes — you can average probabilities from two checkpoints (fast and usually effective).
7. Add a **QA step** where you present top-5 low-confidence predictions to a clinician for relabeling (active learning loop).

---

# 5) Short checklist to report back (so I can help more)

After you run the `analyze_results.py`, tell me:

* which confusion pairs (top 3) the script found and a short sample of what those images look like,
* whether you found label errors / duplicates,
* whether train vs val loss/metric curves show overfitting early (I can give a small plotting snippet if you have the per-epoch wandb logs or saved epoch files).

With that I will:

* suggest exact hyperparameter edits and a second training command tuned to the failure modes (e.g., class weights, focal loss, augmentation list), or
* provide the Grad-CAM overlay script and a staged stage-2 workflow once data issues are resolved.

---

If you want, I can immediately (pick one):

* produce the small Grad-CAM overlay script for the misclassified examples, or
* generate a tuned `--best` training command (exact hyperparameters) targeting the Oblique/Transverse failure cases (with a short explanation).

Which would you like next?
