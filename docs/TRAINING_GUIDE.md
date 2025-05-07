# Model Training Guide - Simple Steps

## What This Does

This training script will:
- **Automatically** use your existing 2,135 training images (no manual upload!)
- **Balance** the classes so model learns all fruits equally
- **Augment** data (create variations) to prevent overfitting
- **Train** in 2 phases for best results
- **Save** the best model automatically

Expected improvement: **21.87% → 75-90% accuracy**

---

## Prerequisites

Before running:
1. Virtual environment activated
2. TensorFlow installed (already done)
3. GPU detected (optional, but faster)
4. Dataset in correct location (already there)

---

## How to Run

### Simple Method (One Command):

```bash
python scripts/train_balanced_model.py
```

That's it! The script does everything automatically.

---

## How Long Does It Take?

**With GPU (RTX 4070):**
- Phase 1 (10 epochs): ~5-10 minutes
- Phase 2 (up to 50 epochs): ~20-40 minutes
- **Total**: 25-50 minutes

**With CPU only:**
- Phase 1: ~30-60 minutes
- Phase 2: ~2-4 hours
- **Total**: 2.5-5 hours

The script stops automatically if accuracy stops improving (early stopping).

---

## TensorBoard Live Dashboard

To view live training progress:

```bash
tensorboard --logdir=logs/fit
```

Then open: http://localhost:6006

---

## 📂 Output Files

After training completes, you'll find:

```
models/
├── fruit_ripeness_balanced_YYYYMMDD_HHMMSS.keras  # Best model (auto-saved)
├── fruit_ripeness_final.keras                     # Final model
├── class_labels.json                              # Class names
├── training_info.json                             # Training details
└── training_log_YYYYMMDD_HHMMSS.csv              # Full training history
```

---

## After Training

1. **Update your app** to use the new model
2. **Test the new model**:
   ```bash
   python apps/app_flask.py
   # or
   streamlit run apps/app_streamlit.py
   ```

3. **Compare results** with old 21.87% accuracy

---

## Troubleshooting

### "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "GPU not detected"
- Training will still work (just slower)
- Make sure you're in Windows environment, not WSL2

### "Out of memory"
- Reduce `BATCH_SIZE` in the script
- Change from 32 to 16 or 8

---

## Expected Results

**Current Model:**
- Accuracy: 21.87%
- Predicts only "rottenbanana" (100% of time)

**New Model (Expected):**
- Accuracy: **75-90%**
- Predicts all 6 classes correctly
- **Improvement: ~60% accuracy increase!**
