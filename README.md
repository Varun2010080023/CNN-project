# Fruit Lens

A Streamlit app that classifies a photo as **apple** or **banana** with a custom PyTorch CNN trained from scratch on `apple_banana_dataset/`. No pretrained network, external image API, or downloaded training data is used.

## Run

Tested with Python 3.12. Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

Train (or retrain) the model, then start the standalone glassmorphism interface:

```powershell
python train.py
python web_app.py
```

If `artifacts/fruit_cnn.pt` already exists, you can start the web interface directly. Open http://127.0.0.1:8501 and upload a JPG/PNG/WebP photo of your fruit. The interface uses the local CNN and reads facts and nutrition from `fruit_info.json`; it does not require Streamlit. The older Streamlit interface remains available with `python -m streamlit run app.py`.

## Your facts and nutrition information

Edit `fruit_info.json`. Facts and nutrition are intentionally empty until you provide your content. The model predicts the fruit; the app looks up the matching information. Nutrition values are **not** inferred from image pixels.

For each fruit:

- `facts`: a list of fact strings. The app picks one randomly for each image. “Give me another fact” selects a different fact when available.
- `nutrition.serving`: the serving size your values describe, such as `100 g` or `1 medium fruit (126 g)`.
- `nutrition.nutrients`: nutrient names mapped to your values, including units, such as `"Protein": "YOUR VALUE g"`.
- `nutrition.source`: an optional source name or URL.

Save the file and rerun/refresh the app to see the updates. Nutrition values remain tied to the stated serving size; the app does not estimate fruit weight.

## Training and evaluation

The original dataset has 700 images per class for training, 150 per class for validation, and 150 per class for testing. The training script preserves those folders and excludes repeated decoded images from later occurrences (priority: train, validation, test); no source images are modified or deleted. Conflicting labels on identical images cause an explicit error. The audit checks exact duplicates, not related photos or near duplicates.

Images are oriented using EXIF metadata, converted to RGB, and padded to 96 × 96 with aspect ratio preserved. Training adds horizontal flips, small affine transformations, and mild color changes. Inference uses the same deterministic preprocessing without augmentation.

The CNN has four convolution blocks (16, 32, 64, and 96 channels), batch normalization, ReLU, max pooling, and a small classifier with dropout. All weights start randomly. AdamW updates weights only on training images. Validation loss controls learning-rate reduction, checkpoint selection, and early stopping. The test split is evaluated after training finishes.

```powershell
python train.py --epochs 25 --batch-size 32 --patience 7 --threads 4 --seed 42
```

The script uses CUDA if available, otherwise CPU. The saved checkpoint loads on CPU for the app. Seeded runs improve reproducibility; exact results may vary across hardware and package versions.

Generated artifacts:

- `artifacts/fruit_cnn.pt`: selected trained weights, class order, image size, seed, and epoch.
- `artifacts/metrics.json`: validation/test metrics and confusion matrices (rows = true, columns = predicted; apple then banana).
- `artifacts/history.json`: loss and accuracy for each training epoch.
- `artifacts/dataset_audit.json`: retained-image manifest and excluded duplicates.

This is a two-class image classifier, not an object detector or a general fruit recognizer. It will choose apple or banana even for unrelated objects; low confidence is not a reliable unknown-object detector. Dataset accuracy does not guarantee performance on different photos or backgrounds.

## Checks

```powershell
python -m unittest discover -s tests -v
```
