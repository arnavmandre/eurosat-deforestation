# README outline – for you to write

Write the real `README.md` (in the repo root) **in your own words**, section by section, using the prompts below. Rough is fine; I'll review each section when you're done.

Tip: a good README is short. Aim for something a recruiter can skim in 2 minutes. Use headings, short paragraphs and 3–4 images.

---

## 1. Title + one-line summary
- A clear title (e.g. "Detecting deforestation from space with a CNN" — make up your own).
- One sentence: what the project does, where, and with what.

## 2. Why this matters (3–4 sentences)
- Why deforestation monitoring matters.
- Why Hasdeo Arand: one of central India's largest dense forests, partly cleared for open-pit coal mining.
- What question you asked: can a lightweight model trained on European images detect forest loss in India?

## 3. How it works (a short numbered list)
Name each step in one line. The facts:
1. **Data:** EuroSAT RGB – 27,000 Sentinel-2 images (64×64 px, 10 m/pixel), 10 land-cover classes.
2. **Model:** ResNet50 pretrained on ImageNet, last layer replaced with 10 classes. Trained in 2 stages: last layer only (5 epochs, lr 0.001), then whole network (5 epochs, lr 0.0001). Batch 64, mixed precision, RTX 5060 laptop GPU.
3. **Study area:** 16 × 16 km around the PEKB mine (22.838° N, 82.814° E). Sentinel-2 L2A images from 20 Feb 2019 and 8 Feb 2025.
4. **Change detection:** cut each image into 625 patches (640 m), classify each, flag patches that went Forest → something else.
5. **Validation:** compared against Global Forest Watch (Hansen et al., v1.13) tree cover loss for 2019–2024.
6. **Second test area:** Talabira coal mine, Odisha, with all settings fixed in advance.

## 4. Results
Put these in a small table, then 1–2 sentences on what they mean.

| | Value |
|---|---|
| EuroSAT test accuracy | 97.70% (2,700 test images) |
| Best / worst class | Forest F1 99.5% / PermanentCrop F1 95.3% |
| Most common mix-up | HerbaceousVegetation → PermanentCrop (12) |
| Hasdeo – GFW loss 2019–24 | 2.09 km² |
| Hasdeo – method A (label change) | precision 47%, recall 82% |
| Hasdeo – method B (prob. drop ≥ 0.5) | precision 53%, recall 82% *(cut-off tuned here)* |
| Talabira – method A | precision 50%, recall 75% |
| Talabira – method B | precision 55%, recall 75% *(cut-off fixed in advance)* |
| Area estimate (Hasdeo) | 7.8 km² vs GFW 2.09 km² (3.7× overestimate) |

Suggested images (all in `figures/`):
- `hasdeo_2019_vs_2025.png` – before / after
- `validation_map_simple.png` or `validation_map_comparison.png` – agreement with GFW
- `confusion_matrix_percent.png` – classifier errors
- `hasdeo_vs_talabira.png` – the second-area test

Markdown for an image: `![Alt text](figures/hasdeo_2019_vs_2025.png)`

## 5. What I learned / limitations (bullet list, your words)
Things you discovered — pick the ones you can explain:
- Domain shift: EuroSAT is bluish European forest; Hasdeo is brown dry tropical forest. Colour-matching made it *worse*.
- No "mine" class: the model called the mine HerbaceousVegetation / SeaLake / Highway.
- Hill shadows → "SeaLake" false alarms (and the model was *confident* about them).
- Clouds and cloud shadows → false alarms at Talabira.
- Mixed patches: 640 m patches overestimate cleared area ~3.7×.
- Confidence filtering helped on EuroSAT but not in India.
- Tuning a cut-off on the same data you score it on is flattering → tested on Talabira.

## 6. How to run it
- Requirements: Python 3, NVIDIA GPU (optional but faster), ~3 GB disk.
- Steps: create the venv, install packages (list them: torch, torchvision, jupyterlab, scikit-learn, matplotlib, seaborn, pillow, tqdm, ipywidgets, rasterio, pystac-client), download EuroSAT RGB to `C:/data/EuroSAT`, run notebooks 01 → 06 in order.
- A one-line description of each notebook.

## 7. Credits and AI use
- Data: EuroSAT (Helber et al., 2019), Sentinel-2 (ESA Copernicus, via Element 84 Earth Search), Global Forest Watch / Hansen et al. (2013).
- Model: ResNet50 (He et al., 2016), PyTorch.
- **AI use (be honest):** say that the code and notebooks were written with an AI coding assistant (Claude) under your direction, and that you ran, checked and interpreted the results.

## 8. Next steps (2–3 bullets)
Ideas: use a Feb 2026 image to capture 2025 loss; add a "mine/bare ground" class; use all 13 Sentinel-2 bands; smaller patches; mask clouds; train on Indian imagery.
