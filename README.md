# Cricket Analyzer
![Trajactory](https://github.com/user-attachments/assets/fbec3350-e785-4c8d-9b05-9be6d7b32365)


Open-source repository for **Cricket_Analyzer** — vision-based cricket ball detection, trajectory tracking, and **ball speed estimation** (Streamlit app + YOLO).

**Live App:** http://cloud.pakhpc.com:8502/

---

## Highlights
- **Streamlit UI** for uploading cricket videos and running analysis
- **YOLO-based detection** (ball + wicket)
- **Crease / pitch-region handling** to stabilize tracking
- **Kalman filtering + smoothing** for stable trajectories
- **Speed estimation** (m/s and km/h) + plots + Excel export
- **Model conversion utilities** (ONNX → TFLite script included)

---

## Dataset & PLOS ONE compliance

This GitHub repository contains:
- **All source code**
- A **functional subset of the dataset (~1.16 GB)** for quick testing and validation

The **full dataset (~18–20 GB)** used for the complete study is archived on **Zenodo** for long-term preservation and reproducibility.

✅ **Full Dataset DOI (Zenodo):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18607407.svg)](https://doi.org/10.5281/zenodo.18607407)

### What’s included in the full Zenodo archive (recommended for reviewers)
- Raw videos / curated frames used in the paper
- YOLO labels/annotations
- Split structure (train/val/test lists if used)
- Metadata (venue, lighting, camera type, etc.)
- Any CSV/JSON files used to generate plots/tables in the manuscript

---

## Repository structure (matches current repo)

```text
.
├── Testing
│   ├── Inference_Cloud_Model
│   │   ├── best3.pt
│   │   ├── cricket_analyzer.py
│   │   └── onnx_to_tflite.py
│   └── TESTING_Data
│       ├── 7.mp4
│       ├── 8.mp4
│       ├── 9.mp4
│       ├── 10.mp4
│       ├── ...
│       ├── 138.mp4
│       ├── 139.MP4
│       ├── 140.MP4
│       └── ... (more demo videos)
└── Training
    └── Data
        ├── images
        │   ├── train
        │   │   └── red_balls/box_cricket/cemented_pitch/...
        │   └── val
        │       └── red_balls/box_cricket/cemented_pitch/...
        └── labels
            ├── train
            │   └── red_balls/box_cricket/cemented_pitch/...
            └── val
                └── red_balls/box_cricket/cemented_pitch/...
