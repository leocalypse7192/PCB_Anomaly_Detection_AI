# PCB Anomaly Detection using YOLOv5
This repository contains a cleaned, modular version of the **Google Colab**
pipeline used for PCB anomaly detection (object detection for PCB defects)
trained with **YOLOv5**.
## What is included
- `notebooks/pcb_anomaly_detection_colab.ipynb` — the original Colab notebook
(full pipeline)
- `src/` — modular scripts for preprocessing, splitting, label conversion and
visualization
- `data/` — instructions to download the Kaggle dataset and example `data.yaml`
- `requirements.txt` — Python dependencies
## Dataset
- Dataset source (Kaggle): https://www.kaggle.com/datasets/akhatova/pcb-defects
- Classes: `spurious_copper`, `mouse_bite`, `open_circuit`, `missing_hole`,
`spur`, `short`

## Quick start (run locally or on Colab)
1. **Clone the repo**
```bash
git clone https://github.com/FarhanMS7192/PCB_Anomaly_Detection_AI
cd pcb-anomaly-detection
```
2. **Install dependencies (recommended: create a virtual environment)**
```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Install PyTorch separately as per your machine (CPU or CUDA) — see
# https://pytorch.org/get-started/locally/
```
3. **Preprocess (resize images & annotations)**
```bash
python src/preprocess.py --images_dir "path/to/raw/images" --ann_dir "path/
to/raw/annotations" --out_dir data/processed --size 640
```
4. **Split dataset into train/val**
```bash
python src/split_dataset.py --processed_dir data/processed --out_dir data/
split --train_ratio 0.8
```
5. **Convert Pascal VOC XML to YOLO format**
```bash
python src/convert_to_yolo.py --annotations_dir data/split/annotations --
images_dir data/split/images --out_labels_dir data/split/labels
```
6. **Train YOLOv5**
```bash
Clone the YOLOv5 repo and run training from its folder (instructions below). Use the included data/
data.yaml (edit paths) for training.
```
7. **Run detection & visualize results**
```bash
# Run YOLOv5 detect.py (from the yolov5 folder)
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf
0.5 --source path/to/test_images --project runs/detect --name pcb_detect
# Visualize a saved result (side-by-side)
python src/detect_results.py --original_dir path/to/test_images --
result_dir runs/detect/exp
```
## How to run on Google Colab
If you prefer Colab, upload the notebook notebooks/pcb_anomaly_detection_colab.ipynb
(or open it from your Drive). The notebook contains the step-by-step pipeline from your original
implementation.
