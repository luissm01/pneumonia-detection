# Pneumonia Detection and Interpretability using CNNs

This project demonstrates how to build, train, and interpret a convolutional neural network (CNN) for pneumonia detection using chest X-ray images. It includes preprocessing, model training, evaluation, and visual explanation using Class Activation Maps (CAM).

## 📁 Project Structure

```
pneumonia-detection-cam/
│
├── notebooks/           # Jupyter notebooks for each step
│   ├── 01-Preprocess.ipynb
│   ├── 02-Train-Original.ipynb
│   ├── 02-Train-Balanced.ipynb
│   ├── 03-Interpretability.ipynb
│
├── scripts/             # Python scripts for training
│
├── data/                # Input data and processed .npy files
│
├── weights/             # Saved model weights (.pth) and checkpoints
│
├── metrics/             # CSV files with training and validation metrics
│
├── images/              # CAM visualizations and other figures
│
├── README.md            # Project overview
├── requirements.txt     # Minimum Python dependencies
├── requirements_dev.txt # Exact environment dependencies
└── .gitignore
```

## 🧪 Notebooks Overview

- **01-Preprocess**: Loads and converts DICOM images to normalized `.npy` arrays.
- **02-Train-Original**: Trains a CNN on the dataset without class balancing.
- **02-Train-Balanced**: Uses `WeightedRandomSampler` to balance training samples.
- **03-Interpretability**: Applies Class Activation Maps (CAM) to visualize model decisions.

## 🔧 How to Use

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pneumonia-detection-cam.git
cd pneumonia-detection-cam
```

2. Install dependencies (minimal setup):
```bash
pip install -r requirements.txt
```

To exactly replicate the development environment:
```bash
pip install -r requirements_dev.txt
```

3. Run the notebooks in order or execute training scripts:
```bash
python scripts/train_balanced.py
```

4. View visualizations in `notebooks/03-Interpretability.ipynb`.

## 🎯 Model Interpretability

This project uses Class Activation Maps (CAM) to generate heatmaps that highlight the most important regions of a chest X-ray used for the model’s decision. This is essential in medical applications to ensure transparency and trust.



## 📦 Data Preparation

The dataset used for this project is based on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), which provides labeled chest X-ray DICOM images.

To preprocess the data:

1. Download the dataset from the Kaggle competition page (requires account and acceptance of terms).
2. Place the files in a suitable `data/rsna-pneumonia-detection-challenge/` directory.
3. Run the notebook:

```bash
notebooks/01-Preprocess.ipynb
```

This notebook will convert the DICOM files into 224x224 grayscale NumPy arrays (`.npy`) and organize them into a structure like:

```
data/Processed/train/0/
data/Processed/train/1/
data/Processed/val/0/
data/Processed/val/1/
```

These `.npy` files are used as input for the training and evaluation phases.
## 👤 Author

Luis Sánchez Moreno – Biomedical Engineer specialized in AI for medical imaging.
