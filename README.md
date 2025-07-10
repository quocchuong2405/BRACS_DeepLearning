# 🧠 Detection of Regions of Interest (RoIs) in Breast Cancer Biopsy using Deep Learning

This repository contains our STAT3007 group project focused on leveraging deep learning to detect Regions of Interest (RoIs) in Hematoxylin and Eosin (H&E) stained breast tissue biopsy images. The goal is to assist in more accurate and efficient diagnosis of breast cancer by reducing observer variability using segmentation models trained on the BRACS dataset.

---

## 📌 Project Overview

Breast cancer diagnosis is traditionally dependent on pathologist interpretation of H&E slides, a process that can be time-consuming and subjective. To address this, we developed and tested deep learning models capable of identifying key RoIs automatically. This project explores U-Net and its variants using pretrained encoders like ResNet50 and DenseNet121.

---

## 📂 Dataset: BRACS

- **547 Whole Slide Images (WSIs)**
- **4539 RoIs** from 189 patients
- 7 lesion types: Normal, Benign, UDH, FEA, ADH, DCIS, IC
- High-resolution `.svs` and `.png` files
- Expert pathologist annotations

---

## 🧠 Models Implemented

### 1. Vanilla U-Net
Standard encoder-decoder architecture for semantic segmentation.

### 2. U-Net with ResNet50 Encoder
Pretrained on ImageNet via transfer learning.

### 3. U-Net with DenseNet121 Encoder
DenseNet was first trained as a tile classifier and then repurposed as an encoder in a U-Net framework.

### 4. DenseNet121 Classifier
Standalone tile-level binary classifier to distinguish UDH vs non-UDH regions.

---

## ⚙️ Methodology

- **Tiling:** WSIs split into 512x512 patches
- **Mask Generation:** Using QuPath + GeoJSON + rasterization
- **Filtering:** Remove white background tiles via RGB std deviation threshold
- **Augmentation:** Random horizontal & vertical flips
- **Normalization:** Using ImageNet mean and std
- **Training Environment:** Google Colab Pro, UQ HPC (Bunya)

---

## 📊 Key Results

| Model                     | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| **Vanilla U-Net**        | ~0.87    | ~0.23     | ~0.00  | ~0.00    |
| **U-Net + ResNet50**     | ~0.17    | ~0.18     | ~0.99  | ~0.30    |
| **DenseNet Classifier**  | ~0.87    | ~0.55     | ~0.82  | ~0.56    |

- U-Net failed to detect masked regions (class imbalance).
- ResNet50 model overfit to positive class (high recall, poor precision).
- DenseNet classifier showed promising results for classification but not segmentation.

---

## 🚧 Limitations

- Limited access to high-end GPUs delayed training
- Severe class imbalance in RoIs vs non-RoIs
- Small dataset size for segmentation task
- Poor generalization of U-Net variants despite augmentation

---

## 🔮 Future Directions

- Incorporate spatial transcriptomics for multimodal segmentation
- Apply focal loss or Dice loss to handle class imbalance
- Explore unsupervised domain adaptation (e.g., self-training)
- Extend dataset with more WSIs for training

---

## 👨‍💻 Team Members

- **Nguyen Van Quoc Chuong**
- Jenlarp Jenlarpwattanakul
- Wilson Cheng-Han Wang
- Clancy Heyworth

---

## 📄 Report

See full report here: [`STAT3007_Group_Project.pdf`](./STAT3007_Group_Project__Copy_.pdf)

---

## 📚 References

Key references include:
- Brancati et al., *BRACS Dataset*, Database (Oxford), 2022
- Ronneberger et al., *U-Net*, MICCAI, 2015
- Patil et al., *Generating RoIs for Breast Cancer*, COMPSAC, 2020
- Iakubovskii, *Segmentation Models PyTorch*, GitHub, 2019

For the full list, please refer to the final section of the report.
