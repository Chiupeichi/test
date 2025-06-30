# Pneumonia Classification with ResNet18

This repository presents a practical implementation of pneumonia detection from chest X-ray images using transfer learning with a pretrained ResNet18 model. It highlights the application of PyTorch for medical image classification in a binary setting (Pneumonia vs. Normal).

## 📌 Project Overview

- **Model**: Pretrained ResNet18 (from `torchvision.models`)
- **Task**: Binary image classification (Pneumonia vs Normal)
- **Framework**: PyTorch
- **Dataset**: [Chest X-Ray Images (Pneumonia) Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Goal**: Build a reliable classifier with good validation/test performance using data augmentation, transfer learning, and model fine-tuning.

## 📁 Dataset Structure

The dataset should be structured as follows:

```
Chest_XRay/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 🛠️ Dependencies

Install the necessary packages with:

```bash
pip install torch torchvision matplotlib tqdm
```

## 📚 Tutorial Sections

1. **Environment Setup**  
   - Load libraries and detect device (GPU/CPU)

2. **Dataset Preparation**  
   - Load and verify Chest X-ray dataset from local path  
   - Visual inspection of image samples

3. **Data Augmentation and Transformation**  
   - Normalize, Resize, RandomHorizontalFlip, etc.

4. **Model Architecture**  
   - Load pretrained ResNet18  
   - Modify final FC layer for binary classification

5. **Training & Evaluation**  
   - Loss: CrossEntropyLoss  
   - Optimizer: Adam  
   - Metrics: Accuracy  
   - Includes training loop with tqdm progress bars

6. **Validation and Testing**  
   - Evaluate model on unseen data  
   - Visualize predictions and errors

## 🖼️ Sample Image

Here is an example of a chest X-ray image used in the dataset:

![Example X-ray Image](sample_image_example.png)

*Note: This is a placeholder. Replace with actual dataset image if available.*

## 🧠 Skills Demonstrated

- Computer Vision (CV) with PyTorch
- Transfer Learning with Pretrained Models
- Data Augmentation Techniques
- Model Evaluation and Debugging
- Clean Code and Modular Notebook Organization

## 📊 Example Results

- Training Accuracy: ~98%
- Validation Accuracy: ~95%
- Test Accuracy: ~93%

> 📎 *These numbers may vary depending on training duration, preprocessing, and hardware.*

## 💼 About the Author

This notebook was designed for educational and professional demonstration purposes, such as inclusion in a resume or GitHub portfolio. It reflects applied deep learning and computer vision expertise using real-world medical imaging data.

---

## 🔗 Acknowledgments

- Dataset: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Model: [PyTorch ResNet18](https://pytorch.org/vision/stable/models.html)

---

## 📎 License

This project is for educational use only.
