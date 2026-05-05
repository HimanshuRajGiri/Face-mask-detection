# 😷 Face Mask Detection using Custom CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)

## 📌 Overview
This project is an advanced **Face Mask Detection System** built using **Deep Learning**. Originally based on traditional Machine Learning algorithms (like SVM and MLP which capped at ~71% accuracy), this project has been fully upgraded with a custom **Convolutional Neural Network (CNN)** implemented in PyTorch to achieve an outstanding **96.22% Accuracy**.

To make it user-friendly, a beautiful, interactive **Streamlit Web Application** has been integrated. It features **OpenCV-based Face Detection (Haar Cascades)** to automatically detect and crop faces from uploaded photos before passing them to the CNN model for prediction.

## 🚀 Key Features
- **Custom CNN Architecture**: A lightweight 3-block CNN optimized for fast CPU inference.
- **High Accuracy**: Reached **96.22%** Test Accuracy by handling dataset imbalances.
- **OpenCV Integration**: Automatically detects and crops faces from full images to prevent background interference.
- **Interactive Web App**: A modern, glassmorphism-styled Streamlit UI for seamless drag-and-drop image testing.

## 📂 Project Structure
- `app.py`: The main Streamlit web application.
- `train_cnn.py`: The script used to define and train the PyTorch CNN model.
- `mask_cnn_model.pth`: The trained model weights (The Brain).
- `Mask vs No Mask Face Detection.ipynb`: The legacy baseline notebook showing old traditional ML models.
- `requirements.txt`: Project dependencies.

## ⚙️ Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/HimanshuRajGiri/Mask_model.git
cd Mask_model
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Web Application:**
```bash
streamlit run app.py
```
*This will open the web app automatically in your default browser!*

## 📊 Results Comparison
| Model Approach | Test Accuracy | Inference Type |
|----------------|---------------|----------------|
| Legacy ML (SVM / MLP) | ~71% | Traditional ML |
| **Custom CNN (Our Model)** | **96.2%** | **Deep Learning** |

---

## 👨‍💻 Developer

**Himanshu Raj Giri**
- GitHub: [@HimanshuRajGiri](https://github.com/HimanshuRajGiri)

*Developed as a Minor Project. If you find this project helpful, don't forget to leave a ⭐ on the repository!*
