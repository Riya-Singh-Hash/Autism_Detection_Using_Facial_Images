
# 🧠 Autism Detection Using Facial Images

## 📌 Overview

This project presents a **deep learning–based system for Autism Spectrum Disorder (ASD) detection using facial images**. A **Convolutional Neural Network (CNN)** is trained to learn discriminative facial patterns associated with ASD and classify images into **Autistic** and **Non-Autistic** categories.

The project also includes a **Flask-based web application** that allows users to upload an image and receive a prediction in real time.



## ✨ Key Features

* Binary classification: **Autistic vs Non-Autistic**
* Custom CNN architecture using **TensorFlow / Keras**
* Image preprocessing and normalization
* Performance evaluation using **Accuracy, Precision, and Recall**
* **Flask web interface** for real-time predictions
* Modular, scalable, and well-structured codebase



## 📂 Dataset 
The project uses the Autism Image Dataset available on Kaggle:

Dataset Source: Autism Image Data on Kaggle(https://www.kaggle.com/datasets/cihan063/autism-image-data)

The dataset is organized into **training, validation, and test splits** with balanced classes.

### 📊 Dataset Distribution

| Split          | Autistic | Non-Autistic | Total    |
| -------------- | -------- | ------------ | -------- |
| **Training**   | 1270     | 1270         | **2540** |
| **Validation** | 50       | 50           | **100**  |
| **Test**       | 150      | 150          | **300**  |
| **Overall**    | **1470** | **1470**     | **2940** |

✔ Dataset is **perfectly balanced**, reducing class bias

✔ Images are organized using directory-based class labels



## 🧪 Model Architecture

The model follows a **Sequential CNN architecture**:

* Convolutional layers with **ReLU activation**
* **MaxPooling** layers for dimensionality reduction
* **Batch Normalization** for training stability
* **Fully Connected Dense layer**
* **Dropout** for regularization
* **Sigmoid output layer** for binary classification

**Total Parameters**: **3,697,905**
**Trainable Parameters**: **3,697,265**



## 🏋️ Training Configuration

* **Optimizer**: Adam
* **Loss Function**: Binary Crossentropy
* **Input Image Size**: 256 × 256 × 3
* **Epochs**: Up to 50 (Early Stopping applied)
* **Learning Rate Scheduler**: ReduceLROnPlateau

Training automatically stops when validation performance stops improving.



## 📊 Model Performance (Test Set)

| Metric        | Value      |
| ------------- | ---------- |
| **Accuracy**  | **81.37%** |
| **Precision** | **82.69%** |
| **Recall**    | **81.13%** |

These results demonstrate **good generalization** and balanced performance across both classes.



## 🌐 Web Application

The Flask web app allows users to:

1. Upload a facial image
2. Automatically preprocess it
3. Perform inference using the trained CNN
4. Display the predicted class (**Autistic / Non-Autistic**)

The UI also includes an **autism awareness questionnaire** and a **basic chatbot** for user interaction.



## 📁 Project Structure

```
Autism-Detection/
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
│   └── trained_model.h5
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── style.css
├── static/
│   └── uploads/
├── notebooks/
│   └── data_analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```



## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ShalabhRanjan19/Autism-Detection-using-Image.git
cd Autism-Detection
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare dataset

Place the dataset inside the `data/` directory following the structure shown above.

---

## ▶️ Usage

### Train the model

```bash
python src/train_model.py
```

### Run prediction script

```bash
python src/predict.py
```

### Launch web application

```bash
python app.py
```

Open browser:

```
http://localhost:5000
```



## ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.
It is **not a medical diagnostic tool** and should not replace professional clinical evaluation.




## ⭐ Acknowledgment

If you find this project useful, please consider giving it a ⭐ on GitHub.


