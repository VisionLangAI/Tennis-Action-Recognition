# Enhanced Tennis Action Recognition Using Hybrid CoaT Transformer with Explainable AI

## 📌 Project Overview
This project focuses on **tennis player/action recognition** using a **hybrid CoaT Transformer** model, integrating **deep learning with explainable AI techniques** (Grad-CAM, SHAP). The aim is to classify tennis actions from images and videos while ensuring model interpretability.

## 📂 Dataset
- Dataset: **Tennis Player / Tennis Action Dataset**  
- Location: `/kaggle/input/tennis-playerdataset/Tennis_player/`  
- Structure:
  - `Train/` – training images categorized by player/action classes  
  - `Test/` – testing images categorized by player/action classes  

## ⚙️ Requirements
Install dependencies before running the notebook:

```bash
pip install tensorflow keras matplotlib numpy scikit-learn shap
```

### Core Libraries
- **TensorFlow / Keras** – Model building & training  
- **Matplotlib** – Visualization  
- **NumPy, Pandas** – Data handling  
- **SHAP, Grad-CAM** – Explainable AI  

## 🚀 How to Run
1. Open the Jupyter Notebook:  
   ```bash
   jupyter notebook tennis-player-code.ipynb
   ```
2. Set dataset path (`train_dir`, `test_dir`).  
3. Run cells step by step to:
   - Preprocess images  
   - Build and train the hybrid CoaT Transformer model  
   - Evaluate performance with accuracy, precision, recall, F1-score  
   - Generate explainability plots (Grad-CAM & SHAP).  

## 📊 Model Workflow
1. **Data Preprocessing** – Image resizing, normalization, augmentation  
2. **Model Training** – Hybrid CoaT Transformer for feature extraction & classification  
3. **Evaluation** – Confusion matrix, classification metrics  
4. **Explainability** – Grad-CAM heatmaps, SHAP feature attributions  

## 📈 Results
- Achieved high accuracy in tennis player/action recognition  
- Explainability ensures trust by highlighting decision-making regions  


## 👨‍💻 Author
- **Your Name**  
Department of [Your Department]  
[Your University]
