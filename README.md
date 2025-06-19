# Voice-Based Machine Learning Model for Early Diagnosis of   Parkinson’s Disease.


This project implements a step-by-step machine learning pipeline to detect Parkinson’s Disease using biomedical voice measurements. The model is trained using a Support Vector Machine (SVM) classifier and visualizes results with various plots.

## 🧠 About the Project

- Model: **SVM with linear kernel**
- Dataset: Biomedical voice measurements from people with and without Parkinson’s Disease
- Features: 24 voice measurements parameters
- Target: `status` (1 = Parkinson’s, 0 = Healthy)

## 📄 Main Code

📂 File: [`step_by_step_code.ipynb`](step_by_step_code.ipynb)

This notebook includes:
- Data preprocessing
- Feature scaling
- Model training & evaluation
- Prediction for new data
- Visualizations: class distribution, heatmap, accuracy bar, prediction, and confusion matrix

## With the Accuracy of:

-Training Accuracy: 0.8846153846153846 <br>
-Testing Accuracy: 0.8717948717948718<br>

## 📊 Visualizations

- Class distribution plot
- Feature correlation heatmap
- Accuracy comparison (train vs test)
- Confusion matrix
- Output prediction bar (for new user data)

## 📁 Project Structure

├── Dataset/<br>
│ └── dataset.csv<br>
├── step_by_step_code.ipynb<br>
├── README.md<br>
├── requirements.txt<br>
├── full_code.py<br>
├── voice_based_parkinson's_disease_detection.pdf<br>