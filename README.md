# Voice Based Parkinson's Disease Detection using SVM

This project implements a step-by-step machine learning pipeline to detect Parkinson’s Disease using biomedical voice measurements. The model is trained using a Support Vector Machine (SVM) classifier and visualizes results with various plots.

## 🧠 About the Project

- Model: **SVM with linear kernel**
- Dataset: Biomedical voice measurements from people with and without Parkinson’s Disease
- Features: 24 voice measurements
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

-Training Accuracy: 0.8846153846153846
Testing Accuracy: 0.8717948717948718

## 📊 Visualizations

- Class distribution plot
- Feature correlation heatmap
- Accuracy comparison (train vs test)
- Confusion matrix
- Output prediction bar (for new user data)

## 📁 Project Structure

├── Dataset/
│ └── dataset.csv
├── step_by_step_code.ipynb
├── README.md
├── requirements.txt