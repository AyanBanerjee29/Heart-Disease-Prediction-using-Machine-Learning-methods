# Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting the presence of heart disease in patients using various clinical and demographic features from the Heart Disease Dataset available on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). Early and accurate detection of heart disease can save lives and reduce medical costs.

---

## 📂 Dataset Description

The dataset contains 14 features that represent patient health information:

| Feature    | Description |
|-----------|-------------|
| age       | Age of the patient (in years) |
| sex       | Sex (1 = male; 0 = female) |
| cp        | Chest pain type (0–3) |
| trestbps  | Resting blood pressure (in mm Hg) |
| chol      | Serum cholesterol in mg/dl |
| fbs       | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| restecg   | Resting electrocardiographic results (0–2) |
| thalach   | Maximum heart rate achieved |
| exang     | Exercise induced angina (1 = yes; 0 = no) |
| oldpeak   | ST depression induced by exercise relative to rest |
| slope     | The slope of the peak exercise ST segment |
| ca        | Number of major vessels (0–3) colored by fluoroscopy |
| thal      | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) |
| target    | Presence of heart disease (1 = disease; 0 = no disease) |

---

## 🚀 Project Pipeline

### 1. Data Loading & Exploration
- Loaded the dataset using Pandas.
- Checked for missing values and basic dataset information.
- Performed Exploratory Data Analysis (EDA) using visualization libraries like Seaborn and Matplotlib.

### 2. Data Preprocessing
- Handled categorical and numerical features appropriately.
- Scaled numerical features using StandardScaler.
- Verified data distributions and class balance.

### 3. Model Building
- Split the data into training and testing sets (80:20 ratio).
- Trained multiple machine learning models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree Classifier
  - Random Forest Classifier
- Performed hyperparameter tuning using GridSearchCV where applicable.

### 4. Model Evaluation
- Evaluated the models using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC Curve & AUC Score

### 5. Conclusion
- Identified the best-performing model based on evaluation metrics.
- Discussed potential areas for further improvement.

---

## 📝 Results Summary

| Model                  | Accuracy Score |
|-----------------------|---------------|
| Logistic Regression    | XX.XX%        |
| K-Nearest Neighbors    | XX.XX%        |
| Support Vector Machine | XX.XX%        |
| Decision Tree          | XX.XX%        |
| Random Forest          | XX.XX%        |

*Note: Replace `XX.XX%` with the actual results from your notebook.*

---

## 📊 Key Visualizations

- Correlation heatmap between features.
- Distribution plots for key variables.
- Confusion matrix heatmaps for model evaluation.
- ROC Curves to compare classifier performances.

---

## 💡 Future Work

- Apply deep learning models for potential improvement.
- Test the models on real-world clinical data for better generalization.
- Develop an interactive web app using Streamlit or Flask for easy deployment.

---

## 🛠️ Technologies Used

| Tool/Library    | Purpose                     |
|----------------|-----------------------------|
| Python 3.x     | Programming Language         |
| Pandas         | Data Manipulation            |
| NumPy          | Numerical Operations         |
| Scikit-learn   | Machine Learning Models      |
| Matplotlib     | Data Visualization           |
| Seaborn        | Data Visualization           |
| Jupyter Notebook | Interactive Development   |

---


---

## 🔗 Dataset Source

- [Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## ✍️ Author

**Ayan Banerjee**  
MSc Big Data Analytics Student | Machine Learning Enthusiast  
GitHub: [AyanBanerjee29](https://github.com/AyanBanerjee29)

---





