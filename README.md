# Multi-Disease Prediction System

## Introduction

This project aims to predict multiple diseases, including **Brain Tumor**, **Heart Disease**, and **Chronic Kidney Disease**. Various classification algorithms, such as **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, **Random Forest**, **Logistic Regression**, and **Naive Bayes**, were applied to detect these diseases. The accuracy of each model was assessed and compared to determine the most effective one. The best-performing model for each disease was integrated into a **web application** to allow users to input relevant data and receive predictions.

## Workflow

The workflow involved several stages:
1. **Data Collection**: Sourced datasets from platforms like Kaggle.
2. **Data Preprocessing**: Applied preprocessing techniques specific to each dataset.
3. **Model Training**: Trained and evaluated different machine learning models to find the best-performing one for each disease.
4. **Model Deployment**: Created a unified web application using **Streamlit** for easy user interaction and prediction.

## Disease Prediction Systems

### 1. Chronic Kidney Disease Prediction System

**Data Collection**: The dataset `kidney_disease.csv` from Kaggle was used, containing 25 attributes (11 numerical, 14 nominal) like age, blood pressure, specific gravity, hemoglobin, etc.

**Data Preprocessing**: Missing values were handled, categorical variables were label-encoded, and numerical features were normalized.

**Exploratory Data Analysis (EDA)**: Key insights included:
- Age distribution, specific gravity, and albumin levels provided useful indicators.
- Density plots helped reveal patterns across different classes.

**Models Used**:
- **KNN**
- **Random Forest**
- **Logistic Regression**
- **SVM**

The **Random Forest** model achieved the highest testing accuracy at 96.25%.

### 2. Heart Disease Prediction System

**Data Collection**: The heart disease dataset from Kaggle was used, containing 14 attributes such as age, gender, chest pain type, resting blood pressure, cholesterol levels, etc.

**Features**: Important features included resting blood pressure, cholesterol, maximum heart rate, and more.

**EDA Findings**:
- Frequency graphs showed balanced data.
- Cholesterol distribution revealed that most values ranged between 200 and 250.

**Models Used**:
- **Logistic Regression**
- **Random Forest**
- **SVM**
- **KNN**

The **Random Forest** model demonstrated the best performance, with an AUC of 0.95, indicating its strong discriminative ability.

## Methodology

- **Machine Learning Models**: Employed KNN, Random Forest, Logistic Regression, and SVM.
- **Evaluation Metrics**: Used confusion matrices, classification reports, and **AUC-ROC** curves to measure performance.
- **Hyperparameter Tuning**: Applied techniques like **GridSearchCV** for optimization.

### Model Comparison (AUC Scores)
- **Logistic Regression**: 0.90
- **SVM**: 0.73
- **KNN**: 0.63
- **Random Forest**: 0.95

## Clinical Implications

A high AUC indicates the model’s ability to distinguish between positive and negative cases. The **Random Forest** model, with an AUC of 0.98 for kidney disease prediction, proved to be the most reliable.

## Results

The **Random Forest** classifier consistently outperformed other models, achieving high accuracy in both kidney and heart disease prediction. This classifier demonstrated robustness, making it ideal for clinical applications.

## Discussion

The **AUC-ROC Curve**:
- **AUC > 0.5**: Better than random prediction.
- **AUC = 1**: Perfect discrimination between classes.

The Random Forest model achieved a high AUC, demonstrating its ability to effectively predict diseases with high confidence.

## Technologies Used
- **Python** (with libraries such as Scikit-Learn, Pandas, NumPy)
- **Streamlit** (for web application)
- **GridSearchCV** (for hyperparameter tuning)
- **Kaggle Datasets** (for data sources)

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/multi-disease-prediction.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Conclusion

This project successfully integrates machine learning models for disease prediction into an accessible web application.



---

Feel free to contribute or report issues!

