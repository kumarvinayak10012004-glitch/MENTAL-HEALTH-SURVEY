# 🧠 Predicting Depression from Mental Health Survey Data using Deep Learning

## 📌 Project Overview
Mental health disorders like depression are often underdiagnosed due to lack of awareness and timely screening.  
This project aims to build a **Deep Learning–based predictive system** that identifies individuals at risk of depression using **mental health survey data**.

The solution leverages **PyTorch**, **data preprocessing pipelines**, and a **Streamlit web application** to deliver real-time predictions while ensuring **fairness and bias evaluation** across diverse demographic groups.

---

## 🎯 Problem Statement
The objective of this project is to predict whether an individual may experience **depression** based on:
- Demographic information  
- Lifestyle choices  
- Medical and family history  
- Behavioral and sleep patterns  

The model must:
- Handle real-world healthcare data challenges
- Address bias and fairness issues
- Provide reliable and explainable predictions

---

## 🏥 Domain
**Mental Health | Healthcare AI | Deep Learning**

---

## 💼 Business Use Cases

### 🏥 Healthcare Providers
- Early identification of patients at risk of depression
- Enable timely medical intervention and preventive care

### 🧑‍⚕️ Mental Health Clinics
- Assist clinicians in data-driven treatment planning
- Prioritize high-risk individuals

### 🏢 Corporate Wellness Programs
- Monitor employee mental health trends
- Proactively offer mental health support

### 🌍 Government & NGOs
- Identify vulnerable population groups
- Allocate mental health resources efficiently

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | AWS (EC2 / Elastic Beanstalk) or Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 📊 Dataset Description
- **Source:** Mental Health Survey Data  
- **Format:** CSV / Excel  
- **Features Include:**  
  - Age  
  - Gender  
  - Lifestyle factors (sleep, physical activity)  
  - Medical & family history  
  - Stress levels, work-life balance  
- **Target Variable:**  
  - Binary classification  
    - `1` → Depression  
    - `0` → No Depression  

---

## 🔄 Data Preprocessing Steps
- Handle missing values (imputation / removal)
- Encode categorical variables (One-Hot Encoding)
- Normalize numerical features
- Train–test split
- Bias-aware preprocessing

---

## 🤖 Model Architecture
- **Model Type:** Multilayer Perceptron (MLP)
- **Framework:** PyTorch
- **Layers:**
  - Input Layer
  - Hidden Layers with ReLU activation
  - Output Layer with Sigmoid activation
- **Loss Function:** Binary Cross-Entropy Loss
- **Optimizer:** Adam

---

## 📈 Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Bias & Fairness Evaluation (across gender, age, etc.)

---

## 🔁 Pipeline Design

### 🔹 Data Pipeline
- Data loading
- Preprocessing
- Feature transformation

### 🔹 Model Training Pipeline
- Training
- Validation
- Evaluation
- Model saving (`.pth` file)

---

## 🌐 Streamlit Application
- User-friendly interface for real-time prediction
- Accepts user inputs such as:
  - Age
  - Gender
  - Lifestyle habits
  - Medical history
- Displays:
  - Depression prediction result
  - Risk indication

---

## ☁️ Deployment
- **Option 1:** AWS EC2 / Elastic Beanstalk
- **Option 2:** Streamlit Cloud  
- Deployed application allows real-time access for testing and usage

---

## 📁 Project Structure

├── data/
│ └── mental_health_survey.csv
├── notebooks/
│ └── EDA_and_Model_Training.ipynb
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ ├── train.py
│ └── evaluate.py
├── app.py # Streamlit app
├── model/
│ └── depression_model.pth
├── requirements.txt
└── README.md

## 👤 Author
**Vinayak Kumar**  
_Data Science | Machine Learning | Deep Learning_

---

⭐ If you like this project, don’t forget to **star the repository**!
