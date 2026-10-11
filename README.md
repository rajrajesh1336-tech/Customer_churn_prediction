# 📊 Churn Prediction App (Telco Customer Churn)
---
## 🚀 Project Overview
---
- This project focuses on predicting customer churn for a telecom company using Machine Learning. The goal is to identify customers who are likely to leave the service so that proactive retention strategies can be applied.

**The project follows a complete Data Science workflow:**

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training & Evaluation
- Deployment using Streamlit
---
## 🌐 Live App
👉 https://cc-prediction1.streamlit.app/

---
## 📂 Project Structure
---
```text
Customer_churn_predection/
│
├── Data/
│   ├── Telco_Customer_Churn.csv
│   ├── cleaned_data.csv
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│
├── SRC/
│
├── venv/
├── Model_training_Completed.ipynb
├── requirements.txt
├── best_model.pkl
├── app.py
```
---
## ⚙️ Tech Stack
- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit (Deployment)
---
🧠 Machine Learning Workflow

---
### 1. Data Collection
- Telco Customer Churn dataset
### 2. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
### 3. Exploratory Data Analysis (EDA)
- Customer demographics
- Service usage patterns
- Churn distribution
### 4. Model Training
- Multiple models tested
- Best model selected and saved (best_model.pkl)
### 5. Deployment
- Streamlit app for real-time prediction
---
▶️ How to Run Locally

---
### 1. Clone the repository
- git clone <your-repo-link>
- cd churn_prediction
### 2. Create virtual environment
- python -m venv venv
- venv\Scripts\activate   # Windows
### 3. Install dependencies
- pip install -r requirements.txt
### 4. Run the app
- streamlit run app.py
---
📌 Features
---
- User-friendly UI
- Real-time churn prediction
- Supports multiple input features
- Scalable ML pipeline
## ⚠️ Note
- Dataset is not included in the repository.
- Please update the data path accordingly if running locally.
