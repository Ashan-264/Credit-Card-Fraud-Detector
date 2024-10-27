# Credit Card Fraud Detection App

This project provides a **fraud detection application** built using `Streamlit`, `pandas`, `scikit-learn`, and `OpenAI`. The application predicts the probability of fraudulent transactions using multiple machine learning models and offers detailed explanations to guide bank officials in decision-making. Additionally, the app generates **customer-friendly emails** to alert users about unusual activities.

---

## Features

- **Transaction Fraud Detection**:  
  Predicts fraud probabilities using models like **Decision Tree**, **XGBoost**, and **Logistic Regression**.
- **Multi-Model Support**:  
  Aggregates predictions from multiple models to calculate an **average probability**.
- **Interactive Interface**:  
  Allows users to **select transactions**, input or modify transaction details, and view predictions and visualizations.
- **Automated Explanations**:  
  Provides detailed fraud explanations using **OpenAI’s API**.
- **Customer Email Generation**:  
  Creates **personalized emails** to inform customers about potential fraud and suggested actions.

---

## Prerequisites

- **Python 3.8 or later**  
- **API Key for OpenAI** (stored as an environment variable `GROQ_API_KEY`)
- Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn openai plotly

## 📁 File Structure

- **`app.py`**:  
  The main Streamlit application script.  
- **`dt_model.pkl`**, **`mb_model.pkl`**, **`hgb_model.pkl`**:  
  Serialized machine learning models used for predictions.  
- **`FraudTest.csv`**:  
  Sample dataset with transaction records.  
- **`utils.py`**:  
  Utility functions for creating charts.  

---

## 🚀 Usage

1. **Select a transaction** from the dropdown in the app.  
2. **Modify transaction details**, such as amount, category, latitude, and longitude.  
3. **View the fraud probability** and **explanation of the prediction**.  
4. **Generate an email** to alert the customer about suspicious activity.  
5. **Visualize model probabilities** using interactive charts.  

---

## 📊 Models Used

- **Decision Tree**  
- **XGBoost**  
- **Logistic Regression**  
- **Gradient Boosting**  

