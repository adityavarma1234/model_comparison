import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="German Credit Risk Classifier", layout="wide")

st.title("🏦 German Credit Risk Classification")
st.markdown("""
This app predicts whether a bank customer is a **Good** or **Bad** credit risk.
Upload your test dataset (CSV) to evaluate different machine learning models.
""")

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("Model Configuration")
model_options = [
    "Logistic Regression", 
    "Decision Tree", 
    "KNN", 
    "Naive Bayes", 
    "Random Forest", 
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select ML Model", model_options)

# --- HELPER FUNCTION: PREPROCESSING ---
def preprocess_data(df):
    """
    Preprocesses the uploaded dataframe to match the training format.
    """
    data = df.copy()
    
    # 1. Handle Target Variable (if present)
    # The dataset uses 1=Good, 2=Bad. Convert to 0=Good, 1=Bad
    if 'class' in data.columns:
        data['class'] = data['class'].map({1: 0, 2: 1})
        y = data['class']
        X = data.drop('class', axis=1)
    else:
        y = None
        X = data

    # 2. Encode Categorical Variables
    # Note: In a production app, we would load saved encoders. 
    # For this assignment, we re-apply LabelEncoding to simple strings.
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
        
    # 3. Scale Data (Required for Logistic Regression & KNN)
    # Try to load the saved scaler; if not found, skip scaling (with warning)
    try:
        scaler = joblib.load('model/scaler.pkl')
        # identifying columns to scale - usually all features in this specific dataset logic
        # However, to be safe, we apply it to the whole X if dimensions match
        X_scaled = scaler.transform(X)
        # Convert back to dataframe for consistency
        X = pd.DataFrame(X_scaled, columns=X.columns)
    except FileNotFoundError:
        st.warning("⚠️ Scaler file not found. Data will be used unscaled (might affect KNN/Logistic Regression).")
    except Exception as e:
        # If feature mismatch occurs, we proceed without scaling to prevent crash
        pass
        
    return X, y

# --- MAIN APP LOGIC ---

# 1. Dataset Upload [Assignment Requirement]
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        # Using the same separator logic as the training data usually requires ' ' or ','
        # We try standard CSV reading first
        df = pd.read_csv(uploaded_file)
        
        # If the file looks like the raw German dataset (space separated), reload it
        if len(df.columns) < 5: 
             uploaded_file.seek(0)
             names = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 
                     'savings_status', 'employment', 'installment_rate', 'personal_status', 'other_parties', 
                     'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing', 
                     'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class']
             df = pd.read_csv(uploaded_file, sep=' ', names=names)

        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Preprocess
        X_test, y_test = preprocess_data(df)

        # 2. Load Selected Model
        model_filename = f"model/{selected_model_name.replace(' ', '_').lower()}.pkl"
        
        try:
            model = joblib.load(model_filename)
            
            # Make Predictions
            y_pred = model.predict(X_test)
            # Check if model supports predict_proba (SVM sometimes doesn't, but our list does)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred # Fallback

            # 3. Display Metrics [Assignment Requirement]
            if y_test is not None:
                st.write(f"### 📊 Evaluation Metrics: {selected_model_name}")
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)

                # Display in columns
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col1.metric("Precision", f"{prec:.4f}")
                
                col2.metric("AUC Score", f"{auc:.4f}")
                col2.metric("Recall", f"{rec:.4f}")
                
                col3.metric("F1 Score", f"{f1:.4f}")
                col3.metric("MCC Score", f"{mcc:.4f}")

                # 4. Confusion Matrix [Assignment Requirement]
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            else:
                st.info("Uploaded data does not have a 'class' column. Showing predictions only.")
                st.write(y_pred)

        except FileNotFoundError:
            st.error(f"Model file '{model_filename}' not found. Please ensure you have trained and saved the models.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Awaiting CSV file upload. Please upload the German Credit Data test set.")