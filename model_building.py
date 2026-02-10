import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib

# --- 1. Load Dataset ---
# Loading directly from UCI to ensure reproducibility
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
names = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 
         'savings_status', 'employment', 'installment_rate', 'personal_status', 'other_parties', 
         'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing', 
         'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class']
data = pd.read_csv(url, sep=' ', names=names)

# --- 2. Preprocessing ---
# The dataset uses '1' for Good and '2' for Bad. We convert this to 0 (Good) and 1 (Bad) for standard binary classification.
data['class'] = data['class'].map({1: 0, 2: 1})

# Encode Categorical Variables
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Split Features and Target
X = data.drop('class', axis=1)
y = data['class']

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features (Important for KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Model Initialization ---
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# --- 4. Training & Evaluation ---
results = {}

print(f"{'Model':<25} | {'Acc':<5} | {'AUC':<5} | {'Prec':<5} | {'Recall':<5} | {'F1':<5} | {'MCC':<5}")
print("-" * 75)

for name, model in models.items():
    # Train
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    
    # Store results for later use (e.g., in a table)
    results[name] = [acc, auc, prec, rec, f1, mcc]
    
    print(f"{name:<25} | {acc:.3f} | {auc:.3f} | {prec:.3f} | {prec:.3f} | {f1:.3f} | {mcc:.3f}")
    
    # Save the model
    filename = f"model/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)

# Save the scaler as well (needed for app)
joblib.dump(scaler, 'model/scaler.pkl')
print("\nModels and scaler saved successfully to 'model/' directory.")