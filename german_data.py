import pandas as pd

# 1. Define the column names (The raw data doesn't have them)
columns = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 
    'savings_status', 'employment', 'installment_rate', 'personal_status', 'other_parties', 
    'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing', 
    'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

# 2. Load the raw data from UCI (it is space-separated)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, sep=' ', names=columns)

# 3. Save as a standard Comma-Separated Values (CSV) file
df.to_csv("german_credit_data.csv", index=False)

print("✅ Success! 'german_credit_data.csv' has been created.")
print(f"File contains {df.shape[0]} rows and {df.shape[1]} columns.")