import pandas as pd
import torch


df = pd.DataFrame(
    {
        "Company": [
            "Microsoft",
            "Apple",
            "Klarna",
            "Tesla",
            "Amazon",
        ],
        "reporting_year": [2024, 2024, 2024, 2024, 2024],
        "focus": ["Carbon Negative 2030", "Carbon Neutral 2030", "Carbon Neutral 2040", "Carbon Negative 2040", "Carbon Neutral 2050"],
        "sustainability_score": [95, 90, 85, 90, 80],
    }
)

print (df)

# Copying the dataframe to keep the original clean
df_encoded = df.copy()

# 1. Extract focus type (Negative/Neutral) and year from focus column
df_encoded['focus_type'] = df_encoded['focus'].str.extract(r'(Carbon Negative|Carbon Neutral)')
df_encoded['target_year'] = df_encoded['focus'].str.extract(r'(\d{4})').astype(int)

# 2. Convert Company to category codes
df_encoded['Company'] = df_encoded['Company'].astype('category').cat.codes

# 3. Normalize year and score
df_encoded['reporting_year'] = df_encoded['reporting_year'] - 2024  # makes them all 0
df_encoded['target_year'] = (df_encoded['target_year'] - 2024) / 10.0  # normalize year difference
df_encoded['sustainability_score'] = df_encoded['sustainability_score'] / 100.0 

print("\nEncoded DataFrame:")
print(df_encoded)

# Separate features (X) and target (y)
X_columns = ['Company', 'reporting_year', 'focus_type', 'target_year']
y_column = 'sustainability_score'

# Convert focus_type to category codes
df_encoded['focus_type'] = df_encoded['focus_type'].astype('category').cat.codes

# Convert to PyTorch Tensors
X_tensor = torch.tensor(df_encoded[X_columns].values, dtype=torch.float32)
y_tensor = torch.tensor(df_encoded[y_column].values, dtype=torch.float32).unsqueeze(1)

print("\nPyTorch X Tensor:")
print(X_tensor)
print("\nPyTorch y Tensor:")
print(y_tensor)
