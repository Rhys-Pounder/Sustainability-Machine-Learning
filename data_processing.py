import pandas as pd
import torch

def get_prepared_data():
    df = pd.DataFrame({
        "Company": [
            "Stripe", "Square", "Revolut", "Monzo", "Chime", 
            "N26", "Starling Bank", "PayPal", "Wise", "SoFi",
            "Klarna", "JPMorgan Chase", "Citibank", "Barclays", "HSBC"
        ],
        "reporting_year": [2024] * 15,
        "focus": [
            "Carbon Negative 2025", 
            "Carbon Neutral 2025", 
            "Carbon Negative 2030", 
            "Carbon Neutral 2030", 
            "Carbon Negative 2035", 
            "Carbon Neutral 2035", 
            "Carbon Negative 2040", 
            "Carbon Neutral 2040", 
            "Carbon Negative 2045", 
            "Carbon Neutral 2045",
            "Carbon Neutral 2030",
            "Carbon Negative 2050",
            "Carbon Neutral 2050",
            "Carbon Negative 2030",
            "Carbon Neutral 2035"
        ],
        "sustainability_score": [
            99, 95, 89, 82, 75, 
            68, 62, 58, 55, 53,
            82, 51, 50, 89, 68
        ],
    })

    df_encoded = df.copy()

    # Extract features
    df_encoded['focus_type'] = df_encoded['focus'].str.extract(r'(Carbon Negative|Carbon Neutral)')
    df_encoded['target_year'] = df_encoded['focus'].str.extract(r'(\d{4})').astype(int)

    # Convert to categorical codes
    df_encoded['Company'] = df_encoded['Company'].astype('category').cat.codes
    df_encoded['focus_type'] = df_encoded['focus_type'].astype('category').cat.codes

    # Normalize
    df_encoded['reporting_year'] = df_encoded['reporting_year'] - 2024
    df_encoded['target_year'] = (df_encoded['target_year'] - 2024) / 10.0
    df_encoded['sustainability_score'] = df_encoded['sustainability_score'] / 100.0 

    X_columns = ['focus_type', 'target_year']
    y_column = 'sustainability_score'

    X_tensor = torch.tensor(df_encoded[X_columns].values, dtype=torch.float32)
    y_tensor = torch.tensor(df_encoded[y_column].values, dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor, X_columns

if __name__ == "__main__":
    X, y, cols = get_prepared_data()
    print("Data Preprocessing Ready.")

