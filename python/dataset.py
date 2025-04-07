import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_normalize_data(csv_path="data/team_stats_with_elo.csv", 
                              output_path="data/team_stats_with_elo_normalized.csv"):
    """
    Load the team stats data from a CSV file, replace missing values in numeric columns with 
    their column means, and normalize all numeric features (excluding the identifiers and target).
    The normalized data is saved to output_path and returned as a DataFrame.

    Parameters:
        csv_path (str): Path to the raw team stats CSV file.
        output_path (str): Path where the normalized CSV file will be saved.

    Returns:
        normalized_data (pd.DataFrame): The normalized DataFrame.
    """
    # Load the data
    data = pd.read_csv(csv_path)
    
    # Replace all missing values in numeric columns with the column mean
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Define identifier columns to exclude from scaling
    identifier_cols = ['meta_data_team_id', 'meta_data_team_name', 'league_id', 'season_id']
    
    # Set the target column (the Elo rating)
    target_col = 'Elo'
    
    target = data[target_col]
    # Select features for scaling: all numeric columns except the identifiers and the target
    features = data.drop(columns=identifier_cols + [target_col])
    
    # Keep only numeric features (dropping any stray non-numeric columns)
    numeric_features = features.select_dtypes(include=['int64', 'float64'])
    
    # Initialize and apply the StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    
    # Convert the scaled features back to a DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, 
                                      columns=numeric_features.columns, 
                                      index=numeric_features.index)
    
    # Combine the identifiers and target back with the scaled features
    normalized_data = data[identifier_cols].copy()
    normalized_data = normalized_data.join(scaled_features_df)
    normalized_data[target_col] = target  # add back the target column
    
    # Save the normalized DataFrame to a CSV file
    normalized_data.to_csv(output_path, index=False)
    
    return normalized_data
