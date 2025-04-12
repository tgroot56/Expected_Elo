import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_normalize_data(csv_path="data/team_stats_with_elo.csv", 
                            output_path="data/team_stats_with_elo_normalized.csv",
                            apply_pca=False,
                            pca_components=None):
    """
    Load the team stats data from a CSV file, replace missing values, normalize features,
    and optionally apply PCA for dimensionality reduction.

    Parameters:
        csv_path (str): Path to the raw team stats CSV file.
        output_path (str): Path where the normalized CSV file will be saved.
        apply_pca (bool): Whether to apply PCA for dimensionality reduction.
        pca_components (int): Number of PCA components to retain. If None, uses min(n_samples, n_features).

    Returns:
        normalized_data (pd.DataFrame): The normalized DataFrame.
    """
    # Load the data
    data = pd.read_csv(csv_path)
    print("Initial data shape:", data.shape)
    
    # Replace all missing values in numeric columns with the column mean
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Define identifier columns to exclude from scaling
    if 'league_id' not in data.columns:
        identifier_cols = ['meta_data_team_id', 'meta_data_team_name', 'season_id']
    else:
        identifier_cols = ['meta_data_team_id', 'meta_data_team_name', 'league_id', 'season_id']
    
    # Set the target column (the Elo rating)
    target_col = 'Elo'
    
    target = data[target_col]
    # Select features for scaling: all numeric columns except the identifiers and the target
    features = data.drop(columns=identifier_cols + [target_col])
    
    # Keep only numeric features (dropping any stray non-numeric columns)
    numeric_features = features.select_dtypes(include=['int64', 'float64'])
    
    print("Numeric features shape:", numeric_features.shape)
    # Initialize and apply the StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    print("Scaled features shape:", scaled_features.shape)
    
    # Apply PCA if requested
    if apply_pca:
        # Determine number of components
        if pca_components is None:
            pca_components = min(scaled_features.shape[0], scaled_features.shape[1])
        
        # Apply PCA
        pca = PCA(n_components=pca_components)
        transformed_features = pca.fit_transform(scaled_features)
        
        # Get feature names for the PCA components
        feature_names = [f'PC{i+1}' for i in range(pca_components)]
        
        # Create DataFrame with PCA features
        scaled_features_df = pd.DataFrame(
            transformed_features, 
            columns=feature_names,
            index=numeric_features.index
        )
        
        # Print variance information
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Original dimensions: {scaled_features.shape[1]}")
        print(f"Reduced dimensions: {pca_components}")
        print(f"Total explained variance: {explained_variance:.2%}")
    else:
        # Use original scaled features (no PCA)
        scaled_features_df = pd.DataFrame(
            scaled_features, 
            columns=numeric_features.columns, 
            index=numeric_features.index
        )
    
    # Combine the identifiers and target back with the processed features
    normalized_data = data[identifier_cols].copy()
    normalized_data = normalized_data.join(scaled_features_df)
    normalized_data[target_col] = target  # add back the target column
    
    # Save the normalized DataFrame to a CSV file
    normalized_data.to_csv(output_path, index=False)
    
    return normalized_data