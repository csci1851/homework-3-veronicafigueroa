"""Data loader for Homework 3: SVMs"""

import os
import pandas as pd
import numpy as np
import pickle
import urllib.request
import ssl
from typing import Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
import kagglehub


class HW3DataLoader:
    def __init__(self):
        """Initialize data loader with cache directory for datasets"""
        # Create a data directory in the same directory as this file
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def get_aging_data(
        self, 
        pickle_path=None, 
        n_features: Optional[int] = None,
        feature_selection: str = "random",
        specific_features: Optional[List[str]] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Load the GSE139307 genomic aging dataset from pickle file

        This dataset contains methylation data related to aging studies.
        Due to the high dimensionality of the data (21,368 columns), 
        this method provides options to select a subset of features.

        Args:
            pickle_path: Path to the pickle file (optional) if data is already downloaded elsewhere.
            n_features: Number of features (columns) to select. If None, all features are returned.
            feature_selection: Strategy for selecting features:
                - "random": Randomly select n_features columns
                - "variance": Select n_features columns with highest variance
                - "specific": Use the list provided in specific_features
            specific_features: List of column names to select when feature_selection="specific"
            random_seed: Random seed for reproducibility when using random selection

        Returns:
            data_dict: Dictionary containing the genomic aging dataset components with selected features
        """
        # Path to the genomic aging pickle file
        pickle_path = (
            os.path.join(self.data_dir, "GSE139307.pkl")
            if pickle_path is None
            else pickle_path
        )

        # Download dataset if it doesn't exist locally
        if not os.path.exists(pickle_path):
            print("Downloading GSE139307 genomic aging dataset...")
            try:
                # URL for the dataset
                url = "https://pyaging.s3.amazonaws.com/example_data/GSE139307.pkl"

                # Handle SSL certificate issues on macOS
                try:
                    # First attempt: standard download
                    urllib.request.urlretrieve(url, pickle_path)
                except Exception as ssl_error:
                    print(f"SSL error encountered: {ssl_error}")
                    print("Trying alternative download method...")

                    # Second attempt: disable SSL verification (less secure but works)
                    ssl_context = ssl._create_unverified_context()
                    urllib.request.urlretrieve(url, pickle_path, context=ssl_context)

                print(f"Downloaded GSE139307 dataset to {pickle_path}")

            except Exception as e:
                print(f"Error downloading genomic aging dataset: {e}")
                return None

        # Load the data from local memory
        try:
            with open(pickle_path, "rb") as f:
                data_dict = pickle.load(f)
            
            # Apply feature selection if n_features is specified
            if n_features is not None or specific_features is not None:
                # Get the methylation data
                methylation_data = data_dict.get('methylation_data', None)
                
                if methylation_data is not None and isinstance(methylation_data, pd.DataFrame):
                    # Select features based on the specified strategy
                    if feature_selection == "specific" and specific_features is not None:
                        # Select specific features
                        valid_features = [f for f in specific_features if f in methylation_data.columns]
                        if len(valid_features) < len(specific_features):
                            print(f"Warning: {len(specific_features) - len(valid_features)} requested features not found in the dataset")
                        
                        if not valid_features:
                            print("No valid features found. Returning all features.")
                        else:
                            methylation_data = methylation_data[valid_features]
                    
                    elif feature_selection == "variance" and n_features is not None:
                        # Select features with highest variance
                        variances = methylation_data.var()
                        top_features = variances.nlargest(min(n_features, len(variances))).index.tolist()
                        methylation_data = methylation_data[top_features]
                        print(f"Selected {len(top_features)} features with highest variance")
                    
                    elif feature_selection == "random" and n_features is not None:
                        # Randomly select features
                        np.random.seed(random_seed)
                        all_features = methylation_data.columns.tolist()
                        selected_features = np.random.choice(
                            all_features, 
                            size=min(n_features, len(all_features)), 
                            replace=False
                        )
                        methylation_data = methylation_data[selected_features]
                        print(f"Randomly selected {len(selected_features)} features")
                    
                    # Update the data dictionary with the selected features
                    data_dict['methylation_data'] = methylation_data
                    print(f"Final methylation data shape: {methylation_data.shape}")
            
            return data_dict
        except Exception as e:
            print(f"Error loading genomic aging dataset: {e}")
            return None

    def get_heart_disease_data(self, csv_path=None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Heart Disease dataset from UCI via Kaggle

        Args:
            csv_path: Path to the CSV file (optional) if data is already downloaded elsewhere.

        Returns:
            X: Features DataFrame
            y: Target Series (presence of heart disease)
        """
        # Path to the heart disease CSV file
        csv_path = (
            os.path.join(self.data_dir, "heart.csv") if csv_path is None else csv_path
        )

        # download dataset if it doesn't exist locally
        if not os.path.exists(csv_path):
            print("Downloading heart disease dataset from Kaggle...")
            try:
                dataset_dir = kagglehub.dataset_download(
                    "johnsmith88/heart-disease-dataset"
                )
                csv_file = os.path.join(dataset_dir, "heart.csv")

                if csv_file and os.path.exists(csv_file):
                    os.rename(csv_file, csv_path)
                    print(f"Downloaded and renamed to heart.csv")

            except Exception as e:
                print(f"Error downloading dataset: {e}")

        # load the data from local memory
        try:
            data = pd.read_csv(csv_path)
            print(f"Successfully loaded heart disease data with {len(data)} rows")

            target_col = "target"
            X = data.drop(target_col, axis=1)
            y = pd.Series(data[target_col], name=target_col)

            return X, y
        except Exception as e:
            print(f"Error loading heart disease data: {e}")
            return None, None
