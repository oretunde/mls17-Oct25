# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    args = parser.parse_args()

    return args

def main(args):
    '''Read, split, and save datasets'''

    # Log arguments
    logging.info(f"Input data path: {args.raw_data}")
    logging.info(f"Test-train ratio: {args.test_train_ratio}")

    # Reading Data
    df = pd.read_csv(args.raw_data)
    
    # Step 1: Perform label encoding to convert categorical features into numerical values for model compatibility.
    label_encoder = LabelEncoder()
    df['Segment'] = label_encoder.fit_transform(df['Segment'])

    # Log the first few rows of the dataframe
    logging.info(f"Transformed Data:\n{df.head()}")

    # Step 2: Split the dataset into training and testing sets using train_test_split with specified test size and random state. 
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Step 3: Save the training and testing datasets as CSV files in separate directories for easier access and organization.  
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Step 4: Log the number of rows in the training and testing datasets as metrics for tracking and evaluation.  
    # Log the metrics
    mlflow.log_metric('train size', train_df.shape[0])
    mlflow.log_metric('test size', test_df.shape[0])

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)
    
    main(args)
    mlflow.end_run()
