# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using the training dataset and evaluates using the test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt

def parse_args():
    '''Parse input arguments'''

    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults. 
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The function to measure the quality of a split')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Step 2: Read the train and test datasets from the provided paths
    # Read train and test data from CSV
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.  
    # Split the data into input(X) and output(y)
    y_train = train_df['price']
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.  
    # Initialize and train a RandomForest Regressor
    rforest_model =RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    rforest_model= rforest_model.fit(X_train, y_train) 

    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.  
    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.  
    # Predict using the RandomForest Regressor on test data
    rforest_predictions = rforest_model.predict(X_test)  # Predict the test data

    # Step 7: Log the RandomForest Regressor metrics in MLflow for model evaluation, and save the trained model to the specified output path.  
    # Compute and log RMSE for test data
    mse = mean_squared_error(y_test, rforest_predictions)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    # Logging the mse as a metric
    mlflow.log_metric("MSE", float(mse))  # Log the MSE

    # Root Mean Squared Error (RMSE) represents the square root of MSE — 
    # it indicates how much the predicted prices deviate, on average, from the actual prices.
    # Lower RMSE values indicate better model performance.

    # Compute and log RMSE for test data
    rmse = mean_squared_error(y_test, rforest_predictions, squared=False)
    print('Root Mean Squared Error (RMSE) of RandomForest Regressor on test set: {:.2f}'.format(rmse))
    # Logging the RMSE as a metric
    mlflow.log_metric("RMSE", float(rmse))  # Log the RMSE

    # Adjusted R-squared (Adjusted R²) accounts for the number of predictors in the model.
    # Unlike regular R², it penalizes the addition of irrelevant features and increases only 
    # if new predictors improve the model more than would be expected by chance.
    # Higher Adjusted R² values indicate a better-fitting model after adjusting for model complexity.

    # Compute and log Adjusted R² for test data
    r2 = r2_score(y_test, rforest_predictions)
    n = len(y_test)
    p = X_test.shape[1]

    if n > p + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = float('nan')  # Safeguard against invalid denominator when n <= p + 1
        print("Warning: Adjusted R² undefined because number of samples ≤ number of predictors + 1.")

    print('Adjusted R-squared (Adjusted R²) of RandomForest Regressor on test set: {:.4f}'.format(adj_r2))
    # Logging the Adjusted R² as a metric
    mlflow.log_metric("Adjusted_R2", float(adj_r2))  # Log the Adjusted R²

    # Save the model
    mlflow.sklearn.save_model(sk_model=rforest_model, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
