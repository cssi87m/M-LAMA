import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


def calculate_quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate Quadratic Weighted Kappa (QWK)
    Convert scores to integer scale (multiply by 2 to handle 0.5 increments)
    """
    # Convert to integer scale: 0.5 -> 1, 1.0 -> 2, 1.5 -> 3, etc.
    y_true_int = (y_true * 2).astype(int)
    y_pred_int = (y_pred * 2).astype(int)
    return cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE)
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_accuracy_at_1(y_true, y_pred):
    """
    Calculate Accuracy within 1 point error
    Accuracy@1: percentage of predictions within 1 point of the true score
    """
    abs_errors = np.abs(y_true - y_pred)
    within_1_point = np.sum(abs_errors <= 1.0)
    accuracy = within_1_point / len(y_true)
    return accuracy


def evaluate_predictions(csv_file):
    """
    Read predictions from CSV and calculate all evaluation metrics
    
    Args:
        csv_file: Path to the CSV file containing predictions
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Extract true and predicted scores
    y_true = df['true_score'].values
    y_pred_rounded = df['pred_score_rounded'].values
    
    # Calculate metrics
    mae = calculate_mae(y_true, y_pred_rounded)
    rmse = calculate_rmse(y_true, y_pred_rounded)
    qwk = calculate_quadratic_weighted_kappa(y_true, y_pred_rounded)
    acc_at_1 = calculate_accuracy_at_1(y_true, y_pred_rounded)
    
    # Create results dictionary
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'QWK': qwk,
        'Accuracy@1': acc_at_1
    }
    
    return metrics


def print_metrics(metrics):
    """
    Print evaluation metrics in a formatted way
    """
    print("=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"MAE (Mean Absolute Error):      {metrics['MAE']:.4f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}")
    print(f"QWK (Quadratic Weighted Kappa): {metrics['QWK']:.4f}")
    print(f"Accuracy@1 (within 1 point):    {metrics['Accuracy@1']:.4f} ({metrics['Accuracy@1']*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    # Path to the test predictions CSV file
    csv_file = "/home/user06/Interspeech_2026/Model/Model/Preds_grammar/test_predictions.csv"
    
    # Calculate metrics
    metrics = evaluate_predictions(csv_file)
    
    # Print results
    print_metrics(metrics)
    
    # Optionally save to file
    import json
    with open('/home/user06/Interspeech_2026/Model/Model/Preds_grammar/evaluation_test_results.json', 'a') as f:
        json.dump(metrics, f, indent=4)
    print("\nResults saved to /home/user06/Interspeech_2026/Model/Model/Preds_grammar/evaluation_test_results.json")
