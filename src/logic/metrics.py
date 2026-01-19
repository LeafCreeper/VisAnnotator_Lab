from sklearn.metrics import cohen_kappa_score, accuracy_score
import pandas as pd

def calculate_metrics(df, true_col, pred_col):
    """
    Calculates metrics between two columns in a dataframe.
    """
    # Filter out NaNs
    valid_df = df.dropna(subset=[true_col, pred_col])
    
    if len(valid_df) == 0:
        return {"accuracy": 0.0, "kappa": 0.0, "n": 0}

    y_true = valid_df[true_col].astype(str)
    y_pred = valid_df[pred_col].astype(str)
    
    # Get all unique labels
    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": acc,
        "kappa": kappa,
        "n": len(valid_df)
    }
