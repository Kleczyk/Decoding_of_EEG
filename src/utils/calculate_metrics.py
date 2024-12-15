import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Updated example with valid AUC input data for multiclass classification
def calculate_metrics(y: torch.Tensor, y_hat: torch.Tensor) -> dict[str, float]:
    """
    Calculate classification metrics: Accuracy, Precision, Recall, F1 Score, and AUC.

    :param y_hat: Model predictions (logits) as a torch.Tensor.
    :param y: True labels as a torch.Tensor.
    :return: A dictionary containing calculated metrics.
    """
    metrics = {}

    # Converting logits to predicted classes
    y_pred = torch.argmax(y_hat, dim=1)

    # Calculating basic metrics (ensure tensors are moved to CPU)
    metrics['accuracy'] = accuracy_score(y.cpu(), y_pred.cpu())
    metrics['precision'] = precision_score(y.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y.cpu(), y_pred.cpu(), average='weighted')

    # Attempt AUC computation for multi-class
    try:
        y_prob = torch.softmax(y_hat, dim=1).cpu().detach().numpy()
        y_true = y.cpu().numpy()
        metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except ValueError:
        metrics['auc'] = float('nan')  # Handle cases where AUC computation fails

    return metrics


# Adjusted test case to ensure AUC can be computed
if __name__ == "__main__":
    """
    Example usage of the calculate_metrics function.

    :example y_hat_example: Fake logits for demonstration as a torch.Tensor.
    :example y_example: True labels for demonstration as a torch.Tensor.
    """
    # Updated logits and labels with all classes represented
    y_hat_example = torch.tensor([
        [2.5, 0.3, 1.2],
        [0.2, 3.1, 0.7],
        [1.5, 2.0, 0.5],
        [0.1, 0.2, 3.7]
    ])
    y_example = torch.tensor([0, 1, 2, 2])

    results = calculate_metrics(y_example, y_hat_example)
    for metric, value in results.items():
        print(f"{metric}: {value}")
