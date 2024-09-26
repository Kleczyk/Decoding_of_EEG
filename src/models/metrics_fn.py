from sklearn.metrics import accuracy_score, roc_auc_score

def compute_accuracy(labels, preds):
    """
    Oblicza dokładność modelu.

    Parametry:
    - labels (array): Prawdziwe etykiety.
    - preds (array): Przewidywane etykiety.

    Zwraca:
    - accuracy (float): Dokładność.
    """
    return accuracy_score(labels, preds)

def compute_auc(labels, probs):
    """
    Oblicza AUC ROC modelu.

    Parametry:
    - labels (array): Prawdziwe etykiety.
    - probs (array): Prawdopodobieństwa klas pozytywnych.

    Zwraca:
    - auc (float): Wartość AUC ROC.
    """
    return roc_auc_score(labels, probs)
