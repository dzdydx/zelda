import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    def __init__(
        self,
        name,
        maximize,
    ):
        """
        :param name: Override the name of this scoring function.
        :param maximize: Maximize this score? (Otherwise, it's a loss or energy
            we want to minimize, and I guess technically isn't a score.)
        """
        if name:
            self.name = name
        self.maximize = maximize

    def __call__(self, *args, **kwargs):
        """
        Calls the compute function of the metric, and after validating the output,
        returns the metric score
        """
        ret = self._compute(*args, **kwargs)
        return ret

    def _compute(
        self, predictions, targets, **kwargs
    ):
        """
        Compute the score based on the predictions and targets.
        This is a private function and the metric should be used as a functor
        by calling the `__call__` method which calls this and also validates
        the return type
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name

class Top1Accuracy(ScoreFunction):
    name = "top1_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            if predicted_class == target_class:
                correct += 1

        return correct / len(targets)

class MeanAveragePrecision(ScoreFunction):
    """
    Average Precision is calculated in macro mode which calculates
    AP at a class level followed by macro-averaging across the classes.
    """

    name = "mAP"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot

        """
        Based on suggestions from Eduardo Fonseca -
        Equal weighting is assigned to each class regardless
        of its prior, which is commonly referred to as macro
        averaging, following Hershey et al. (2017); Gemmeke et al.
        (2017).
        This means that rare classes are as important as common
        classes.

        Issue with average_precision_score, when all ground truths are negative
        https://github.com/scikit-learn/scikit-learn/issues/8245
        This might come up in small tasks, where few samples are available
        """
        return average_precision_score(targets, predictions, average="macro")
    
class AUCROC(ScoreFunction):
    """
    AUCROC (macro mode) is calculated per class followed by averaging across the
    classes
    """

    name = "aucroc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            # Macro mode auc-roc. Please check `MeanAveragePrecision`
            # for the reasoning behind using using macro mode
            auc = roc_auc_score(targets, predictions, average="macro")
            return auc
        except ValueError:
            return np.nan