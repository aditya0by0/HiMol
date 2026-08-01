import torch
import torchmetrics


class MacroF1(torchmetrics.Metric):
    """
    Computes the Macro F1 score, which is the unweighted mean of F1 scores for each class.
    This implementation differs from torchmetrics.classification.MultilabelF1Score in the behaviour for undefined
    values (i.e., classes where TP+FN=0). The torchmetrics implementation sets these classes to a default value.
    Here, the mean is only taken over classes which have at least one positive sample.

    Args:
        num_labels (int): Number of classes/labels.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at each forward
            before returning the value at the step. Default: False.
        threshold (float, optional): Threshold for converting predicted probabilities to binary (0, 1) predictions.
            Default: 0.5.
    """

    def __init__(
        self, num_labels: int, dist_sync_on_step: bool = False, threshold: float = 0.5
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "true_positives",
            default=torch.zeros(num_labels, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "positive_predictions",
            default=torch.zeros(num_labels, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "positive_labels",
            default=torch.zeros(num_labels, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.threshold = threshold

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the state (TPs, Positive Predictions, Positive labels) with the current batch of predictions and labels.

        Args:
            preds (torch.Tensor): Predictions from the model.
            labels (torch.Tensor): Ground truth labels.
        """
        preds_bool = (preds > self.threshold)
        labels_int = labels.to(torch.int64)
        tps = torch.sum(torch.logical_and(preds_bool, labels_int.to(torch.bool)), dim=0)
        self.true_positives += tps.to(torch.int64)
        self.positive_predictions += torch.sum(preds_bool, dim=0).to(torch.int64)
        self.positive_labels += torch.sum(labels_int, dim=0).to(torch.int64)

    def compute(self) -> torch.Tensor:
        """
        Compute the Macro F1 score.

        Returns:
            torch.Tensor: The computed Macro F1 score.
        """

        # ignore classes without positive labels
        # classes with positive labels, but no positive predictions will get a precision of "nan" (0 divided by 0),
        # which is propagated to the classwise_f1 and then turned into 0
        mask = self.positive_labels != 0
        precision = self.true_positives[mask] / self.positive_predictions[mask]
        recall = self.true_positives[mask] / self.positive_labels[mask]
        classwise_f1 = 2 * precision * recall / (precision + recall)
        # if (precision and recall are 0) or (precision is nan), set f1 to 0
        classwise_f1 = classwise_f1.nan_to_num()
        return torch.mean(classwise_f1)
