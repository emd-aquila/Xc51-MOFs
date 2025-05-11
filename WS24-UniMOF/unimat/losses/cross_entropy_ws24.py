import math
import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import f1_score

@register_loss("cross_entropy_ws24")
class CrossEntropyLossWS24(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.task = task

        # Optional: use class counts to compute weights
        if self.task.args.weight_by_class and self.task.class_counts is not None:
            weights = 1.0 / (self.class_counts ** 0.5)
            self.register_buffer("class_weights", weights)
            print("Class weights:", self.class_weights)
        else:
            self.class_weights = None

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["target"]['finetune_target'].size(0)

        if not self.training:
            logging_output = {
                "loss": loss.item(),
                "logits": net_output[0].detach().cpu(),
                "predict": (torch.argmax(F.log_softmax(net_output[0].float(), dim=-1), dim=1) + 1).detach().cpu(),
                "target": sample["target"]['finetune_target'].view(-1).detach().cpu(),
                "bsz": sample_size,
                "sample_size": sample_size,
            }
        else:
            logging_output = {
                "loss": loss.item(),
                "bsz": sample_size,
                "sample_size": sample_size,
            }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = F.log_softmax(net_output[0], dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1)).float()
        target = sample['target']['finetune_target'].view(-1).long() - 1  # Labels 1-indexed?

        # Ensure both are on the same device
        lprobs = lprobs.to(target.device)
        if self.class_weights is not None:
            class_weights = self.class_weights.to(target.device)
        else:
            class_weights = None

        loss = F.nll_loss(
            lprobs,
            target,
            weight=class_weights,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        # Compute F1-score
        all_targets = []
        all_predictions = []
        for log in logging_outputs:
            all_targets.extend(log.get("target", []))
            all_predictions.extend(log.get("predict", []))

        if all_targets and all_predictions:
            f1 = f1_score(all_targets, all_predictions, average="macro")  # Use "macro" for multi-class F1
            metrics.log_scalar("f1", f1, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return True