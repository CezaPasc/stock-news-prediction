from torchmetrics import Metric
import torch
from torch import Tensor
import numpy as np

class ProfitSimulation(Metric):
    def __init__(self, threshold=0.01, base_amount = 500, scaler=None, **kwargs: torch.Any) -> None:
        super().__init__(**kwargs)
        self.add_state("profits", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.threshold = threshold
        self.scaler = scaler
        self.base_amount = base_amount
    
    def update(self, preds: Tensor, target: Tensor):
        preds = preds if not self.scaler else self.scaler.inverse_transform(preds)
        target = target.reshape(-1, 1) if not self.scaler else self.scaler.inverse_transform(target.reshape(-1, 1))
        
        should_open = abs(preds) > self.threshold
        right_direction = (preds >=0) == (target >= 0)
        theoretical_profit = abs(target) * self.base_amount
        profit_multiplier = np.where(right_direction, 1, -1)
        
        simulated_profit = theoretical_profit * profit_multiplier * should_open
        self.profits += simulated_profit.sum()
         
    def compute(self):
        return self.profits

    def _get_name(self):
        return self.get_name()

    def get_name(self):
        return "profit_simulation"


class AvgProfit(Metric):
    def __init__(self, threshold=0.01, base_amount = 500, scaler=None, **kwargs: torch.Any) -> None:
        super().__init__(**kwargs)
        self.add_state("profits", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("opened_positions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold
        self.scaler = scaler
        self.base_amount = base_amount
    
    def update(self, preds: Tensor, target: Tensor):
        preds = preds if not self.scaler else self.scaler.inverse_transform(preds)
        target = target.reshape(-1, 1) if not self.scaler else self.scaler.inverse_transform(target.reshape(-1, 1))
        
        should_open = abs(preds) > self.threshold
        right_direction = (preds >=0) == (target >= 0)
        theoretical_profit = abs(target) * self.base_amount
        profit_multiplier = np.where(right_direction, 1, -1)
        
        simulated_profit = theoretical_profit * profit_multiplier * should_open
        self.profits += simulated_profit.sum()
        self.opened_positions += should_open.sum()
    
    def compute(self):
        return self.profits / self.opened_positions / self.base_amount

    def _get_name(self):
        return self.get_name()

    def get_name(self):
        return "avg_profit"


class SentimentAcc(Metric):
    def __init__(self, threshold_left, threshold_right, name=None) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.name = name
        self.threshold_left = threshold_left
        self.threshold_right = threshold_right
        print("Sentiment Acc, Neutral Class: %f - %f" % (self.threshold_left, self.threshold_right))

    def classify(self, t: Tensor) -> Tensor:
        results = torch.empty(t.shape)
        
        results[t < self.threshold_left] = 0
        results[(t >= self.threshold_left) & (t < self.threshold_right)] = 1 
        results[t >= self.threshold_right] = 2

        return results

    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape[0] == target.shape[0], "preds and target need to have the same length"
        assert type(preds) == type(target), "preds and target need to have the same type"

        preds_over = self.classify(preds)
        target_over = self.classify(target)

        self.correct += sum(preds_over.flatten() == target_over.flatten())
        self.total += len(target)

    def compute(self):
        return self.correct.float() / self.total
        
    def _get_name(self):
        return self.get_name()

    def get_name(self):
        if self.name is not None:
            return "%s_sentiment_accuracy"
        return "sentiment_accuracy"


class DirectionAcc(Metric):
    def __init__(self, middle=0, name=None) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.name = name
        self.middle = middle

    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape[0] == target.shape[0], "preds and target need to have the same length"
        assert type(preds) == type(target), "preds and target need to have the same type"
        
        preds_over = preds > self.middle
        target_over = target > self.middle

        self.correct += sum(preds_over.flatten() == target_over.flatten())
        self.total += len(target)

    def compute(self):
        return self.correct.float() / self.total
        
    def _get_name(self):
        return self.get_name()

    def get_name(self):
        if self.name is not None:
            return "%s_accuracy"
        return "accuracy"
