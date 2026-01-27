import torch
from collections import deque
from torch import Tensor


class VLA0TemporalEnsembler:
    def __init__(self, ensemble_prediction_count: int) -> None:
        """
        Implements the specific ensembling logic used in VLA0 Libero evaluation.

        Args:
            ensemble_prediction_count (int): Corresponds to ensemble_prediction param.
                This limits how many overlapping schedules are averaged.
        """
        self.max_schedules = ensemble_prediction_count
        self.reset()

    def reset(self):
        self.schedules = deque(maxlen=self.max_schedules)

    def update(self, new_action_chunk: Tensor) -> Tensor:
        """
        Args:
            new_action_chunk: Tensor of shape (batch, horizon, action_dim).
                Note: This implementation assumes batch_size=1 for simplicity
                as per standard eval loops, but can be adapted.
        """
        self.schedules.append(new_action_chunk)

        current_actions = []
        for i, schedule in enumerate(reversed(self.schedules)):
            # schedule shape: (Batch, Horizon, Action_Dim)
            horizon_len = schedule.shape[1]

            if i < horizon_len:
                action_at_step_i = schedule[:, i, :]
                current_actions.append(action_at_step_i)
            else:
                break

        if not current_actions:
            return new_action_chunk[:, 0, :]

        stacked_actions = torch.stack(current_actions, dim=0)
        action_to_execute = stacked_actions.mean(dim=0)

        return action_to_execute
