import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ["examples", "labels", "logits", "task_labels"]
        self.dict = {}

        # scores for lossoir
        self.importance_scores = torch.ones(self.buffer_size).to(self.device) * -float(
            "inf"
        )
        # scores for balancoir
        self.balance_scores = torch.ones(self.buffer_size).to(self.device) * -float(
            "inf"
        )
        # merged scores
        self.scores = torch.ones(self.buffer_size).to(self.device) * -float("inf")

    def init_tensors(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        task_labels: torch.Tensor,
    ) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith("els") else torch.float32
                setattr(
                    self,
                    attr_str,
                    torch.zeros(
                        (self.buffer_size, *attr.shape[1:]),
                        dtype=typ,
                        device=self.device,
                    ),
                )

    def merge_scores(self):
        scaling_factor = (
            self.importance_scores.abs().mean() * self.balance_scores.abs().mean()
        )
        norm_importance = self.importance_scores / scaling_factor
        presoftscores = 0.5 * norm_importance + 0.5 * self.balance_scores

        if presoftscores.max() - presoftscores.min() != 0:
            presoftscores = (presoftscores - presoftscores.min()) / (
                presoftscores.max() - presoftscores.min()
            )
        self.scores = presoftscores / presoftscores.sum()

    def update_scores(self, indexes, values):
        self.importance_scores[indexes] = values

    def update_all_scores(self):
        self.balance_scores = (
            torch.tensor([self.dict[x.item()] for x in self.labels])
            .float()
            .to(self.device)
        )

    def functionalReservoir(self, N, m):
        if N < m:
            return N

        rn = np.random.randint(0, N)
        if rn < m:
            self.update_all_scores()
            self.merge_scores()
            index = np.random.choice(range(m), p=self.scores.cpu().numpy(), size=1)
            return index
        else:
            return -1

    def add_data(
        self, examples, labels=None, logits=None, task_labels=None, loss_scores=None
    ):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = self.functionalReservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    if self.num_seen_examples >= self.buffer_size:
                        self.dict[self.labels[index].item()] -= 1
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                self.importance_scores[index] = (
                    -float("inf") if loss_scores is None else loss_scores[i]
                )
                if labels[i].item() in self.dict:
                    self.dict[labels[i].item()] += 1
                else:
                    self.dict[labels[i].item()] = 1

    def get_data(
        self, size: int, transform: transforms = None, return_indexes=False
    ) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > self.num_seen_examples:
            size = self.num_seen_examples

        choice = np.random.choice(self.examples.shape[0], size=size, replace=False)
        if transform is None:
            transform = lambda x: x
        ret_tuple = (
            torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(
                self.device
            ),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        if not return_indexes:
            return ret_tuple
        else:
            return ret_tuple + (choice,)

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            transform = lambda x: x
        ret_tuple = (
            torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, None)
