import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
import time


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


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, args=None, n_tasks=None, mode="reservoir"):
        assert mode in ["ring", "reservoir"]
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.buffer_count = 0
        self.num_seen_class_examples = {}
        self.functional_index = eval(mode)
        self.args = args
        self.num_classes = args.num_classes
        print(n_tasks)
        if mode == "ring":
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ["examples", "labels", "logits", "task_labels", "features"]
        # self.examples = None
        if hasattr(args, "dataset") and "imagenet" in args.dataset:
            self.transform_type_A = False
        else:
            self.transform_type_A = False
        # 初始化按类别计数器
        for i in range(self.num_classes):
            self.num_seen_class_examples[str(i)] = 0

    def init_tensors(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        task_labels: torch.Tensor,
        features: torch.Tensor,
    ) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param features: tensor containing the latent features
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
                
    def reservoir_class(self, label, max_class_buffer_size, abs_add):
        """
        Reservoir sampling algorithm.
        :return: the target index if the current image is sampled, else -1
        """      
        if self.num_seen_class_examples[label] < max_class_buffer_size:
            return self.num_seen_class_examples[label], 1
        if abs_add:
            rand = np.random.randint(0, self.num_seen_class_examples[label])
        else:
            rand = np.random.randint(0, self.num_seen_class_examples[label] + 1)
        if rand < max_class_buffer_size:
            return rand, 0
        else:
            return -1, 0

    def add_data(
        self, examples, labels=None, logits=None, task_labels=None, features=None, abs_add=False
    ):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
                :param features: tensor containing the latent features
        :return:
        """
        if self.buffer_size == 20:
            time.sleep(100)
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels, logits, task_labels, features)

        max_class_buffer_size = self.buffer_size // self.num_classes
        for i in range(examples.shape[0]):
            # print(examples.shape[0])
            # index = reservoir(self.num_seen_examples, self.buffer_size)
            if labels is not None:
                index, count = self.reservoir_class(str(labels[i].item()), max_class_buffer_size, abs_add)
                self.buffer_count += count
            self.num_seen_class_examples[str(labels[i].item())] += 1
            if index >= 0:
                index = index + labels[i] * max_class_buffer_size
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if features is not None:
                    self.features[index] = task_labels[i].to(self.device)
        # print(self.labels)
        # print(self.buffer_count)
            # if index >= 0:
            #     self.examples[index] = examples[i]
            #     if labels is not None:
            #         self.labels[index] = labels[i]
            #     if logits is not None:
            #         self.logits[index] = logits[i]
            #     if task_labels is not None:
            #         self.task_labels[index] = task_labels[i]
            #     if features is not None:
            #         self.features[index] = task_labels[i]

    def add_data_our(
        self, examples, labels=None, logits=None, task_labels=None, features=None
    ):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param features: tensor containing the latent features
        :return:
        """
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels, logits, task_labels, features)

        for i in range(examples.shape[0]):
            self.examples[self.num_seen_examples] = examples[i].to(self.device)
            if labels is not None:
                self.labels[self.num_seen_examples] = labels[i].to(self.device)
            if logits is not None:
                self.logits[self.num_seen_examples] = logits[i].to(self.device)
            if task_labels is not None:
                self.task_labels[self.num_seen_examples] = task_labels[i].to(
                    self.device
                )
            if features is not None:
                self.features[self.num_seen_examples] = features[i].to(self.device)

            self.num_seen_examples += 1

    def get_data(self, size: int, transform: transforms = None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.buffer_count, self.examples.shape[0]):
            size = min(self.buffer_count, self.examples.shape[0])

        choice = np.random.choice(
            min(self.buffer_count, self.examples.shape[0]),
            size=size,
            replace=False,
        )
        if transform is None:
            transform = lambda x: x
        if self.transform_type_A:
            ret_tuple = (
                torch.stack(
                    [
                        transform(image=ee.cpu().numpy())["image"]
                        for ee in self.examples[choice]
                    ]
                ).to(self.device),
            )
        else:
            ret_tuple = (
                torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(
                    self.device
                ),
            )
            aug_2_view = (
                torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(
                    self.device
                ),
            )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        if hasattr(self.args, "model") and self.args.model == "onlinevt":
            ret_tuple += aug_2_view
        return ret_tuple

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        # if hasattr(self, 'examples'):

        if transform is None:
            transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),)
        if self.transform_type_A and transform is None:
            ret_tuple = (
                torch.stack(
                    [transform(image=ee.cpu().numpy())["image"] for ee in self.examples]
                ).to(self.device),
            )
        else:
            ret_tuple = (
                torch.stack([transform(ee.cpu()) for ee in self.examples]).to(
                    self.device
                ),
            )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)

        return ret_tuple

    def get_all_data_domain(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        # if hasattr(self, 'examples'):

        if transform is None:
            transform = lambda x: x
        if self.transform_type_A:
            ret_tuple = (
                torch.stack(
                    [transform(image=ee.cpu().numpy())["image"] for ee in self.examples]
                ).to(self.device),
            )
        else:
            ret_tuple = (
                torch.stack([transform(ee.cpu()) for ee in self.examples]).to(
                    self.device
                ),
            )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.buffer_count == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
