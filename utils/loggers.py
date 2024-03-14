import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

useless_args = [
    "dataset",
    "tensorboard",
    "validation",
    "model",
    "csv_log",
    "notes",
    "load_best_args",
]


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int, setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == "domain-il":
        mean_acc, _ = mean_acc
        print(
            "\nAccuracy for {} task(s): {} %".format(task_number, round(mean_acc, 2)),
            file=sys.stderr,
        )
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print(
            "\nAccuracy for {} task(s): \t [Class-IL]: {} %"
            " \t [Task-IL]: {} %\n".format(
                task_number, round(mean_acc_class_il, 2), round(mean_acc_task_il, 2)
            ),
            file=sys.stderr,
        )


def print_incremental_accuracy(
    mean_acc_list: np.ndarray, task_number: int, setting: str
) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == "domain-il":
        mean_acc, _ = mean_acc_list
        print(
            "\nAccuracy for {} task(s): {} %".format(task_number, round(mean_acc, 2)),
            file=sys.stderr,
        )
    else:
        for mean_acc in mean_acc_list:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            print(
                "\nAccuracy for {} task(s): \t [Class-IL]: {} %"
                " \t [Task-IL]: {} %\n".format(
                    task_number, round(mean_acc_class_il, 2), round(mean_acc_task_il, 2)
                ),
                file=sys.stderr,
            )


class CsvLogger:
    def __init__(
        self, setting_str: str, dataset_str: str, model_str: str, n_task: int
    ) -> None:
        self.accs = []
        self.n_task = n_task
        if setting_str == "class-il":
            self.accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

        self.accs_mask_classes_detail = None
        self.accs_detail = None
        self.time = None

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == "class-il":
            self.fwt_mask_classes = forward_transfer(
                results_mask_classes, accs_mask_classes
            )

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log_class_detail(self, class_acc_detail: np.ndarray) -> None:
        self.accs_detail = class_acc_detail

    def log_task_detail(self, task_acc_detail: np.ndarray) -> None:
        self.accs_mask_classes_detail = task_acc_detail

    def log_time(self, time_train) -> None:
        self.time = time_train

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == "general-continual":
            self.accs.append(mean_acc)
        elif self.setting == "domain-il":
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(round(mean_acc_class_il, 2))
            self.accs_mask_classes.append(round(mean_acc_task_il, 2))

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in useless_args:
            if cc in args:
                del args[cc]

        columns = list(args.keys())

        new_cols = []
        for i, acc in enumerate(self.accs):
            args["task" + str(i + 1)] = acc
            new_cols.append("task" + str(i + 1))

        args["forward_transfer"] = self.fwt
        new_cols.append("forward_transfer")

        args["backward_transfer"] = self.bwt
        new_cols.append("backward_transfer")

        args["forgetting"] = self.forgetting
        new_cols.append("forgetting")

        columns = new_cols + columns

        detail_cols = []
        for i, acc in enumerate(self.accs_detail):
            args["task" + str(i + 1) + "_detail"] = acc
            detail_cols.append("task" + str(i + 1) + "_detail")

        for i in range(len(self.accs_detail)):
            if i < 1:
                args["task" + str(i + 1) + "_forgetting"] = 0
            else:
                args["task" + str(i + 1) + "_forgetting"] = forgetting(
                    self.accs_detail[: i + 1]
                )
            detail_cols.append("task" + str(i + 1) + "_forgetting")

        columns = columns + detail_cols

        time_cols = []
        args["time"] = self.time
        time_cols.append("time")

        columns = columns + time_cols

        create_if_not_exists(base_path() + "results/" + self.setting)
        create_if_not_exists(
            base_path() + "results/" + self.setting + "/" + self.dataset
        )
        create_if_not_exists(
            base_path()
            + "results/"
            + self.setting
            + "/"
            + self.dataset
            + "/"
            + self.model
        )

        write_headers = False
        path = (
            base_path()
            + "results/"
            + self.setting
            + "/"
            + self.dataset
            + "/"
            + self.model
            + "/mean_accs_{}task.csv".format(self.n_task)
        )
        if not os.path.exists(path):
            write_headers = True
        with open(path, "a") as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)

        if self.setting == "class-il":
            create_if_not_exists(base_path() + "results/task-il/" + self.dataset)
            create_if_not_exists(
                base_path() + "results/task-il/" + self.dataset + "/" + self.model
            )

            for i, acc in enumerate(self.accs_mask_classes):
                args["task" + str(i + 1)] = acc

            for i, acc in enumerate(self.accs_mask_classes_detail):
                args["task" + str(i + 1) + "_detail"] = acc

            for i in range(len(self.accs_mask_classes_detail)):
                if i < 1:
                    args["task" + str(i + 1) + "_forgetting"] = 0
                else:
                    args["task" + str(i + 1) + "_forgetting"] = forgetting(
                        self.accs_mask_classes_detail[: i + 1]
                    )

            args["forward_transfer"] = self.fwt_mask_classes
            args["backward_transfer"] = self.bwt_mask_classes
            args["forgetting"] = self.forgetting_mask_classes

            write_headers = False
            path = (
                base_path()
                + "results/task-il"
                + "/"
                + self.dataset
                + "/"
                + self.model
                + "/mean_accs_{}task.csv".format(self.n_task)
            )
            if not os.path.exists(path):
                write_headers = True
            with open(path, "a") as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)
