import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datasets import get_dataset
import torch


def test_prediction(model, dataset):
    model.net.eval()
    true_labels, pred_labels = torch.tensor([]), torch.tensor([])
    for k, test_loader in enumerate(dataset.test_loaders):
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device, non_blocking=True), labels.to(
                model.device, non_blocking=True
            )
            with torch.no_grad():
                if "class-il" not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    if (
                        dataset.args.model == "derppcct"
                        and model.net.net.distill_classifier
                    ):
                        outputs = model.net.net.distill_classification(inputs)
                    else:
                        outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                true_labels = torch.cat((true_labels, labels.cpu()), 0)
                pred_labels = torch.cat((pred_labels, pred.cpu()), 0)

    return true_labels, pred_labels


def plot_confusion(model, args, dataset):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    true_labels, pred_labels = test_prediction(model, dataset_copy)

    # Create confusion matrix
    confusion_mat = confusion_matrix(true_labels, pred_labels)

    normalize = True
    if normalize:
        confusion_mat = confusion_mat.astype("float") / confusion_mat.max()

    # Visualize confusion matrix
    plt.imshow(confusion_mat, interpolation="nearest", cmap=plt.cm.jet)
    plt.xlim(-0.5, len(np.unique(true_labels)) - 0.5)
    plt.ylim(len(np.unique(true_labels)) - 0.5, -0.5)
    # plt.title('Confusion matrix')
    plt.colorbar()
    ticks = np.arange(10) * 10 + 10
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.savefig(
        "graph/confusion_CVT_1000_f.pdf",
        dpi=600,
        format="pdf",
        pad_inches=0.01,
        bbox_inches="tight",
    )
    plt.show()
