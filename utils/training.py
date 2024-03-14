import torch
from torchvision.transforms.functional import to_pil_image
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
from datasets.seq_imgnetsub import SequentialImageNetAnimals
import sys
import copy
from diffusers import StableDiffusionPipeline
from torch import nn
import time
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from label2prompt import label2prompt
import PIL.Image as Image
from PIL import ImageDraw, ImageFont
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
import random
import sys
from .cor_with_slow import Inference_with_slow, InferenceWithSlowModel


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True


def save_normalized_batch_tsne(data_tensor, filename):

    normalized_data = F.normalize(data_tensor, dim=-1).cpu().detach().numpy()

    tsne = TSNE(n_components=2,early_exaggeration=5,learning_rate=30,init='random',n_iter=2000,n_iter_without_progress=300,perplexity=5) 
    tsne_results = tsne.fit_transform(normalized_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization of Normalized Data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    plt.savefig(filename)

def normalize_and_find_outliers(input_tensor):
    
    mean = input_tensor.mean()
    std = input_tensor.std()
    
    normalized_tensor = (input_tensor - mean) / std
    
    outliers_indices = torch.nonzero(normalized_tensor < -0.842)
    
    return outliers_indices

# def save_errorcase(inputs, pred, labels, output_folder, filename):
#     if not os.path.exists(output_folder):
#         os.mkdir(output_folder)

#     mismatched_indices = torch.nonzero(pred != labels, as_tuple=True)[0]
#     if mismatched_indices.size(0) > 0:
#         batch_images = inputs[mismatched_indices]

#         filename = os.path.join(output_folder, filename)
#         save_image(batch_images, filename, nrow=batch_images.size(0))

def save_errorcase(inputs, pred, labels, output_folder, filename_prefix):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    mismatched_indices = torch.nonzero(pred != labels, as_tuple=True)[0]
    if mismatched_indices.size(0) > 0:
        for i, idx in enumerate(mismatched_indices):
            single_image = inputs[idx].unsqueeze(0)
            
            filename = f"{filename_prefix}_error_{i}.png"
            filepath = os.path.join(output_folder, filename)
            
            save_image(single_image, filepath)

def stat_label_prob(label_sums, label_counts, outputs, labels):
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        label_indices = (labels == label).nonzero().view(-1)
        
        label_elements = outputs[label_indices]
        
        label_sum = torch.sum(label_elements, dim=0)
        
        label_sums[label.item()] = label_sum
        label_counts[label.item()] = label_indices.size(0)
    
    return label_sums, label_counts

def plot_bar_chart(data, filename):
    
    a_values = [item[0] for item in data]
    b_values = [item[1] for item in data]

    x = range(len(data)) 
    width = 0.35 

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, a_values, width, label='Right')
    rects2 = ax.bar([i + width for i in x], b_values, width, label='All')

    ax.set_xlabel('task') 
    ax.set_ylabel('num') 
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels([str(i) for i in x])
    ax.legend()
    plt.savefig(filename)

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """

    if dataset.NAME == "seq-core50":
        N_CLASSES_PER_TASK = [10, 5, 5, 5, 5, 5, 5, 5, 5]
        FROM_CLASS = int(np.sum(N_CLASSES_PER_TASK[:k]))
        TO_CLASS = int(np.sum(N_CLASSES_PER_TASK[: k + 1]))
        outputs[:, 0:FROM_CLASS] = -float("inf")
        outputs[:, TO_CLASS:50] = -float("inf")
    else:
        outputs[:, 0 : k * dataset.N_CLASSES_PER_TASK] = -float("inf")
        outputs[
            :,
            (k + 1)
            * dataset.N_CLASSES_PER_TASK : dataset.N_TASKS
            * dataset.N_CLASSES_PER_TASK,
        ] = -float("inf")

# def save_goodcase(inputs, pred, labels, output_folder, filename):
#     if not os.path.exists(output_folder):
#         os.mkdir(output_folder)

#     matched_indices = torch.nonzero(pred == labels, as_tuple=True)[0]
#     if matched_indices.size(0) > 0:
#         selected_index = random.choice(matched_indices)
#         image_to_save = inputs[selected_index].unsqueeze(0)  # Unsqueeze to add a batch dimension

#         filename = os.path.join(output_folder, filename)
#         save_image(image_to_save, filename)

def save_goodcase(inputs, pred, labels, output_folder, filename_prefix):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    matched_indices = torch.nonzero(pred == labels, as_tuple=True)[0]
    if matched_indices.size(0) > 0:
        for i, idx in enumerate(matched_indices):
            single_image = inputs[idx].unsqueeze(0)
            
            filename = f"{filename_prefix}_good_{i}.png"
            filepath = os.path.join(output_folder, filename)
            
            save_image(single_image, filepath)

def evaluate(
    model: ContinualModel, dataset: ContinualDataset, task_id, last=False
) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    gamma = 1
    accs, accs_mask_classes = [], []
    wrong_pic = []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            (
                inputs,
                labels,
            ) = data
            inputs, labels = inputs.to(model.device, non_blocking=True), labels.to(
                model.device, non_blocking=True
            )
            with torch.no_grad():
                if "class-il" not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                elif last:
                    outputs = model.old_means_pre(inputs)
                elif dataset.args.model == "our":
                    outputs = model(inputs, dataset)
                else:
                    if (
                        dataset.args.model == "derppcct"
                        or dataset.args.model == "onlinevt"
                    ) and model.net.net.distill_classifier:
                        outputs = model.net.net.distill_classification(inputs)
                        # outputs = model.ncm(inputs)
                        outputs[
                            :,
                            (task_id)
                            * dataset.N_CLASSES_PER_TASK : (task_id)
                            * dataset.N_CLASSES_PER_TASK
                            + dataset.N_CLASSES_PER_TASK,
                        ] = (
                            outputs[
                                :,
                                (task_id)
                                * dataset.N_CLASSES_PER_TASK : (task_id)
                                * dataset.N_CLASSES_PER_TASK
                                + dataset.N_CLASSES_PER_TASK,
                            ]
                            * gamma
                        )
                    else:
                        outputs = model.net(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                wrong_tensor = inputs[pred!=labels]
                wrong_pic.append(wrong_tensor)
                total += labels.shape[0]

                if dataset.SETTING == "class-il":
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(
            round(correct / total * 100, 3) if "class-il" in model.COMPATIBILITY else 0
        )
        accs_mask_classes.append(round(correct_mask_classes / total * 100, 3))

    # image_compose(wrong_pic, "/home/bqqi/lifelong_research/workspace/wrong_pic/wrong_pic_compose.jpg", 10)
    model.net.train(status)
    return accs, accs_mask_classes

def evaluate_brain_co(
    model: ContinualModel, dataset: ContinualDataset, task_id, last=False, mem_inputs=None, slow_coworker = None
):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    gamma = 1
    uncer_correct = 0
    choose_imgs = 0
    num_adjusted = 0
    num_wrong = 0
    accs, accs_mask_classes = [], []
    accs_oot, num_oot = [], []
    wrong_pic = []
    label_sums = {}
    label_counts = {}
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        outofthr = 0.0
        correct_of_oot = 0.0
        nu = 0
        for data in test_loader:
            if len(data) == 4:
                inputs, labels, _, origin_imgs = data
                inputs, labels, origin_imgs = inputs.to(model.device), labels.to(model.device), origin_imgs.to(model.device)
            elif len(data) == 3:
                inputs, labels, _ = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
            else:
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
            inputs, labels = inputs.to(model.device, non_blocking=True), labels.to(
                model.device, non_blocking=True
            )
            with torch.no_grad():
                if "class-il" not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                elif last:
                    outputs = model.old_means_pre(inputs)
                elif dataset.args.model == "our":
                    outputs = model(inputs, dataset)
                elif dataset.args.model == "onlinevt" and dataset.args.with_brain:
                    img_brain_mem = torch.index_select(mem_inputs, 0, torch.tensor(model.net.net.opened_memories).to(inputs.device))
                    #print(inputs.shape, img_brain_mem.shape)
                    #inputs = torch.cat([inputs.unsqueeze(1), img_brain_mem.unsqueeze(1)], dim=1)
                    outputs = model.net(inputs, mem=img_brain_mem)
                    #print(outputs,outputs.shape)
                elif dataset.args.model == "onlinevt" and dataset.args.with_brain_vit:
                    outputs = model.net(inputs)
                else:
                    if (
                        dataset.args.model == "derppcct"
                        or dataset.args.model == "onlinevt"
                    ) and model.net.net.distill_classifier:
                        outputs = model.net.net.distill_classification(inputs)
                        # outputs = model.ncm(inputs)
                        outputs[
                            :,
                            (task_id)
                            * dataset.N_CLASSES_PER_TASK : (task_id)
                            * dataset.N_CLASSES_PER_TASK
                            + dataset.N_CLASSES_PER_TASK,
                        ] = (
                            outputs[
                                :,
                                (task_id)
                                * dataset.N_CLASSES_PER_TASK : (task_id)
                                * dataset.N_CLASSES_PER_TASK
                                + dataset.N_CLASSES_PER_TASK,
                            ]
                            * gamma
                        )
                    else:
                        outputs = model(inputs)
                # probs = F.softmax(outputs.data, dim=1)
                # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                # print(entropy.mean(dim=0))
                threshold = 0.9
                probs, pred = torch.max(outputs.data, 1)
                # uncertain_indices = torch.where(_ < 0.2)[0]
                uncertain_indices = normalize_and_find_outliers(probs)
                choose_imgs += torch.numel(uncertain_indices)
                uncertain_correct = torch.sum(pred[uncertain_indices] == labels[uncertain_indices]).item()
                if slow_coworker is not None and torch.numel(uncertain_indices)>0 and task_id == (dataset.N_TASKS-1): # and task_id == (dataset.N_TASKS-1)
                    if slow_coworker.slow_model == 'minigpt4':
                        adjusted_out = slow_coworker.Inference_slow(inputs[uncertain_indices], outputs[uncertain_indices])
                    elif slow_coworker.slow_model in ['PureMM', 'INF-MLLM']:
                        adjusted_out = slow_coworker.Inference_slow(inputs[uncertain_indices], outputs[uncertain_indices])
                    wrong_num = torch.sum(pred[uncertain_indices]!=labels[uncertain_indices]).item()
                    pred[uncertain_indices] = adjusted_out.long()
                    num_wrong += wrong_num
                    num_adjusted += torch.sum(pred[uncertain_indices]==labels[uncertain_indices]).item()
                    uncer_correct += uncertain_correct
                    print(uncer_correct)
                    print(num_adjusted)
                    print(choose_imgs)
                    outofthr += len(uncertain_indices)
                    correct_of_oot += uncertain_correct
                correct += torch.sum(pred == labels).item()
                # _, pred = torch.max(outputs.data, 1)
                # correct += torch.sum(pred == labels).item()

                # sorted_indices = torch.argsort(_)

                # num_samples = outputs.size(0)
                # num_to_select = int(num_samples * 0.1)  

                # uncertain_indices = sorted_indices[:num_to_select]
                # print(uncertain_indices, num_samples)
                # uncertain_correct = torch.sum(pred[uncertain_indices] == labels[uncertain_indices]).item() if len(uncertain_indices)>0 else 0
                
                
                # Record accuracy for uncertain predictions
                
                label_elements = outputs.data[torch.arange(outputs.size(0)), labels]
                stat_label_prob(label_sums, label_counts, label_elements, labels)
                # save_errorcase(inputs, pred, labels, '/home/bqqi/lifelong_research/errorcase', str(k)+str(nu))
                # save_goodcase(inputs, pred, labels, '/home/bqqi/lifelong_research/goodcase', str(k)+str(nu))
                nu+=1
                wrong_tensor = inputs[pred!=labels]
                wrong_pic.append(wrong_tensor)
                total += labels.shape[0]

                if dataset.SETTING == "class-il":
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()
        # uncertain_accuracy = correct_of_oot / outofthr if outofthr > 0 else 1
        # print('oot:%d, oot_acc:%2f'%(outofthr, uncertain_accuracy))
        num_oot.append(outofthr)
        accs_oot.append(correct_of_oot)
        accs.append(
            round(correct / total * 100, 3) if "class-il" in model.COMPATIBILITY else 0
        )
        accs_mask_classes.append(round(correct_mask_classes / total * 100, 3))

    # image_compose(wrong_pic, "/home/bqqi/lifelong_research/workspace/wrong_pic/wrong_pic_compose.jpg", 10)
    model.net.train(status)
    label_means = {label: label_sums[label] / label_counts[label] for label in label_sums.keys()}
    lowest_means = sorted(label_means.items(), key=lambda item: item[1])
    lowest_labels = [label for label, mean in lowest_means]
    print(lowest_labels)
    if slow_coworker is not None:
        return accs, accs_mask_classes, [np.array(accs_oot).sum(), np.array(num_oot).sum()], [num_wrong, num_adjusted]
    else:
        return accs, accs_mask_classes, [np.array(accs_oot).sum(), np.array(num_oot).sum()]

def evaluate_finetune(
    model: ContinualModel, dataset: ContinualDataset, task_id, last=False
):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            if len(data) == 4:
                inputs, labels, _, super_class = data
                inputs, labels, super_class = inputs.to(model.device), labels.to(model.device), super_class.to(model.device)
            elif len(data) == 3:
                inputs, labels, _ = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
            else:
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
            inputs, labels = inputs.to(model.device, non_blocking=True), labels.to(
                model.device, non_blocking=True
            )
            with torch.no_grad():
                outputs = model.net(inputs)
                _, pred = torch.max(outputs.data, 1)
                print(pred, labels)
                correct += torch.sum(pred == labels).item()
                
                total += labels.shape[0]

                if dataset.SETTING == "class-il":
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(
            round(correct / total * 100, 3) if "class-il" in model.COMPATIBILITY else 0
        )
        accs_mask_classes.append(round(correct_mask_classes / total * 100, 3))

        return accs, accs_mask_classes

def generate_gan_memory(
    current_task_id, dataset: ContinualDataset, num_per_task, device
):
    import os, sys

    sys.path.append(os.getcwd())
    from datasets.utils.GAN_datagenerator import Animal_Generator, task_range

    generator = Animal_Generator("cuda")
    images_all = []
    targets_all = []
    for i in range(current_task_id):
        images = generator.generate(
            224,
            task=task_range[i],
            num=num_per_task,
        )
        targets = torch.randint(low=0, high=6, size=[num_per_task]) + i * 6
        images = torch.tensor(images)
        targets = torch.tensor(targets)
        images_all.append(images)
        targets_all.append(targets)
    images = torch.cat(images_all)
    images = images.permute(0, 3, 1, 2) / 255
    images = dataset.get_normalization_transform()(images)
    targets = torch.cat(targets_all)
    return images.to(device), targets.to(device)

def generate_diffu_memory(
    diffusion_pipe, transform, num_classes, device
):
    toprompt = label2prompt()
    targets_all = []
    images_all = []
    for i in range(num_classes//12):
        targets = torch.arange(start=i*12, end=(i+1)*12)
        targets_all.append(targets)
        #targets = torch.tensor(targets)
        prompts = toprompt.map_labels_to_prompts(label_tensor = targets)
        if i==0:
            images = diffusion_pipe(prompts, num_inference_steps=50, eta=0.3, guidance_scale=6).images
            for image in images:
                images_all.append(transform(image)) 
        else:
            images = diffusion_pipe(prompts, num_inference_steps=50, eta=0.3, guidance_scale=6).images
            for image in images:
                images_all.append(transform(image))
        
    print('generation done')
    images = torch.stack(images_all)
    targets = torch.cat(targets_all)
    return images.to(device), targets.to(device)

def tensor2image(tensor: torch.Tensor):
    """
    将Tensor转换为图片，并保存为列表
    param tensor: 图片的张量
    """
    # 判断 Tensor 是不是四维张量
    if tensor.shape[1] != 3:
        print("张量的第二维不是通道数！")
        
    if len(tensor.shape) != 4:
        print("张量的维度不是4维！")
    
    images = []
    for i in range(tensor.shape[0]):
        # 获取当前图片的张量
        image_tensor = tensor[i]

        # 将张量转换为 PIL 图像对象
        # image = Image.fromarray(np.uint8(image_tensor.transpose((1, 2, 0)) * 255))
        image = to_pil_image(image_tensor)

        # 添加到图片列表
        images.append(image)
    return images


def image_compose(tensor_list, save_path, image_col, image_row = None):
    """
    定义图像拼接函数
    """
    # Tensor转为图片
    image_list = []
    for tensor in tensor_list:
        images = tensor2image(tensor)
        image_list += images

    if image_row is not None:
        if len(image_list) != image_row * image_col:
            print("图片数量与行数*列数不符合！") 
    else:
        image_row = len(image_list) // image_col + 1
        
    width, height = image_list[0].size
    padding = 5
    
    to_image = Image.new('RGB',(image_col * width + padding * (image_col - 1), image_row * height + padding * (image_row - 1)), 'white' )  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    image_count = 0
    for y in range(1, image_col + 1):
        for x in range(1, image_row + 1):
            if image_count < len(image_list):
                from_image = image_list[image_count]
                to_image.paste(from_image, ((x - 1) * width + padding * (x - 1), (y - 1) * height + padding * (y - 1)))
                image_count += 1
            else:
                break

    
    # 为每一列添加标题
    # draw = ImageDraw.Draw(to_image)
    # text = ['Original','Cat','Cat2','Cat3','Cat4','Original-Cat']
    # text_offset = [80,90,90,90,90,50]
    # font = ImageFont.truetype(r'C:\Windows\Fonts\timesbd.ttf', 32)

    # for i in range(1, image_col + 1):
    #     draw.text((text_offset[i-1]+256*(i-1), 10), text[i-1], fill='#666', font=font)

    to_image.save(save_path)  # 保存新图


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    model_stash = create_stash(model, args, dataset)
    results, results_mask_classes = [], []
    results_oot = []
    results_wrong_inoot = []
    results_adjusted_inoot = []
    if args.with_brain:
        model_id = "/home/bqqi/.cache/huggingface/transformers/stable-diffusion-v1-5"
        diffusion_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        diffusion_pipe = diffusion_pipe.to('cuda')
        transform = transforms.Compose([
                transforms.Resize((args.imsize, args.imsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        mem_input, mem_labels = generate_diffu_memory(
              diffusion_pipe, transform, args.num_classes, torch.device('cuda')
            )
        print('initialized done')
    # if args.with_slow:
    #     if args.slow_model == 'minigpt4':
    #         slow_coworker = Inference_with_slow(dataset.NAME, args)
    #     elif args.slow_model in ['PureMM', 'INF-MLLM']:
    #         slow_coworker = InferenceWithSlowModel(dataset.NAME, args)
    if args.csv_log:
        csv_logger = CsvLogger(
            dataset.SETTING, dataset.NAME, model.NAME, dataset.N_TASKS
        )
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash["tensorboard_name"] = tb_logger.get_name()

    if (
        model.NAME != "icarl"
        and model.NAME != "pnn"
        and model.NAME != "our"
        and model.NAME != "our_reservoir"
        and model.NAME != "er_tricks"
        and model.NAME != "onlinevt"
    ):
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        random_results_class, random_results_task = evaluate(model, dataset_copy, 0)

    start = time.time()
    print(file=sys.stderr)

    if hasattr(model.args, "ce"):
        ce = model.args.ce
        model.args.ce = 1
    print('Start Iteration')
    for t in range(dataset.N_TASKS):
        model.net.train()
        if args.with_brain_vit:
            model.net.net.task_now = t
        if args.vit_finetune:
            model.task_now_f = t
        if args.use_lr_scheduler:
            model.set_opt()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, "begin_task"):
            if model.NAME != "our":
                model.begin_task(dataset)
            elif args.begin_task:
                model.begin_task(dataset)
        if hasattr(model.args, "ce"):
            if t > 0:
                model.args.ce = ce * model.args.ce
                pass
            print("model.args.ce: ", model.args.ce)
        for epoch in range(args.n_epochs - int(t * 0)):
            if args.use_distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)

            model.display_img = False
            for i, data in enumerate(train_loader):
                if i == (train_loader.__len__() - 1):
                    model.display_img = True
                if hasattr(dataset.train_loader.dataset, "logits"):
                    inputs, labels, not_aug_inputs, logits = data
                    if (
                        t > 0
                        and isinstance(dataset, SequentialImageNetAnimals)
                        and args.ganmem
                    ):
                        mem_input, mem_labels = generate_gan_memory(
                            t, dataset, 6, device=inputs.device
                        )
                        inputs = torch.cat([inputs, mem_input])
                        labels = torch.cat([labels, mem_labels])
                    '''if args.with_brain:
                        if (i+1)%100==0:
                            print('Update Brain Memories')
                            mem_input, mem_labels = generate_diffu_memory(
                            diffusion_pipe, transform, args.num_classes, torch.device('cuda:0')
                            )'''
                    inputs = inputs.to(model.device, non_blocking=True)
                    labels = labels.to(model.device, non_blocking=True)
                    not_aug_inputs = not_aug_inputs.to(model.device, non_blocking=True)
                    logits = logits.to(model.device, non_blocking=True)
                    if args.with_brain:
                        loss = model.observe(inputs, labels, not_aug_inputs, logits, mem_input)
                    else:
                        loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    # inputs, labels, not_aug_inputs = data
                    if len(data) == 4:
                        inputs, labels, not_aug_inputs, _ = data
                        inputs, labels, not_aug_inputs = inputs.to(model.device), labels.to(model.device), not_aug_inputs.to(model.device)
                    elif len(data) == 3:
                        inputs, labels, not_aug_inputs = data
                        inputs, labels, not_aug_inputs = inputs.to(model.device), labels.to(model.device), not_aug_inputs.to(model.device)
                    else:
                        inputs, labels = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device)
                    if (
                        t > 0
                        and isinstance(dataset, SequentialImageNetAnimals)
                        and args.ganmem
                    ):
                        mem_input, mem_labels = generate_gan_memory(
                            t, dataset, 8, device=inputs.device
                        )
                        inputs = torch.cat([inputs, mem_input])
                        labels = torch.cat([labels, mem_labels])
                    if args.with_brain:
                        '''if (i+1)%100==0:
                            print('Update Brain Memories')
                            mem_input, mem_labels = generate_diffu_memory(
                            diffusion_pipe, transform, args.num_classes, torch.device('cuda:0')
                            )'''
                        
                    # inputs, labels = inputs.to(
                    #     model.device, non_blocking=True
                    # ), labels.to(model.device, non_blocking=True)
                    # not_aug_inputs = not_aug_inputs.to(model.device)
                    if args.with_brain:
                        loss = model.observe(inputs, labels, not_aug_inputs, mem_input)
                    elif args.vit_finetune:
                        loss = model.observe(inputs, labels, not_aug_inputs)
                    else:
                        loss = model.observe(inputs, labels, not_aug_inputs, obs_num = i)
                    #model.scheduler_mem.step()
                    #model.scheduler_proj.step()
                # progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                if (
                    hasattr(model, "middle_task")
                    and (i % 2000) == 0
                    and i > 0
                    and dataset.NAME == "seq-mnist"
                ):
                    tmp_buffer = copy.deepcopy(model.buffer)
                    model.middle_task(dataset)
                    accs = evaluate(model, dataset, t)
                    model.buffer = tmp_buffer
                    print(accs)
                    mean_acc = np.mean(accs, axis=1)
                    print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

                model_stash["batch_idx"] = i + 1
            model_stash["epoch_idx"] = epoch + 1
            model_stash["batch_idx"] = 0
            if model._scheduler is not None:
                model._scheduler.step()
        if args.with_brain_vit:
            
            # weight = model.net.net.Brain_embedding.weight
            # with torch.no_grad():
            #     new_weight = weight[:(t+1)*dataset.N_CLASSES_PER_TASK].detach().clone()
            #     new_weight.requires_grad_(False)
            #     weight[:(t+1)*dataset.N_CLASSES_PER_TASK].copy_(new_weight)
            #     model.net.net.Brain_embedding.weight = weight
            with torch.no_grad():
                new_weight = model.net.net.memory_2btrain.weight[t*dataset.N_CLASSES_PER_TASK:(t+1)*dataset.N_CLASSES_PER_TASK].detach().clone()
                model.net.net.Brain_embedding.weight[t*dataset.N_CLASSES_PER_TASK:(t+1)*dataset.N_CLASSES_PER_TASK] = new_weight
                new_tsks = model.net.net.memory_2btrain_tsk.weight[t].detach().clone()
                model.net.net.Brain_embedding_tsk.weight[t] = new_tsks
                if model.net.net.vit is not None:
                    for i, layer in enumerate(model.net.net.vit.model.vit.encoder.layer):
                        for name, module in layer.named_modules():
                            if isinstance(module, CustomViTAttention):
                                # new_weight = module.memory_2btrain.weight[t*dataset.N_CLASSES_PER_TASK:(t+1)*dataset.N_CLASSES_PER_TASK].detach().clone()
                                # module.Brain_embedding.weight[t*dataset.N_CLASSES_PER_TASK:(t+1)*dataset.N_CLASSES_PER_TASK] = new_weight
                                new_tsks = module.memory_2btrain_tsk.weight[t].detach().clone()
                                module.Brain_embedding_tsk.weight[t] = new_tsks
                                # new_in = module.memoryin_2btrain.weight[t*dataset.N_CLASSES_PER_TASK:(t+1)*dataset.N_CLASSES_PER_TASK].detach().clone()
                                # module.memoryin.weight[t*dataset.N_CLASSES_PER_TASK:(t+1)*dataset.N_CLASSES_PER_TASK] = new_in
                                # new_in = module.memoryin_2btrain.weight[t].detach().clone()
                                # module.memoryin.weight[t] = new_in

                #print(model.net.net.Brain_embedding.weight)
            #model.net.net.Brain_embedding.weight[:(t + 1) * dataset.N_CLASSES_PER_TASK].requires_grad = False
            # model.net.net.Brain_embedding.weight.detach()
            # model.net.net.Brain_embedding.weight.requires_grad = False
        model_stash["task_idx"] = t + 1
        model_stash["epoch_idx"] = 0
        
        if hasattr(model, "end_task"):
            if model.NAME == "our":
                model.end_task(dataset, t)
            else:
                model.end_task(dataset)
        if args.with_brain or args.with_brain_vit:
            if args.with_slow and t==(dataset.N_TASKS-1):
                if args.slow_model == 'minigpt4':
                    slow_coworker = Inference_with_slow(dataset.NAME, args)
                elif args.slow_model in ['PureMM', 'INF-MLLM']:
                    slow_coworker = InferenceWithSlowModel(dataset.NAME, args)
            else:
                slow_coworker = 1
            if args.with_slow:
                accs = evaluate_brain_co(model, dataset, t, False, slow_coworker=slow_coworker)
            else:
                accs = evaluate_brain_co(model, dataset, t, False)
            if args.with_slow:
                results_wrong_inoot.append(accs[3][0])
                results_adjusted_inoot.append(accs[3][1])
            results_oot.append(accs[2])
            accs = accs[:2]
            # save_normalized_batch_tsne(model.net.net.Brain_embedding.weight[0:(t+1)*dataset.N_CLASSES_PER_TASK], dataset.NAME+'task %d'%(t))
        elif args.vit_finetune:
            accs = evaluate_finetune(model, dataset, t)
        else:
            accs = evaluate(model, dataset, t)
        
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        # print('Task Trick', evaluate(model, dataset, last=True))

        model_stash["mean_accs"].append(mean_acc)

        if args.csv_log:
            csv_logger.log(mean_acc)
            csv_logger.log_class_detail(results)
            csv_logger.log_task_detail(results_mask_classes)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

        if hasattr(model.net, "frozen"):
            model.net.frozen(t)
    # torch.save(model.net.net.Brain_embedding.weight, 'embd_weight_before.txt')
    # torch.save(model.net.net.Brain_embedding_tsk.weight, 'embd_task_before.txt')
    end = time.time()
    time_train = round(end - start, 1)
    print("running time: ", time_train, " s")
    if args.with_brain_vit:
        if args.with_slow:
            num_oot = sum(results_oot[i][1] for i in range(len(results_oot)))
            num_wrong = sum(results_wrong_inoot)
            num_adjusted = sum(results_adjusted_inoot)
            print('selected oot: %d, wrong: %d, adjusted by slow: %d'%(num_oot, num_wrong, num_adjusted))
        plot_bar_chart(results_oot, dataset.NAME+'oot_acc')
    if args.csv_log:
        csv_logger.log_time(time_train)
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if (
            model.NAME != "icarl"
            and model.NAME != "pnn"
            and model.NAME != "our"
            and model.NAME != "our_reservoir"
            and model.NAME != "er_tricks"
            and model.NAME != "onlinevt"
        ):
            csv_logger.add_fwt(
                results, random_results_class, results_mask_classes, random_results_task
            )

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))

from transformers.models.vit.modeling_vit import ViTAttention
class CustomViTAttention(ViTAttention):
    def __init__(self, num_classes, task_num, config):
        super().__init__(config)
        self.num_classes = num_classes
        self.task_num = task_num
        self.cls_per_tsk = int(num_classes/task_num)

        self.Brain_embedding = nn.Embedding(num_classes, 576)
        nn.init.normal_(self.Brain_embedding.weight, mean=0, std=1)

        self.memory_2btrain = nn.Embedding(num_classes, 576)
        nn.init.normal_(self.memory_2btrain.weight, mean=0, std=1)

        self.Brain_embedding_tsk = nn.Embedding(task_num, 192)
        nn.init.normal_(self.Brain_embedding_tsk.weight, mean=0, std=1)

        self.memory_2btrain_tsk = nn.Embedding(task_num, 192)
        nn.init.normal_(self.memory_2btrain_tsk.weight, mean=0, std=1)

        self.memoryin = nn.Embedding(num_classes, 768)
        nn.init.normal_(self.memoryin.weight, mean=0, std=1)

        self.memoryin_2btrain = nn.Embedding(num_classes, 768)
        nn.init.normal_(self.memory_2btrain.weight, mean=0, std=1)

        self.memory_map = self._gen_projector(768, 3072, 576)
        self.embedding_map = self._gen_projector(768, 3072, 192)
        self.opened_memories = []
        self.opened_tasks = []

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _gen_projector(self, in_features, hidden_dim, out_dim):
        projector = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        #projector = ResidualLinear(in_features, hidden_dim, out_dim)
        projector.apply(self.initialize_weights)
        return projector

    def check_memories(self, labels):  
        for label in labels.unique():
            if label.item() not in self.opened_memories:
                self.opened_memories.append(label.item())
        self.opened_memories.sort()

    def check_tasks(self, tasks):  
        for task in tasks.unique():
            if task.item() not in self.opened_tasks:
                self.opened_tasks.append(task.item())
        self.opened_tasks.sort()

    def label_task(self, labels):
        tasks = torch.zeros_like(labels)
        bin_num = self.num_classes//self.cls_per_tsk
        for i in range(bin_num):
            bin_start = i * self.cls_per_tsk
            bin_end = (i+1) * self.cls_per_tsk
            tasks[(labels >= bin_start) & (labels < bin_end)] = i
        return tasks

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor = None,
        output_attentions: bool = False,
        labels: torch.Tensor = None,
        task_now: int = None,
    ):
        if labels is not None:
            self.check_memories(labels)
            condition = labels >= task_now * self.cls_per_tsk
            brain_embeddings_cls = torch.where(condition.unsqueeze(1), self.memory_2btrain(labels), self.Brain_embedding(labels))
            embeddings_2bin = torch.where(condition.unsqueeze(1), self.memoryin_2btrain(labels), self.memoryin(labels))
            tsk = self.label_task(labels)
            self.check_tasks(tsk)
            condition = tsk >= task_now
            brain_embeddings_tsk = torch.where(condition.unsqueeze(1), self.memory_2btrain_tsk(tsk), self.Brain_embedding_tsk(tsk))
            brain_embeddings = torch.cat([brain_embeddings_tsk, brain_embeddings_cls], dim=-1)
            
            #hidden_states = brain_embeddings.unsqueeze(-1) + hidden_states
            hidden_states = torch.cat([hidden_states, embeddings_2bin.unsqueeze(1)], dim=1)
            query_feature = hidden_states.mean(dim=1)
            query_feature_cls = self.memory_map(query_feature)
            query_featrue_tsk = self.embedding_map(query_feature)
            query_feature = torch.cat([query_featrue_tsk, query_feature_cls], dim=-1)
            query_feature = F.normalize(query_feature, dim=-1)
            brain_embeddings = F.normalize(brain_embeddings, dim=-1)
            logit_brain_mem = torch.matmul(query_feature, brain_embeddings.t())
            outputs = super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
            return outputs, logit_brain_mem
        
        else:
            query_feature = hidden_states.mean(dim=1)
            query_feature_cls = self.memory_map(query_feature)
            query_featrue_tsk = self.embedding_map(query_feature)
            query_feature = torch.cat([query_featrue_tsk, query_feature_cls], dim=-1)
            brain_embeddings_cls = self.Brain_embedding(torch.tensor(self.opened_memories, device=query_feature.device))
            brain_embeddings_tsk = self.Brain_embedding_tsk(torch.tensor(self.opened_tasks, device=query_feature.device))
            brain_embeddings = torch.cat([brain_embeddings_tsk.repeat_interleave(self.cls_per_tsk, dim=0), brain_embeddings_cls], dim=-1)
            embeddings_2bin = self.memoryin(torch.tensor(self.opened_memories, device=query_feature.device))
            normalized_brain_embeddings = F.normalize(brain_embeddings, dim=-1)
            normalized_query_feature = F.normalize(query_feature, dim=-1)
            logit_brain_mem = torch.matmul(normalized_query_feature, normalized_brain_embeddings.t())
            max_indices = torch.argmax(logit_brain_mem, dim=1)
            selected_embeddings = torch.index_select(embeddings_2bin, dim=0, index=max_indices)
            #hidden_states = selected_embeddings.unsqueeze(-1) + hidden_states
            hidden_states = torch.cat([hidden_states, selected_embeddings.unsqueeze(1)], dim=1)
            outputs = super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)
            return outputs