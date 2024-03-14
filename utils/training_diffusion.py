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
from torchvision import transforms
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


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


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
            32,
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
    current_task_id, diffusion_pipe, transform, dataset: ContinualDataset, num_per_task, device
):
    import os, sys

    sys.path.append(os.getcwd())
    toprompt = label2prompt()
    #targets_all = []
    #prompts_all = []
    '''for i in range(current_task_id):
        targets = torch.arange(start=0, end=num_per_task) + i * 6
        #targets = torch.tensor(targets)
        prompts = toprompt.map_labels_to_prompts(label_tensor = targets)
        prompts_all.extend(prompts)
        targets_all.append(targets)'''
    targets_all = torch.arange(start=0, end=num_per_task*current_task_id)
    prompts_all = toprompt.map_labels_to_prompts(label_tensor = targets_all)
    images = diffusion_pipe(prompts_all, num_inference_steps=50, eta=0.3, guidance_scale=6).images
    resized_images = [transform(image) for image in images]
    images = torch.stack(resized_images)
    #images = torch.cat(images_all)
    #images = images.permute(0, 3, 1, 2) / 255
    #images = dataset.get_normalization_transform()(images)
    #targets = torch.cat(targets_all)
    return images.to(device), targets_all.to(device)

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

    if args.diffumem:
        model_id = "/home/bqqi/.cache/huggingface/transformers/stable-diffusion-v1-5"
        diffusion_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        diffusion_pipe = diffusion_pipe.to("cuda")
        transform = transforms.Compose([
                transforms.Resize((args.imsize, args.imsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    start = time.time()
    print(file=sys.stderr)

    if hasattr(model.args, "ce"):
        ce = model.args.ce
        model.args.ce = 1
    for t in range(dataset.N_TASKS):
        if t > 0 and isinstance(dataset, SequentialImageNetAnimals) and args.diffumem:
            print("set initial diffusionmemory")
            mem_input, mem_labels = generate_diffu_memory(
                t, diffusion_pipe, transform, dataset, 6, device=inputs.device
            )
            #print(inputs.shape, mem_input.shape)
            
        model.net.train()
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
                            t, dataset, 8, device=inputs.device
                        )
                        inputs = torch.cat([inputs, mem_input])
                        labels = torch.cat([labels, mem_labels])
                    inputs = inputs.to(model.device, non_blocking=True)
                    labels = labels.to(model.device, non_blocking=True)
                    not_aug_inputs = not_aug_inputs.to(model.device, non_blocking=True)
                    logits = logits.to(model.device, non_blocking=True)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    if (
                        t > 0
                        and isinstance(dataset, SequentialImageNetAnimals)
                        and args.ganmem
                    ):
                        print("ganmemory")
                        mem_input, mem_labels = generate_gan_memory(
                            t, dataset, 8, device=inputs.device
                        )
                        inputs = torch.cat([inputs, mem_input])
                        labels = torch.cat([labels, mem_labels])

                    elif t > 0 and isinstance(dataset, SequentialImageNetAnimals):
                        if args.diffumem and i%100==0:

                            print("diffusionmemory update")
                            mem_input, mem_labels = generate_diffu_memory(
                               t, diffusion_pipe, transform, dataset, 6, device=inputs.device
                            )
                            #print(inputs.shape, mem_input.shape)
                            #inputs = torch.cat([inputs, mem_input])
                            #labels = torch.cat([labels, mem_labels])
                        inputs = torch.cat([inputs, mem_input])
                        labels = torch.cat([labels, mem_labels])
                    #print(inputs.shape,labels.shape)
                    inputs, labels = inputs.to(
                        model.device, non_blocking=True
                    ), labels.to(model.device, non_blocking=True)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

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
        model_stash["task_idx"] = t + 1
        model_stash["epoch_idx"] = 0

        if hasattr(model, "end_task"):
            if model.NAME == "our":
                model.end_task(dataset, t)
            else:
                model.end_task(dataset)

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

    end = time.time()
    time_train = round(end - start, 1)
    print("running time: ", time_train, " s")
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
