import torch
import os, sys
from os.path import join as pjoin


from typing import Literal
from random import sample

task_range = ["fish", "bird", "snake", "dog", "butterfly", "insect"]
TaskType = Literal["fish", "bird", "snake", "dog", "butterfly", "insect"]
gan_path = pjoin(__file__, *(4 * [os.path.pardir]), "ganmemory")
gan_path = os.path.abspath(gan_path)
ws_path = pjoin(__file__, *(5 * [os.path.pardir]), "workspace")
ws_path = os.path.abspath(ws_path)

sys.path.append(gan_path)
from gan_training.config import load_config, build_models
from gan_training.utils_model_load import (
    model_equal_part_embed,
    load_model_norm,
)
from gan_training.distributions import get_ydist, get_zdist


class Animal_Generator:
    def __init__(self, device="cpu") -> None:
        config_path = pjoin(gan_path, "configs", "ImageNet_classify_53.yaml")
        config = load_config(config_path, pjoin(gan_path, "configs", "default.yaml"))
        config["generator"]["name"] = "resnet4_AdaFM_accumulate_multitasks"
        config["discriminator"]["name"] = "resnet4_AdaFM_accumulate_multitasks"
        config["data"]["nlabels"] = 6
        generator, _ = build_models(config)
        generator = generator.to(device)
        pre_train_weight_path = pjoin(
            ws_path, "pretrained_model", "CELEBAPre_generator"
        )
        dict_G = torch.load(pre_train_weight_path)
        dict_G = {name.replace("module.", ""): p for name, p in dict_G.items()}
        generator = model_equal_part_embed(generator, dict_G)
        generator = load_model_norm(generator)

        for task_id in range(6):
            # 导入训练好的GAN模型
            model_file = pjoin(
                ws_path, "trained_generators_per_task", task_range[task_id], "models"
            )
            temp = os.listdir(model_file)
            maxid = max([int(m.split("_")[1]) for m in temp])
            dict_G = torch.load(
                pjoin(model_file, task_range[task_id] + "_%08d_Pre_generator" % maxid)
            )
            generator = model_equal_part_embed(generator, dict_G)
            generator(task_id=task_id, UPDATE_GLOBAL=True)
        self._generator = generator
        self.device = device
        self._config = config

    def generate(self, image_size: int, task: TaskType, num: int):
        task_id = task_range.index(task)
        with torch.no_grad():
            self._generator.eval()
            y_sample = get_ydist(6, device=self.device)
            z_sample = get_zdist(
                self._config["z_dist"]["type"],
                self._config["z_dist"]["dim"],
                device=self.device,
            )
            y_0 = y_sample.sample((num,)).to(self.device)
            z = z_sample.sample((num,)).to(self.device)
            _images, _ = self._generator(z, y_0, task_id=task_id)
            _images = torch.nn.functional.interpolate(
                _images, size=image_size, mode="bilinear"
            )
            _images = _images / 2 + 0.5
            _images = (
                _images.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", torch.uint8)
                .numpy()
            )
            return _images
