import argparse
import os
import sys
sys.path.append("..")
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList
from PIL import Image
from datasets import get_test_datasets
from MiniGPT4.minigpt4.common.config import Config
from MiniGPT4.minigpt4.common.dist_utils import get_rank
from MiniGPT4.minigpt4.common.registry import registry
from MiniGPT4.minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from MiniGPT4.minigpt4.datasets.builders import *
from MiniGPT4.minigpt4.models import *
from MiniGPT4.minigpt4.processors import *
from MiniGPT4.minigpt4.runners import *
from MiniGPT4.minigpt4.tasks import *
import time
from torchvision import transforms
from datasets.label2name import LabelConverter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

# import sys
# if 'bitsandbytes' in sys.modules:
#     del sys.modules['bitsandbytes']
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default = '/home/bqqi/lifelong_research/src/CL_Transformer/utils/MiniGPT4/eval_configs/minigpt4_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model. Use -1 for CPU.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--seed", default=42)
    parser.add_argument("--dataset", default="seq-cifar10")
    parser.add_argument("--batchsize", default=32)
    args = parser.parse_args()
    return args

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class InferenceGPT():
    def __init__(self, args):
        
        # GPT Model Initialization
        self.label2name = LabelConverter(args.dataset)
        # 准备第二次的prompt
        # self.name_list = self.label2name.labels
        if args.dataset == 'seq-imagenet-r':
            self.name_list = list(self.label2name.labels.values())
        else:
            self.name_list = self.label2name.labels
        fill = ', '.join(self.name_list[:-1]) + ' or ' + self.name_list[-1]
        self.query = 'From your description, answer which category the picture is belong to: {}? Please choose one most possible category.'.format(fill)
        
        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                    'pretrain_llama2': CONV_VISION_LLama2}

        print('Initializing Chat')
        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        self.CONV_VISION = conv_dict[model_config.model_type]

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
        print('Initialization Finished')
        
    def Inference_one(self, input, query):
        user_message = "Describe the the given image."

        chatbot = []
        img_list = []  
        chat_state = self.CONV_VISION.copy()
        self.chat.upload_img(input, chat_state, img_list)
        self.chat.encode_img(img_list)
        self.chat.ask(user_message, chat_state)
        chatbot += [[user_message, None]]

        llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=1,
                                temperature=1.0,
                                max_new_tokens=300,
                                max_length=2000)[0]
        chatbot[-1][1] = llm_message
        # 第二次prompt 问结果
        user_message = query
        self.chat.ask(user_message, chat_state)
        chatbot = [[user_message, None]]

        llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=1,
                                temperature=1.0,
                                max_new_tokens=300,
                                max_length=2000)[0]
        chatbot[-1][1] = llm_message

        return chatbot[-1][1]
    

    def classification(self, inference_out, name_list):
        count = 0
        for name in name_list or name.replace("_", " ") in name_list:
            if name in inference_out:
                gpt_pred = name
                count += 1
        if count != 1:
            return None
        else:
            return gpt_pred

    def Inference(self, inputs):
        pred_results = []
        
        for i in range(inputs.shape[0]):
            inference_out = self.Inference_one(inputs[i], self.query)
            # print(inference_out)
            gpt_pred = self.classification(inference_out, self.name_list)
            # print(gpt_pred)
            if gpt_pred is not None:
                pred_results.append(self.label2name.name2label(gpt_pred))
            else:
                pred_results.append(-1)
        # return torch.tensor(pred_results, device = inputs.device).reshape(-1,1)
        return pred_results
    
    def Inference_batch(self, inputs):
        user_message = "Describe the the given image."

        chatbot = []
        img_list = []  
        chat_state = self.CONV_VISION.copy()
        self.chat.upload_img(inputs, chat_state, img_list)
        self.chat.encode_img(img_list)
        self.chat.ask(user_message, chat_state)
        chatbot += [[user_message, None]]

        llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=1,
                                temperature=1.0,
                                max_new_tokens=300,
                                max_length=2000)[0]
        print(llm_message)
        chatbot[-1][1] = llm_message
        # 第二次prompt 问结果
        user_message = self.query
        self.chat.ask(user_message, chat_state)
        chatbot = [[user_message, None]]

        llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=1,
                                temperature=1.0,
                                max_new_tokens=300,
                                max_length=2000)[0]
        print(llm_message)
        chatbot[-1][1] = llm_message

        return chatbot[-1][1]
    

if __name__ == "__main__":
    args = parse_args()
    setup_seeds(args.seed)
    
    test_transform = transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor(), 
                 #transforms.Normalize(
                  #      (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
                ]
            )
    
    dataset = get_test_datasets(args, test_transform)
    
    dataset_size = len(dataset)

    # 计算要选择的数据的数量（例如，10%）
    subset_percentage = 0.1
    subset_size = int(dataset_size * subset_percentage)

    # 创建 SubsetRandomSampler，该采样器将随机选择数据集的子集
    subset_sampler = SubsetRandomSampler(range(subset_size))
    
    data_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=2, sampler=subset_sampler)
    
    minigpt = InferenceGPT(args)
    
    correct = 0
    fail_det = 0
    total = 0
    for data in tqdm(data_loader):
        # print(len(data))
        inputs, labels, _ = data
        inputs = inputs.cuda()
        
        minigpt_pred = minigpt.Inference(inputs)
        # minigpt_pred =minigpt.Inference_batch(inputs)
        
        for i, pred in enumerate(minigpt_pred):
            if pred == -1:
                fail_det +=1
                total += 1
            elif pred == labels[i]:
                correct += 1
                total += 1
        print('acc: ', correct/total*100, 'fail detect: ', fail_det/total*100, 'total: ', total)
                
    print('acc: ', correct/total*100, 'fail detect: ', fail_det/total*100, 'total: ', total)