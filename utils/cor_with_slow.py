import argparse
import os
import re
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList
from PIL import Image
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
import sys
sys.path.append(os.getcwd())
from datasets.label2name import LabelConverter
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
# PureMM
from PureMM.eval.model_vqa import *
from PureMM.model.conversation import conv_templates, SeparatorStyle
from PureMM.model.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

class Inference_with_slow():
    def __init__(self, dataset_name, args):
        self.k = args.k
        self.slow_model = args.slow_model
        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                    'pretrain_llama2': CONV_VISION_LLama2}

        print('Initializing Chat')
        # args = self.parse_args()
        cfg = Config(args)
        self.label2name = LabelConverter(dataset_name)
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

    # def parse_args(self):
    #     parser = argparse.ArgumentParser(description="Demo")
        
    #     args = parser.parse_args()
    #     return args
    
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
        # pre_des = entry[1]
        user_message = query#"With your describe, choose a most suitable one of the object in the image: Cat, Dog, Squirrel. Cat is the most impossible answer."#.format(entry[1])
        # user_message = 'This is your description:{}, '.format(pre_des)+user_message
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
    
    def formulate_query(self, fast_pred_topk):
        name_list_topk = self.label2name.convert(fast_pred_topk)
        fill = ', '.join(name_list_topk[:-1]) + ' or ' + name_list_topk[-1]
        query = 'From your description, answer which category the picture is belong to: {}? Please choose one most possible category.'.format(fill)
        return query, name_list_topk

    def classification(self, inference_out, name_list_topk):
        count = 0
        for name in name_list_topk or name.replace("_", " ") in name_list_topk:
            if name in inference_out:
                slow_pred = name
                count += 1
        if count != 1:
            return None
        else:
            return slow_pred

    def Inference_slow(self, inputs, fast_out):
        _, fast_pred_topk = torch.topk(fast_out, self.k, dim = -1)
        pred_results = []
        for i in range(fast_out.shape[0]):
            query, name_list_topk = self.formulate_query(fast_pred_topk[i])
            inference_out = self.Inference_one(inputs[i], query)
            slow_pred = self.classification(inference_out, name_list_topk)
            if slow_pred is not None:
                pred_results.append(self.label2name.name2label(slow_pred))
            else:
                # print(fast_out[i].shape)
                pred_results.append(torch.max(fast_out[i], dim=-1)[1].item())
        return torch.tensor(pred_results, device = fast_out.device).reshape(-1,1)
    
    
class InferenceWithSlowModel():
    def __init__(self, dataset_name, args):
        self.label2name = LabelConverter(dataset_name)
        disable_torch_init()
        self.slow_model = args.slow_model
        self.k = args.k
        self.args = args
        if args.slow_model == 'PureMM':
            model_path = os.path.expanduser(args.pure_model_path) #../model/PureMM_v1.0
            model_name = get_model_name_from_path(model_path)
            print(f'model_name: {model_name}')
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, args.pure_model_base, model_name)
            print('load PureMM model done!!!') #../model/vicuna-13b-v1.5
        elif args.slow_model == 'INF-MLLM':
            self.tokenizer = AutoTokenizer.from_pretrained(args.inf_model_path, use_fast=False)
            self.model = AutoModel.from_pretrained(args.inf_model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval() # trust_remote_code=True,   torch_dtype=torch.bfloat16
            # self.model = AutoModelForCausalLM.from_pretrained(args.inf_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
            self.image_processor = self.model.get_model().get_vision_tower().image_processor
            print('load INF-MLLM model done!!!')
        
    def Inference_one(self, input, query):
        input = transforms.Resize((336,336))(input)
        input = input.squeeze(0)
        input = transforms.ToPILImage()(input)
        
        if self.slow_model == 'PureMM':
            image_tensor = process_images([input], self.image_processor, self.model.config)[0]
            # # 计算张量的最小值和最大值
            # min_value = torch.min(image_tensor)
            # max_value = torch.max(image_tensor)

            # # 缩放张量到 [0,1] 范围内
            # image_tensor = (image_tensor - min_value) / (max_value - min_value)
        elif self.slow_model == 'INF-MLLM':
            # image_tensor = process_images([input], self.image_processor, self.model.config)[0]
            image_tensor = self.image_processor.preprocess(input, return_tensors='pt')['pixel_values'][0]
            # # 计算张量的最小值和最大值
            # min_value = torch.min(image_tensor)
            # max_value = torch.max(image_tensor)

            # # 缩放张量到 [0,1] 范围内
            # image_tensor = (image_tensor - min_value) / (max_value - min_value)
            
        cur_prompt = query
        qs = DEFAULT_IMAGE_TOKEN + '\n' + query
        conv = conv_templates[self.args.conv_mode] #vicuna_v1
        conv.messages = conv.messages[2:]
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # print(conv.messages)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        if self.slow_model == 'PureMM':
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True)
        elif self.slow_model == 'INF-MLLM':
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    # images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda'),
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if self.args.inf_temperature > 0 else False,
                    temperature=self.args.inf_temperature,
                    top_p=self.args.inf_top_p,
                    num_beams=self.args.inf_num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=128,
                    use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
    
    def formulate_query(self, fast_pred_topk):
        name_list_topk = self.label2name.convert(fast_pred_topk)
        fill = ', '.join(name_list_topk[:-1]) + ' or ' + name_list_topk[-1]
        query = 'Answer which class the picture is belong to: {}? Please choose one most possible class.'.format(fill)
        # query = 'If there is one category in following choices {}, please choose a most possible one.'.format(fill)
        # query = 'What is in this image?'
        return query, name_list_topk
    
    def classification(self, inference_out, name_list_topk):
        count = 0
        inference_out = re.sub(r'[^a-zA-Z]', ' ', inference_out)
        inference_out = re.sub(r'\s+', ' ', inference_out)
        print(inference_out)
        for name in name_list_topk: #  or name.replace("_", " ") in name_list_topk
            name_copy = name
            if name.replace("_", " ").lower() in inference_out.lower():
                slow_pred = name_copy
                count += 1
        if count != 1:
            return None
        else:
            print('slow pred: ', slow_pred)
            return slow_pred
        
    def Inference_slow(self, inputs, fast_out):
        # print(f'Now using {self.slow_model} !')
        _, fast_pred_topk = torch.topk(fast_out, self.k, dim = -1)
        pred_results = []
        for i in range(fast_out.shape[0]):
            query, name_list_topk = self.formulate_query(fast_pred_topk[i])
            inference_out = self.Inference_one(inputs[i], query)
            slow_pred = self.classification(inference_out, name_list_topk)
            if slow_pred is not None:
                pred_results.append(self.label2name.name2label(slow_pred))
            else:
                # print(fast_out[i].shape)
                pred_results.append(torch.max(fast_out[i], dim=-1)[1].item())
        return torch.tensor(pred_results, device = fast_out.device).reshape(-1,1)

# slow_agent = Inference_with_slow('seq-imagenet-r', k=3)
# img = Image.open('/home/bqqi/lifelong_research/errorcase/00_error_1.png').convert('RGB')
# transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
# img = transform(img).unsqueeze(0)
# fast_out = torch.ones((1,200))
# fast_out[:2]=100
# pred = slow_agent.Inference_slow(img, fast_out)
# pred_text = LabelConverter('seq-imagenet-r').convert(pred)
# print(pred_text)



