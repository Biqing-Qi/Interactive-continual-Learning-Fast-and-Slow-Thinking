import torch
import torch.nn as nn
from torch.nn import MultiLabelSoftMarginLoss
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
# from utils.cor_with_slow import Inference_with_slow
from copy import deepcopy
import numpy as np
import pdb
from datasets import get_dataset
from pytorch_metric_learning import losses as torch_losses
from losses.SupConLoss import SupConLoss
from kornia.augmentation import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
)

def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return trainable_num


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Online continual learning via self-supervised Transformer"
    )
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument("--alpha", type=float, required=True, help="Penalty weight.")
    parser.add_argument("--beta", type=float, required=True, help="Penalty weight.")
    return parser


class Onlinevt(ContinualModel):
    NAME = "onlinevt"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Onlinevt, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, args)
        self.soft_loss = MultiLabelSoftMarginLoss()
        self.dataset = get_dataset(args)
        self.use_screening = args.use_screening
        if args.vit_finetune:
            self.total_num_class = backbone.num_classes
        else:
            self.total_num_class = backbone.net.num_classes
        self.with_brain = args.with_brain
        self.with_brain_vit = args.with_brain_vit
        self.vit_finetune = args.vit_finetune
        self.gamma = None
        self.old_net = None
        self.current_task = 0
        self.iter = 0
        self.l2_current = False
        self.print_freq = 500
        self.descending = False
        self.MSE = False
        self.BCE = True
        self.use_l1_change = True
        self.class_means = None
        self.fish = None
        self.temperature = 2
        self.n_views = 2
        self.min_gradnorm = torch.tensor([1e7]*self.total_num_class)
        # print(self.min_gradnorm.shape)
        self.task_now = 0
        self.logsoft = nn.LogSoftmax(dim=1)
        self.MSloss = torch_losses.MultiSimilarityLoss(alpha=2, beta=10, base=0.5)
        self.ce_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.sc_loss = SupConLoss(temperature=self.temperature)
        self.mse = torch.nn.MSELoss().to(self.device)
        self.task_now_f = 0
    
    def observe(self, inputs, labels, not_aug_inputs, mem_input=None, obs_num = None):
        if self.with_brain:

            self.buffer.add_data(
                    examples=not_aug_inputs, labels=labels, logits=None
                )
            if not hasattr(self, "classes_so_far"):
                self.register_buffer("classes_so_far", labels.unique().to("cpu"))
            else:
                self.register_buffer(
                    "classes_so_far",
                     torch.cat((self.classes_so_far, labels.to("cpu"))).unique(),
                )

            for step in range(2):
                if step == 0:
                    self.opt_mem.zero_grad()
                    
                    loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
                    loss.backward()

                    self.opt_mem.step()
                else:
                    self.opt_other.zero_grad()
                    
                    loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
                    loss.backward()

                    self.opt_other.step()
        
        elif self.with_brain_vit:
            if self.task_now != self.net.net.task_now:
                self.task_now = self.net.net.task_now
                self.min_gradnorm = torch.tensor([1e7]*self.total_num_class)
            if not self.use_screening:
                self.buffer.add_data(
                        examples=not_aug_inputs, labels=labels, logits=None
                    )
            if not hasattr(self, "classes_so_far"):
                self.register_buffer("classes_so_far", labels.unique().to("cpu"))
            else:
                self.register_buffer(
                    "classes_so_far",
                     torch.cat((self.classes_so_far, labels.to("cpu"))).unique(),
                )
            
            if obs_num<60:
                for step in range(2):
                    if step == 0:
                        self.opt_brain.zero_grad()
                        #self.opt_mem.zero_grad()
                        
                        if self.use_screening:
                            inputs.requires_grad = True
                            loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
                            grad = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
                            loss.backward()
                            gradnorm = torch.mean(torch.norm(grad, p='fro', dim = (2,3)), dim=1)
                            
                            label_grads = torch.index_select(self.min_gradnorm.to(labels.device), 0, labels)
                            true_ids = torch.where(gradnorm < label_grads)[0]
                            
                            # self.min_gradnorm = torch.min(gradnorm).item()
                            # print((add_mask.view(-1, 1, 1, 1)*torch.zeros_like(not_aug_inputs)).shape, add_mask.shape, not_aug_inputs.shape)
                            masked_inputs = torch.index_select(not_aug_inputs, 0,  true_ids)
                            masked_labels = torch.index_select(labels, 0, true_ids)
                            
                            if masked_labels.shape[0] > 0:
                                label_num = 0
                                for label in masked_labels:
                                    self.min_gradnorm[label] = gradnorm[true_ids[label_num]]
                                    label_num+=1
                                # print(self.min_gradnorm)
                                self.buffer.add_data(
                                examples=masked_inputs, labels=masked_labels, logits=None, abs_add=True
                            )
                        else:
                            loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
                            loss.backward()
                        #torch.norm(inputs.grad)
                        #torch.nn.utils.clip_grad_norm_(self.net.net., max_grad_norm)
                        #self.opt_mem.step()
                        self.opt_brain.step()
                    else:
                        self.opt_proj.zero_grad()
                        
                        loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
                        loss.backward()
                        
                        self.opt_proj.step()
            else:
                self.opt_proj.zero_grad()
                #self.opt_mem.zero_grad()
                #self.opt_mem.step()
                if self.use_screening:
                    inputs.requires_grad = True
                    loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
                    grad = torch.autograd.grad(loss, inputs, retain_graph=True)[0]
                    gradnorm = torch.mean(torch.norm(grad, p='fro', dim = (2,3)), dim=1)
                    
                    loss.backward()
                    label_grads = torch.index_select(self.min_gradnorm.to(labels.device), 0, labels)
                    true_ids = torch.where(gradnorm < label_grads)[0]
                    # self.min_gradnorm = torch.min(gradnorm).item()
                    masked_inputs = torch.index_select(not_aug_inputs, 0, true_ids)
                    masked_labels = torch.index_select(labels, 0, true_ids)
                    if masked_labels.shape[0] > 0:
                        label_num = 0
                        for label in masked_labels:
                            self.min_gradnorm[label] = gradnorm[true_ids[label_num]]
                            label_num += 1
                        self.buffer.add_data(
                        examples=masked_inputs, labels=masked_labels, logits=None, abs_add=True
                    )
                else:
                    loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
                    loss.backward()
                self.opt_proj.step()

        elif self.vit_finetune:
            
            self.buffer.add_data(
                    examples=not_aug_inputs, labels=labels, logits=None
                )
            if not hasattr(self, "classes_so_far"):
                self.register_buffer("classes_so_far", labels.unique().to("cpu"))
            else:
                self.register_buffer(
                    "classes_so_far",
                     torch.cat((self.classes_so_far, labels.to("cpu"))).unique(),
                )

            self.opt_f.zero_grad()
            loss = self.get_loss(inputs, labels, not_aug_inputs)
            loss.backward()
            self.opt_f.step()
        
        else:
            self.opt.zero_grad()
            if not hasattr(self, "classes_so_far"):
                self.register_buffer("classes_so_far", labels.unique().to("cpu"))
            else:
                self.register_buffer(
                    "classes_so_far",
                    torch.cat((self.classes_so_far, labels.to("cpu"))).unique(),
                )

            loss = self.get_loss(inputs, labels, not_aug_inputs, mem_input)
            loss.backward()

            self.opt.step()

        return loss.item()

    def ncm(self, x):
        with torch.no_grad():
            self.compute_class_means()

        feats = self.net.net.contrasive_f(x)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def one_hot(self, label):
        y_onehot = torch.FloatTensor(label.shape[0], self.total_num_class).to(
            self.device
        )
        y_onehot.zero_()
        y_onehot.scatter_(1, label.unsqueeze(1), 1)
        return y_onehot

    def Brain_co_loss(self, logit_brain_mem, logit_true, labels, y_history, y_brain_mem, alpha=0.4, gamma=1e-5):
        labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #print(labels,labels_mask)
        sum_pos_brain_mem = torch.sum(labels_mask * torch.exp(logit_brain_mem))
        denominator_brain_mem = gamma + torch.sum(torch.exp(logit_brain_mem))

        sum_pos_true = torch.sum(labels_mask * torch.exp(logit_true))
        denominator_true = gamma + torch.sum(torch.exp(logit_true))

        loss_ce_a = -torch.log(sum_pos_brain_mem / denominator_brain_mem)
        loss_ce_b = -torch.log(sum_pos_true / denominator_true)
        loss_mse = alpha*self.mse(y_history, y_brain_mem)
        #print('CEA:', loss_ce_a, 'CEB:', loss_ce_b, 'MSE:', loss_mse)
        loss = loss_ce_a + loss_ce_b + loss_mse
        return loss
    
    def vmf_loss(self, pos_logits, neg_logits, gamma):
        pre_logits = neg_logits
        pre_logits = pre_logits-torch.diag_embed(pre_logits.diag())
        logits = pre_logits + torch.diag_embed(pos_logits.diag())
        #print(logits.shape)
        diag_exp = logits.diag()
        row_exp_sum = logits.sum(dim=-1)
        p = diag_exp/(row_exp_sum+gamma)
        loss = torch.mean(-torch.log(p))
        #print(loss)
        #print(logits[0])
        #print(torch.max(diag_exp), torch.max(row_exp_sum), torch.max(p))
        return loss

    def margin_loss(self, pos_logits, neg_logits, delta):
        margin_pos = torch.abs(pos_logits-1) - delta
        margin_neg = torch.abs(neg_logits+1) - delta
        margin_pos = margin_pos * (margin_pos > 0).float()
        margin_neg = margin_neg * (margin_neg > 0).float()
        return margin_pos, margin_neg

    def Brain_co_loss_vit(self, logit_brain_mem, labels, gamma=1e-5, delta=0.1, lmbda = 0.1, kappa=1):
        labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))
        bs = labels.shape[0]
        #print(labels,labels_mask)
        #print(logit_brain_mem[labels_mask].shape)
        
        pos_logits = torch.masked_select(logit_brain_mem, labels_mask)#labels_mask * torch.exp(logit_brain_mem)
        neg_logits = torch.masked_select(logit_brain_mem, neg_mask)
        
        pos = torch.exp(logit_brain_mem/kappa) * labels_mask
        neg = torch.exp(logit_brain_mem/kappa) * neg_mask
        #print(neg)
        #print(pos, neg)
        #neg_logits = logit_brain_mem - pos_logits
        '''sum_pos_brain_mem = torch.sum(torch.exp(pos_logits))
        denominator_brain_mem = gamma + torch.sum(torch.exp(logit_brain_mem))
        
        loss_ce = -torch.log(sum_pos_brain_mem / denominator_brain_mem)'''
        #pos_pull_vector = torch.ones_like(pos_logits, device = pos_logits.device)
        #neg_pull_vector = -torch.ones_like(neg_logits, device = pos_logits.device)
        loss_vmf = self.vmf_loss(pos, neg, gamma)
        pos_margin, neg_margin = self.margin_loss(pos_logits, neg_logits, delta)
        
        loss_eq = torch.sum(pos_margin) + torch.sum(neg_margin)
        loss_eq = loss_eq/bs
        #loss_eq = 2-torch.cos(torch.dot(pos_logits/torch.norm(pos_logits), pos_pull_vector/torch.norm(pos_pull_vector))) + torch.cos(torch.dot(neg_logits/torch.norm(neg_logits), neg_pull_vector/torch.norm(neg_pull_vector)))
        #print('CEA:', loss_ce_a, 'CEB:', loss_ce_b, 'MSE:', loss_mse)
        loss = loss_vmf + lmbda*loss_eq
        #print(loss)
        #print(loss_vmf)
        # pos_pull_vector = torch.ones_like(pos_logits, device = pos_logits.device)
        # neg_pull_vector = -torch.ones_like(neg_logits, device = pos_logits.device)
        # loss = self.po_trip(pos_logits.unsqueeze(0), pos_pull_vector.unsqueeze(0)) + self.po_trip(neg_logits.unsqueeze(0), neg_pull_vector.unsqueeze(0))
        return loss

    def Brain_co_loss_tsk(self, logit_brain_mem, labels, gamma=1e-7):
        labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))
        bs = labels.shape[0]
        #print(labels,labels_mask)
        #print(logit_brain_mem[labels_mask].shape)
        
        
        pos = torch.exp(logit_brain_mem) * labels_mask
        neg = torch.exp(logit_brain_mem) * neg_mask
        loss_vmf = self.vmf_loss(pos, neg, gamma)
        loss = loss_vmf 

        return loss

    def poincare_distance(self,outputs,target_labels):
        
        outputs_norm_l1 = torch.norm(outputs,p=1,dim=1,keepdim=True)
        u = outputs/(outputs_norm_l1)
        v = F.normalize(target_labels, dim=1)*0.9
        delta = self.delta(u,v)
        poincare = torch.arccosh(1+delta)
        return poincare.squeeze(1)
    
    def delta(self,u,v):
        
        delta = 2*(torch.sum((u-v)**2,dim=1,keepdim=True)/((1-torch.sum(u**2,dim=1,keepdim=True))*(1-torch.sum(v**2,dim=1,keepdim=True))))
        return delta
    
    def triplet_distance(self,outputs,target_labels):
        
        D_tar = 1-torch.abs(torch.diag(torch.matmul(outputs,target_labels.t())))/(torch.norm(outputs,p=2,dim=1)*torch.norm(target_labels,p=2,dim=1))
        gamma = 0.007
        triplet_loss = torch.clamp(D_tar + gamma, min=0, max=1)
        return triplet_loss
    
    def po_trip(self,outputs,target_labels):
        po_trip_loss = self.poincare_distance(outputs,target_labels)#+0.01*self.triplet_distance(outputs, target_labels)
        
        return po_trip_loss.mean()

    def anchor(self, all_memories):
        product = torch.matmul(all_memories, all_memories.t())

        non_diagonal = product - torch.diag(torch.diagonal(product))
        margin_neg = torch.abs(non_diagonal + np.cos(-1/1023))-0.1
        margin_neg = margin_neg * (margin_neg > 0).float()
        sum_non_diagonal = torch.mean(margin_neg)
        return sum_non_diagonal

    def logits_brain_loss(self, logits, labels):
        loss = 0.0
        for logit in logits:
            loss += self.Brain_co_loss_vit(logit, labels)
        return loss

    def get_loss(self, inputs, labels, not_aug_inputs, mem_input=None, return_grad = False):
        if self.with_brain:
            brain_mem_input = torch.index_select(mem_input, 0, labels)
            inputs_aug = self.dataset.TRANSFORM_SC(inputs)
            inputs = torch.cat([inputs.unsqueeze(1), brain_mem_input.unsqueeze(1)], dim=1)
            inputs_aug = torch.cat([inputs_aug.unsqueeze(1), brain_mem_input.unsqueeze(1)], dim=1)
            inputs = torch.cat([inputs, inputs_aug], dim=0)
            labels = torch.cat([labels, labels])
            
            if not self.buffer.is_empty():
                buf_inputs, buf_labels,  _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                brain_mem_input = torch.index_select(mem_input, 0, buf_labels)
                buf_inputs_aug = self.dataset.TRANSFORM_SC(buf_inputs)
                buf_inputs = torch.cat([buf_inputs.unsqueeze(1), brain_mem_input.unsqueeze(1)], dim=1)
                buf_inputs_aug = torch.cat([buf_inputs_aug.unsqueeze(1), brain_mem_input.unsqueeze(1)], dim=1)
                inputs = torch.cat([inputs, buf_inputs, buf_inputs_aug], dim=0)
                labels = torch.cat([labels, buf_labels, buf_labels])
                #buf_logit_brain_mem, buf_logit_true, buf_y_history, buf_y_brain_mem = self.net(buf_inputs, buf_labels)
                
            logit_brain_mem, logit_true, y_history, y_brain_mem = self.net(inputs, labels)
            loss = self.Brain_co_loss(logit_brain_mem, logit_true, labels, y_history, y_brain_mem)
            #loss += self.Brain_co_loss(buf_logit_brain_mem, buf_logit_true, buf_labels, buf_y_history, buf_y_brain_mem)
            if self.iter % self.print_freq == 0:
                print("total loss: ", loss)
            self.iter += 1
            
            return loss
    
        elif self.with_brain_vit:
            inputs_aug = self.dataset.TRANSFORM_SC(inputs)
            #print(inputs.shape, inputs_aug.shape)
            inputs = torch.cat([inputs, inputs_aug], dim=0)
            labels = torch.cat([labels, labels])
            logit_brain_mem_cls, logits, all_memories = self.net(inputs, labels)
            loss = self.Brain_co_loss_vit(logit_brain_mem_cls, labels, gamma=1e-6, delta=self.args.delta, lmbda=self.args.lmbda, kappa = self.args.kappa) #+ 0.1*self.anchor(all_memories)#+ self.Brain_co_loss_vit(all_memories[0], all_memories[1])#+ self.logits_brain_loss(logits, labels)#+ 0.1*self.anchor(all_memories)

            if not self.buffer.is_empty() and self.net.net.task_now>=1:
                #print(self.buffer)
                buf_inputs, buf_labels,  _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                buf_inputs_aug = self.dataset.TRANSFORM_SC(buf_inputs)
                inputs_buf = torch.cat([buf_inputs, buf_inputs_aug], dim=0)
                labels_buf = torch.cat([buf_labels, buf_labels])
                logit_brain_mem_cls_buf, logits_buf, all_memories_buf = self.net(inputs_buf, labels_buf)
                loss += 1*self.Brain_co_loss_vit(logit_brain_mem_cls_buf, labels_buf, gamma=1e-6, delta=self.args.delta, lmbda=self.args.lmbda, kappa = self.args.kappa)#+ self.Brain_co_loss_vit(all_memories_buf[0], all_memories_buf[1]))
            #print(inputs.shape)
            
            #print(self.Brain_co_loss_vit(logit_brain_mem_tsk[0], logit_brain_mem_tsk[1]))
            
            
            if self.iter % self.print_freq == 0:
                print("total loss: ", loss)
            self.iter += 1
            return loss

        elif self.vit_finetune:
            inputs_aug = self.dataset.TRANSFORM_SC(inputs)
            inputs = torch.cat([inputs, inputs_aug], dim=0)
            labels = torch.cat([labels, labels])
            
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
                
            if not self.buffer.is_empty() and self.task_now_f>=1:
                #print(self.buffer)
                buf_inputs, buf_labels,  _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                buf_inputs_aug = self.dataset.TRANSFORM_SC(buf_inputs)
                inputs_buf = torch.cat([buf_inputs, buf_inputs_aug], dim=0)
                labels_buf = torch.cat([buf_labels, buf_labels])
                # inputs = torch.cat([inputs, inputs_buf], dim=0)
                # labels = torch.cat([labels, labels_buf])
                
                outputs_buf = self.net(inputs_buf)
                loss += self.loss(outputs_buf, labels_buf)
            
            if hasattr(self.args, "ce"):
                loss = loss * self.args.ce
                
            if self.iter % self.print_freq == 0:
                print("current task CE total loss: ", loss)
                
            self.iter += 1
            return loss
        
        else:
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            if self.net.net.distill_classifier:
                outputs = self.net.net.distill_classification(inputs)
                loss += (
                    self.loss(outputs, labels) * 0.3
                )  # self.args.distill_ce #/(self.current_task+1)

            if hasattr(self.args, "ce"):
                loss = loss * self.args.ce

            if self.iter % self.print_freq == 0:
                print("current task CE loss: ", loss)

            if hasattr(self.args, "wd_reg"):
                loss.data += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

            if self.old_net is not None and self.l2_current:
                old_output_features = self.old_net.features(inputs)
                features = self.net.features(inputs)
                loss += self.args.alpha * 0.8 * F.mse_loss(old_output_features, features)

            loss_sc = 0
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                if self.net.net.distill_classifier:
                    buf_outputs = self.net.net.distill_classification(buf_inputs)
                else:
                    buf_outputs = self.net(buf_inputs)
                # loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss += self.args.alpha * self.loss(buf_outputs, buf_labels)

                '''buf_inputs, buf_labels, _, buf_inputs_aug = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                if self.net.net.distill_classifier:
                    buf_outputs = self.net.net.distill_classification(buf_inputs)
                else:
                    buf_outputs = self.net(buf_inputs)
                loss += self.args.beta * self.loss(buf_outputs, buf_labels)'''

                combined_batch = torch.cat((buf_inputs, inputs))
                combined_labels = torch.cat((buf_labels, labels))
                # combined_batch = buf_inputs
                # combined_labels = buf_labels
                combined_batch_aug = self.dataset.TRANSFORM_SC(combined_batch)

                '''if self.net.net.distill_classifier:
                    buf_outputs = self.net.net.distill_classification(combined_batch_aug)
                else:
                    buf_outputs = self.net(combined_batch_aug)
                loss += self.args.beta * self.loss(buf_outputs, combined_labels)'''

                features_sc = torch.cat(
                    [
                        self.net.net.contrasive_f(combined_batch).unsqueeze(1),
                        self.net.net.contrasive_f(combined_batch_aug).unsqueeze(1),
                    ],
                    dim=1,
                )
                index = torch.randint(
                    0, len(self.classes_so_far), (np.minimum(10, len(self.classes_so_far)),)
                )
                focuses = self.net.net.focuses_head()[index]
                focus_labels = self.net.net.focus_labels[index]

                loss_sc = self.sc_loss(
                    features_sc, combined_labels#, focuses=focuses, focus_labels=focus_labels
                )

                loss += 0.1*loss_sc
            if self.iter % self.print_freq == 0:
                print("loss_sc: ", loss_sc)
                print("total loss: ", loss)
            self.iter += 1
            self.buffer.add_data(
                examples=not_aug_inputs, labels=labels, logits=outputs.data
            )
            return loss

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _ = self.buffer.get_all_data(transform)
        # examples, labels, _ = self.buffer.get_all_data()
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i] for i in range(0, len(examples)) if labels[i].cpu() == _y]
            ).to(self.device)

            index = torch.randint(
                0, len(self.classes_so_far), (np.minimum(10, len(self.classes_so_far)),)
            )

            focuses = self.net.net.focuses_head()[index]
            focus_labels = self.net.net.focus_labels[index]

            class_means.append(self.net.net.contrasive_f(x_buf).mean(0))

        self.class_means = torch.stack(class_means)

    def loss_trick(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction="mean")
        if self.params.trick["labels_trick"]:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick["separated_softmax"]:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ["SCR", "SCP"]:
            SC = SupConLoss(temperature=self.temperature)
            return SC(logits, labels)
        else:
            return ce(logits, labels)

    def info_nce_loss(self, features):
        # labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.n_views)], dim=0)
        labels = torch.cat(
            [torch.arange(features.shape[0] // 2) for i in range(self.n_views)], dim=0
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def end_task(self, dataset) -> None:
        if self.args.L1 > 0:
            self.old_net = deepcopy(self.net.eval())
            self.net.train()
        self.current_task += 1

    # fill buffer according to loss
    def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        ce_loss_raw = F.cross_entropy
        mode = self.net.training
        self.net.eval()
        # samples_per_class = mem_buffer.buffer_size // (self.dataset.N_CLASSES_PER_TASK * (t_idx + 1))
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)
        print("Classes so far:", len(self.classes_so_far))

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y, buf_f, buf_task_id = self.buffer.get_all_data()
            mem_buffer.empty()

            for _y in buf_y.unique():
                idx = buf_y == _y
                _y_x, _y_y, _y_f, _y_task_id = (
                    buf_x[idx],
                    buf_y[idx],
                    buf_f[idx],
                    buf_task_id[idx],
                )
                mem_buffer.add_data_our(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class],
                    logits=_y_f[:samples_per_class],
                    task_labels=_y_task_id[:samples_per_class],
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1 Extract all features
        a_x, a_y, a_logit, a_loss = [], [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x)
            a_y.append(y)
            outputs = self.net(x)
            a_logit.append(outputs)
            loss_raw = ce_loss_raw(outputs, y, reduction="none")
            a_loss.append(loss_raw)

        a_x, a_y, a_logit, a_loss = (
            torch.cat(a_x),
            torch.cat(a_y),
            torch.cat(a_logit),
            torch.cat(a_loss),
        )

        # 2.2 Compute class means
        for _y in a_y.unique():
            idx = a_y == _y
            _x, _y, _logit, _loss = a_x[idx], a_y[idx], a_logit[idx], a_loss[idx]
            _, index = _loss.sort(descending=self.descending)
            if samples_per_class < _x.shape[0]:
                index = index[:samples_per_class]

            mem_buffer.add_data_our(
                examples=_x[index].to(self.device),
                labels=_y[index].to(self.device),
                logits=_logit[index].to(self.device),
                task_labels=torch.tensor([t_idx] * len(index)).to(self.device),
            )

        assert len(mem_buffer.examples) <= mem_buffer.buffer_size

        self.net.train(mode)

    @torch.no_grad()
    def update(self, classifier, task_size):
        old_weight_norm = torch.norm(classifier.weight[:-task_size], p=2, dim=1)
        new_weight_norm = torch.norm(classifier.weight[-task_size:], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        print(self.gamma.cpu().item())

    @torch.no_grad()
    def post_process(self, logits, task_size):
        logits[:, -task_size:] = logits[:, -task_size:] * self.gamma
        return logits
