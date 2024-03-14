#!/bin/bash
#SBATCH -J vit2       # 作业名为 test
#SBATCH -o vit2.out   # 屏幕上的输出⽂件重定向到 test.out
#SBATCH -p compute    # 作业提交的分区为 compute
#SBATCH -N 1          # 作业申请 1 个节点
#SBATCH -t 12:00:00    # 任务运⾏的最⻓时间为 1 ⼩时
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 # 申请GPU
#SBATCH -w gpu07      # 指定运⾏作业的节点是 gpu06，若不填写则不指定


python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 200  --csv_log --num_classes 10 --num_workers 12 --vit_finetune
python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 200  --num_classes 100 --num_workers 12 --vit_finetune
python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 200  --num_classes 200 --num_workers 12 --vit_finetune