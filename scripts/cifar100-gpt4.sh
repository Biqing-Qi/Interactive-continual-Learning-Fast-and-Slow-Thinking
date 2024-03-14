#!/bin/bash
#SBATCH -J cifar100-gpt4       # 作业名为 test
#SBATCH -o cifar100-gpt4.out   # 屏幕上的输出⽂件重定向到 test.out
#SBATCH -p compute    # 作业提交的分区为 compute
#SBATCH -N 1          # 作业申请 1 个节点
#SBATCH -t 04:00:00    # 任务运⾏的最⻓时间为 1 ⼩时
#SBATCH --gres=gpu:a100-pcie-40gb:1 # 申请GPU
#SBATCH -w gpu21      # 指定运⾏作业的节点是 gpu06，若不填写则不指定

# 输入要执行的命令，例如 ./hello 或 python test.py 等
python gpt4ref.py --dataset seq-cifar100