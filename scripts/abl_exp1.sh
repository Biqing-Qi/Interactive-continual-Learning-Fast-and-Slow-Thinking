#!/bin/bash
#SBATCH -J abl_exp21add       # 作业名为 test
#SBATCH -o abl_exp21add.out   # 屏幕上的输出⽂件重定向到 test.out
#SBATCH -p compute    # 作业提交的分区为 compute
#SBATCH -N 1          # 作业申请 1 个节点
#SBATCH -t 04:00:00    # 任务运⾏的最⻓时间为 1 ⼩时
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 # 申请GPU
#SBATCH -w gpu07      # 指定运⾏作业的节点是 gpu06，若不填写则不指定

# 输入要执行的命令，例如 ./hello 或 python test.py 等
# python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 2 --lmbda 0.1 --delta 0.01 --k 2 --with_slow > ablation_exp/exp2/cifar10_k_2.log 2>&1
# python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 2 --lmbda 0.1 --delta 0.01 --k 3 --with_slow > ablation_exp/exp2/cifar10_k_3.log 2>&1
# python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 2 --lmbda 0.1 --delta 0.01 --k 4 --with_slow > ablation_exp/exp2/cifar10_k_4.log 2>&1
python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 2 --lmbda 0.1 --delta 0.01 --k 5 --with_slow > ablation_exp/exp2/cifar10_k_5.log 2>&1

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --num_classes 10 --num_workers 12 --vit_finetune

python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 600  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 2 --lmbda 0.1 --delta 0.01 --k 2

CUDA_VISIBLE_DEVICES='0' python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 4 --with_slow --slow_model PureMM

CUDA_VISIBLE_DEVICES='1' python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 500  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 4 --with_slow --slow_model PureMM

CUDA_VISIBLE_DEVICES='0' python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 600  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 4 --with_slow --slow_model PureMM

# 
CUDA_VISIBLE_DEVICES='0' python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 4 --with_slow --slow_model INF-MLLM

CUDA_VISIBLE_DEVICES='1' python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 500  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 4 --with_slow --slow_model INF-MLLM

CUDA_VISIBLE_DEVICES='0' python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 600  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 4 --with_slow --slow_model INF-MLLM