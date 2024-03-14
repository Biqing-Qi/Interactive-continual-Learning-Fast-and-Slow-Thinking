#!/bin/bash
#SBATCH -J puremm200       # 作业名为 test
#SBATCH -o puremm200.out   # 屏幕上的输出⽂件重定向到 test.out
#SBATCH -p compute    # 作业提交的分区为 compute
#SBATCH -N 1          # 作业申请 1 个节点
#SBATCH -t 08:00:00    # 任务运⾏的最⻓时间为 1 ⼩时
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 # 申请GPU
#SBATCH -w gpu07      # 指定运⾏作业的节点是 gpu06，若不填写则不指定

# PureMM
# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model PureMM

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 500  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --hidden_dim 768 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model PureMM

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 600  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model PureMM

# InfMLLM
python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model INF-MLLM

python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 500  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --hidden_dim 768 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model INF-MLLM

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 600  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model INF-MLLM

# PureMM
# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 200  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model PureMM

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 200  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --hidden_dim 768 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model PureMM

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 200  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model PureMM

# InfMLLM
python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 200  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model INF-MLLM

python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 200  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --hidden_dim 768 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model INF-MLLM

python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 200  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 --with_slow --slow_model INF-MLLM

# with out slow
# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 500  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --hidden_dim 768 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 600  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5


# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 200  --csv_log --with_brain_vit --num_classes 10 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5 

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 200  --csv_log --with_brain_vit --num_classes 100 --num_workers 12 --hidden_dim 768 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5

# python /home/bqqi/ICL/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 200  --csv_log --with_brain_vit --num_classes 200 --num_workers 12 --kappa 1 --lmbda 0.1 --delta 0.01 --k 5