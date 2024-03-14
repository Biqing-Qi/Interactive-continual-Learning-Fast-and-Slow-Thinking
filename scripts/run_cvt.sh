#!/bin/bash
#SBATCH -J run_cvt       # 作业名为 test
#SBATCH -o run_cvt.out   # 屏幕上的输出⽂件重定向到 test.out
#SBATCH -p compute    # 作业提交的分区为 compute
#SBATCH -N 1          # 作业申请 1 个节点
#SBATCH -t 16:00:00    # 任务运⾏的最⻓时间为 1 ⼩时
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1 # 申请GPU
#SBATCH -w gpu10      # 指定运⾏作业的节点是 gpu06，若不填写则不指定

# 输入要执行的命令，例如 ./hello 或 python test.py 等
python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 200 --imsize 224 --csv_log --num_classes 10 --num_workers 12

python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 200 --imsize 224 --csv_log --num_classes 100 --num_workers 12

python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-animals --buffer_size 200 --imsize 224 --csv_log --num_classes 36 --num_workers 12

python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 200 --imsize 224 --csv_log --num_classes 200 --num_workers 12

python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar10 --buffer_size 500 --imsize 224 --csv_log --num_classes 10 --num_workers 12

python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-cifar100 --buffer_size 500 --imsize 224 --csv_log --num_classes 100 --num_workers 12

python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-animals --buffer_size 500 --imsize 224 --csv_log --num_classes 36 --num_workers 12

python /home/bqqi/lifelong_research/src/CL_Transformer/utils/main.py --model onlinevt --load_best_args --dataset seq-imagenet-r --buffer_size 500 --imsize 224 --csv_log --num_classes 200 --num_workers 12