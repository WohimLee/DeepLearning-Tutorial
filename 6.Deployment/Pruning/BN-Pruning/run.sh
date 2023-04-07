# Baseline
python train.py \
--dataset cifar10 \
--arch vgg \
--depth 11 \
--epochs 10


# Train with Sparsity
python train.py \
-sr --s 0.0001 \
--dataset cifar10 \
--arch vgg \
--depth 11 \
--epochs 10


# Prune
python prune.py \
--dataset cifar10 \
--depth 11 \
--percent 0.6 \
--model  logs/model_best.pth 


# Fine-tune
python train.py \
--refine  logs/pruned.pth \
--dataset cifar10 \
--arch vgg \
--depth 11 \
--epochs 10