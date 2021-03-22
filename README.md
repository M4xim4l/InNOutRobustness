# Adversarial Robustness on In- and Out-Distribution Improves Explainability

This is the official repository for the ECCV paper Adversarial Robustness on In- and Out-Distribution Improves Explainability.

Before running the code, you will have to update the path variable in Line 8 in:

> utils.datasets.path

Cifar10/100 will be downloaded automatically, but you will need to manually place the 80 Million Tiny Images .bin file in a subdirectory:

> 80M Tiny Images 

To train a ResNet50 RATIO Cifar10 models with 10 steps for adversarial training and 20 ACET steps, you can run:

> python run_training_cifar10.py --gpu 0 --net resnet50 --epochs 250 --train_type AdvACET --augm autoaugment_cutout --id_steps 10 --od_steps 20

The gpu parameter supports multiple devices, for example if you want to train on GPUs 2 and 3:

> python run_training_cifar10.py --gpu 2 3 --net resnet50 --epochs 250 --train_type AdvACET --augm autoaugment_cutout --id_steps 10 --od_steps 20

Note that results can vary with different number of devices as this implicitly changes the number of samples used for updating the per-device
batch norm parameters.
