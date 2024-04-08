Smooth Variational Graph Embeddings for Efficient Neural Architecture Search
===============================================================================

["Smooth Variational Graph Embeddings for Efficient Neural Architecture Search"](https://arxiv.org/abs/2010.04683)\
*Jovita Lukasik, David Friede, Arber Zela, Frank Hutter, Margret Keuper*.\
IJCNN 2021

**Note: our jovitalukasik/SVGe repo has been moved to [automl/SVGe](https://github.com/automl/SVGe), and this repo is not maintained. Please use [automl/SVGe](https://github.com/automl/SVGe).**

## Abstract 
Neural architecture search (NAS) has recently been addressed from various directions, including discrete, sampling-based methods and efficient differentiable approaches. While the former are notoriously expensive, the latter suffer from imposing strong constraints on the search space. Architecture optimization from a learned embedding space for example through graph neural network based variational autoencoders builds a middle ground and leverages advantages from both sides. Such approaches have recently shown good performance on several benchmarks. Yet, their stability and predictive power heavily depends on their capacity to reconstruct networks from the embedding space. In this paper, we propose a two-sided variational graph autoencoder, which allows to smoothly encode and accurately reconstruct neural architectures from various search spaces. We evaluate the proposed approach on neural architectures defined by the ENAS approach, the NAS-Bench-101 and the NAS-Bench-201 search space and show that our smooth embedding space allows to directly extrapolate the performance prediction to architectures outside the seen domain (e.g. with more operations). Thus, it facilitates to predict good network architectures even without expensive Bayesian optimization or reinforcement learning.

## Installation 
In order to use this code, install requirements:
```bash
pip install -r requirements.txt
```

## Datasets 
You can download the prepared ENAS dataset  [here](https://drive.google.com/file/d/1_BJLYq-QFhbv5_-xCPkGc6t4Im7hDbLB/view?usp=sharing)  and place them to ``datasets/ENAS``\
You can download the prepared NAS-Bench-101 dataset  [here](https://drive.google.com/file/d/1kRnBNv4UoF7GKQsgy0BXmHypor5CJLj4/view?usp=sharing) and place them to ``datasets/nasbench101``\
You can download the prepared NAS-Bench-B201 dataset  [here](https://drive.google.com/file/d/1rPhQrDH_r8zyfoxfYpz4CieCNTaVmRT9/view?usp=sharing) and place them to ``datasets/nasbench201``


Also download [nasbench_only108.tfrecord](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord) and save it to ``datasets/nasbench101`` and save *NAS-Bench-201-v1_0-e61699.pth* available on the [NAS-Bench-201 github page](https://github.com/D-X-Y/NAS-Bench-201?tab=readme-ov-file) in ``datasets/nasbench201``


Pretrained VAE models:

Load [pretrained state dicts](https://drive.google.com/file/d/1Te2Achfx9AZZooSoYNNd73Q9gJ1ahSJb/view?usp=sharing) to folder ``state_dicts`` 

## Run experiments from the paper 
### Vae training:
```bash
export PYTHONPATH=$PWD
python Training_VAE/train_svge.py --model SVGE --data_search_space NB101 
```
The model can be changed to DGMG, and the data_search space to NB201 or ENAS.

### Performance Prediction:
```bash
export PYTHONPATH=$PWD
python Performance_Prediction/train_surrogate_model.py --model SVGE_acc --data_search_space NB101 
```

### Extrapolation:
```bash
Extrapolation_Ability/eval_extrapolation.py --model SVGE_acc  --data_search_space NB101 --path_state_dict state_dicts/SVGE_acc_NB101/
```
change data_search_space to ENAS and also path_state_dict state_dicts/SVGE_acc_ENAS/

To train the 12-layer ENAS architecture:
cd Bayesian_Optimization and change the flat enas architcture represetation in arcs_scores in `fully_train_ENAS12.py` 
```bash 
python fully_train_ENAS12.py
```
To train the best NAS-Bench-101 cell include adjacency matrix and node operations in `Extrapolation_Ability/cell_training/train_cell.py`
To train certain cell on CIFAR10
```bash 
python train_cell.py --data_set cifar10
```
to train on ImageNet16-120, first download Imagenet16 from [NAS-Bench-201 repo](https://github.com/D-X-Y/NATS-Bench) and save it in 'Extrapolation_Ability/cell_training/data/'
```bash 
python train_cell.py --data_set Imagenet16 --batch_size 256 --epochs 200 --val_portion 0.5
```

## Bayesian Optimization 
For performing Bayesian optimzation, first follow the steps from the official [D-VAE repository](https://github.com/muhanzhang/D-VAE) \
Change to directory `Bayesian Optimization` 
and run for BO on NAS-Bench-101
```bash
./run_bo_Nb101.sh 
```
and 
```bash
./run_bo.sh
```
to run BO on ENAS


## Citation
```bash
@inproceedings{LukasikSVGe2021,
  author    = {Jovita Lukasik and
               David Friede and
               Arber Zela and
               Frank Hutter and
               Margret Keuper},
  title     = {Smooth Variational Graph Embeddings for Efficient Neural Architecture
               Search},
  booktitle = {International Joint Conference on Neural Networks, {IJCNN} 2021, Shenzhen,
               China, July 18-22, 2021},
  year      = {2021},
}
```

## Reference

- [D-VAE: A Variational Autoencoder for Directed Acyclic Graphs, Advances in Neural Information Processing Systems (NeurIPS 2019)](https://github.com/muhanzhang/D-VAE)
- [Does Unsupervised Architecture Representation Learning Help Neural Architecture Search? (NeurIPS 2020)](https://github.com/MSU-MLSys-Lab/arch2vec)
- [NAS-BENCH-201: Extending the Scope of Reproducible Neural Architecture Search(ICLR S2020)](https://github.com/D-X-Y/NATS-Bench)
