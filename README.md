# Kitchenware image classifier (PyTorch)

Image classifier for six kitchenware classes (**cup, glass, plate, spoon, fork, knife**), built for the *Advanced Machine Learning* course at the University of León (2025-26). Group project of three students.

The goal was to see how far a small CNN trained from scratch can go, how much regularisation and optimisation help, and how much transfer learning changes the picture. Everything is measured with **macro F1** on five datasets.

## Datasets

| Name | Content |
|---|---|
| OWN | Photos taken by the group (371 training images) |
| ODD / EVEN | Two class-wide datasets built from photos contributed by the class |
| ODD+EVEN | ODD and EVEN together (4,745 training images, 1,028 test images) |
| Public | A public Kaggle kitchenware dataset (3,891 training images, 834 test images) |

The datasets are not included in this repository. The scripts expect the standard `ImageFolder` layout: `dataset/train/<class>/`, `dataset/val/<class>/`, `dataset/test/<class>/`.

## Experiments and results (test macro F1)

| Experiment | OWN | ODD | EVEN | ODD+EVEN | Public |
|---|---|---|---|---|---|
| 1. Baseline CNN from scratch | 0.685 | 0.516 | 0.621 | 0.635 | 0.697 |
| 2. Baseline + regularisation / optimisation / loss | n/a | n/a | n/a | 0.673 | n/a |
| 3. ConvNeXt-Tiny, pretrained on ImageNet | 1.000 | 0.982 | 0.990 | 0.993 | 0.964 |
| 4. ResNet-50, pretrained on ImageNet | 1.000 | 0.981 | 0.988 | 0.986 | 0.984 |

Experiment 2 was run on ODD+EVEN only. The OWN test set has just 84 images, so the 1.000 there is not very informative.

### Experiment 1: baseline

Three convolutional blocks (Conv, BatchNorm, ReLU, MaxPool) followed by a dense layer with dropout 0.5, about 25.8 M parameters. SGD, learning rate 0.01, batch size 16, 20 epochs, images resized to 224x224. Training curves show clear overfitting after roughly epoch 12.

### Experiment 2: what helps a small CNN

Each team member studied one family of changes on ODD+EVEN, one change at a time (`results/tables/exp2_ablation.csv`):

| Family | Change | Macro F1 | vs. baseline (0.6346) |
|---|---|---|---|
| Regularisation (teammate) | dropout 0.4 | 0.6635 | +0.019 |
| Regularisation (teammate) | weight decay 1e-5 | 0.6488 | +0.004 |
| Regularisation (teammate) | light data augmentation | 0.5739 | -0.071 |
| Optimisation (teammate) | Adam lr 0.001 | 0.5200 | -0.124 |
| Optimisation (teammate) | ReduceLROnPlateau | 0.6595 | +0.015 |
| Optimisation (teammate) | batch size 64 | 0.6713 | +0.027 |
| Loss function (Pablo Merino) | label smoothing 0.1 | 0.6534 | +0.009 |
| Loss function (Pablo Merino) | focal loss, gamma 1.0 | 0.6574 | +0.013 |

Combining the best changes (batch size 64, focal loss gamma 1.0, dropout 0.4, weight decay 1e-5, ReduceLROnPlateau) gives 0.6726 macro F1 on ODD+EVEN (+0.038 over the baseline).

### Experiments 3 and 4: transfer learning

A hyperparameter sweep on ODD+EVEN picked the settings for **ConvNeXt-Tiny** (27.8 M parameters, learning rate 1e-4, batch size 32, 10 epochs, label smoothing 0.02), which was then trained on every dataset. A **ResNet-50** (23.5 M parameters, learning rate 5e-5, batch size 32, 8 epochs) was run the same way. Both reach roughly 0.96-1.00 macro F1 everywhere, against 0.52-0.70 for the CNN trained from scratch.

## How to read these results

- The jump from about 0.65 to about 0.99 comes mainly from ImageNet pretraining, not from a cleverer design.
- Test sets are small: 84 images for OWN, 500-528 for ODD and EVEN, 1,028 for ODD+EVEN and 834 for Public. Differences of one or two points between experiments 3 and 4 are within noise.
- This repository does not include a check for near-duplicate photos across the train and test splits, so very high scores should be read with that in mind.

## Repository layout

```
src/
  baseline_cnn.py        BaselineCNN model
  run_baseline.py        training and evaluation of the baseline
  run_experiment1.py     Experiment 1 runner (metrics, plots, logs, tables)
results/
  tables/                results of experiments 2, 3 and 4 (CSV)
  figures/               confusion matrices and training curves
```

The code for experiments 2 to 4 is not included in this repository; only their results are.

## Running the baseline

```bash
pip install -r requirements.txt
# put your dataset under src/<DATASET_FOLDER_NAME>/{train,val,test}/<class>/
python src/run_experiment1.py
```

`run_experiment1.py` requires a CUDA GPU by default (`REQUIRE_CUDA = True`); set it to `False` to run on CPU.
