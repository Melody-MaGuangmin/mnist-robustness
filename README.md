# MNIST 平移与噪声联合增强实验

配套论文：马光敏，华北电力大学自动化系。

## 下载代码、权重和全部原始结果

[完整实验材料 mnist-experiments.zip](mnist-experiments.zip) 包含真实训练的 **20 份权重**：baseline、shift、noise、joint 各 5 个随机种子，包含老师要求的 baseline 与本文联合增强方法。压缩包还包含原始 MNIST 数据、训练/验证划分、训练代码、逐轮日志、逐样本预测、指标、7 张图和核验程序。

请下载并解压该压缩包，进入其中的 `mnist-robustness` 文件夹，再执行下文命令。网页根目录的代码与结果文件便于直接阅读；复现时使用压缩包中完整的目录结构。

```bash
cd mnist-robustness
python -m pip install -r requirements.txt
python verify.py
python audit_metrics.py
```

如需重新训练，先保留已下载的结果副本，再运行 `python train.py`。重新训练会覆盖同名权重。训练由 NumPy 独立实现，不依赖深度学习框架。

## 本次实际结果

四种方法在同一官方测试集上评估，每种方法重复 5 个优化种子。准确率为均值 ± 样本标准差。

| 方法 | 清洁图像 | 两像素平移 | 平移加噪声 σ=0.20 |
| --- | --- | --- | --- |
| baseline | 97.67 ± 0.18% | 81.37 ± 1.00% | 69.52 ± 3.36% |
| shift | 97.89 ± 0.09% | 96.30 ± 0.19% | 74.52 ± 2.42% |
| noise | 97.66 ± 0.07% | 80.84 ± 0.62% | 79.10 ± 0.93% |
| joint | 97.89 ± 0.12% | 96.28 ± 0.17% | 95.11 ± 0.17% |

联合增强在复合扰动上高于基线和两个单因素消融。仅平移方法在四像素平移中更好，仅噪声方法在 σ=0.30 纯噪声中略好；论文报告这些边界。所有方法均按四条件等权验证准确率选模，测试集未用于调参。五种子的标准差反映固定划分上的训练波动。

## 权重位置

完整材料内的 `weights/baseline_seed11.npz` 至 `baseline_seed55.npz` 为基线权重；`weights/joint_seed11.npz` 至 `joint_seed55.npz` 为本文方法权重，具体种子为 11、22、33、44、55。另有 shift、noise 两个消融方法的 10 份权重。`.npz` 内含 W1、b1、W2、b2、最佳轮次、验证分数、种子和方法名。

`verify.py` 重载全部 20 个模型、重新生成测试输入，并逐项核对 140 条指标及所有预测；`audit_metrics.py` 不导入训练代码，独立复算指标、汇总和测试输入，并抽查 8,960 条前向预测。两项检查实际运行均为 PASS。

## 完整实验说明

# MNIST robustness with translation and noise augmentation

This repository contains an actual, controlled training experiment on the **full MNIST dataset**: 60,000 official training images and 10,000 official test images, each 28×28 pixels. It is separate from the earlier 8×8 scikit-learn digits experiment. Its scores and checkpoints must not be substituted into that earlier experiment.

## Study protocol

`protocol.json` records the design fixed before training and test evaluation. All four methods use the same NumPy multilayer perceptron, 784–128–10 units with ReLU (**101,770 parameters**), Adam, and weight-matrix L2 regularization. There are five seeds (11, 22, 33, 44, 55) and four methods:

1. `baseline`: original training images.
2. `shift`: with probability 0.5, shift by two pixels in one uniformly chosen axial direction, zero padding.
3. `noise`: with probability 0.5, add Gaussian noise with standard deviation 0.15, clipped to [0, 1].
4. `joint`: apply the two independently sampled decisions in shift-then-noise order.

Pixel values are divided by 255. The validation set contains 500 original training images per class (5,000 total), selected by `default_rng(2026)`; the remaining 55,000 images form the training set. The 10,000 official test images are preserved. Split indices are released. The validation selection is image-level; it is not a newly constructed writer-disjoint training/validation split.

Every model trains for **10 epochs, 430 updates per epoch, 4,300 updates total**, with batch size 128 and learning rate 0.001. Within a seed, all methods have identical initial parameters and shuffle order. Both augmentation random streams are drawn for every method, even when their output is unused, so decisions remain directly comparable. Each original training image contributes one possibly transformed view per epoch. The final short batch has 88 images.

Every method selects its checkpoint with the same validation score: the equal-weight mean of clean, four-direction shift-2, noise-0.15, and four-direction shift-2-plus-noise-0.15 accuracy. The criterion is **not** an unweighted average over all 50,000 views: the four conditions each receive weight 1/4. Ties select the earliest epoch. The unaugmented baseline also receives this robustness-oriented validation selection.

All 20 training runs finish before the official test arrays are loaded for evaluation. Test conditions are clean, noise 0.10/0.20/0.30, all four shifts of two pixels, all four shifts with noise 0.20, and all four shifts of four pixels as an amplitude-extrapolation stress test. There is one noise realization per image per condition/direction. Pure-noise conditions share standardized noise to compare severity. All methods see exactly the same test views.

Mean and sample standard deviation describe optimization variability across five seeds on **one fixed split**. Four shifted views of one original image are correlated observations; they are not four independently sampled test subjects. The study does not make a statistical-significance or state-of-the-art claim.

## Actual experimental results

The following accuracy values are computed from the saved predictions of the 20 completed runs. Entries are **mean ± sample standard deviation, in percent**, across five seeds. The complete unrounded accuracy and macro-F1 values are in `results/metrics.csv` and `results/summary.csv`.

| Method | Clean | Noise 0.10 | Noise 0.20 | Noise 0.30 | Shift 2 px | Shift 2 px + noise 0.20 | Shift 4 px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 97.67 ± 0.18 | 96.96 ± 0.34 | 91.74 ± 2.40 | 77.90 ± 5.10 | 81.37 ± 1.00 | 69.52 ± 3.36 | 31.54 ± 1.25 |
| Shift only | 97.89 ± 0.09 | 96.04 ± 0.37 | 76.37 ± 2.42 | 52.51 ± 1.47 | 96.30 ± 0.19 | 74.52 ± 2.42 | 76.06 ± 1.37 |
| Noise only | 97.66 ± 0.07 | 97.45 ± 0.17 | 96.86 ± 0.22 | 95.49 ± 0.29 | 80.84 ± 0.62 | 79.10 ± 0.93 | 31.61 ± 0.59 |
| Joint | 97.89 ± 0.12 | 97.75 ± 0.17 | 97.11 ± 0.23 | 95.40 ± 0.35 | 96.28 ± 0.17 | 95.11 ± 0.17 | 74.36 ± 1.39 |

Joint augmentation improves compound-corruption accuracy over the baseline by **25.59 percentage points** and over shift-only training by **20.59 points**. The latter paired improvement is positive in all five seeds (17.88–23.94 points; mean ± sample SD, 20.59 ± 2.38). Clean accuracy increases by 0.23 points relative to the baseline in this run. Paired percentages and differences are calculated from integer correct/view counts without intermediate rounding, then displayed to two decimal places using decimal round-half-up.

The joint method does **not** dominate every specialized method: noise-only training is slightly higher under noise 0.30 (95.49% versus 95.40%), while shift-only training is higher under four-pixel translation (76.06% versus 74.36%). The two-pixel translation results of shift-only and joint training are close (96.30% and 96.28%). These observations describe the fixed experiment and do not establish statistical significance or universal superiority. The 20-model training, validation, testing and record-writing run took 925.02 seconds; system load varied, so this is not a method-speed comparison.

All 20 checkpoints reproduce all 140 recorded metric rows and every saved prediction in `verification.json`. A second program, `audit_metrics.py`, does not import the training implementation: it independently checks 90 core file hashes, checkpoint selection, all metrics and summaries, all seven corruption-input hashes, and 8,960 deterministic forward-prediction samples. Its actual outcome is saved in `independent_audit.json`.

Seven figures are supplied in `figures/`. Figure 6 uses the first official test image of each digit class. Figure 7 takes three examples from each agreement/error category with no repeated original images; these balanced examples explain behavior rather than represent category frequencies. Their source indices, labels, predictions and model hashes are in `results/visualization_selection.json`. The 12 selected corrupted examples all happen to use the upward two-pixel shift, as their labels state. Across all 40,000 compound views for seed 11, the counts are 26,719 both correct, 11,398 baseline wrong/joint correct, 453 baseline correct/joint wrong, and 1,430 both wrong.

## Run or verify

The recorded experiment uses Python 3.12 and NumPy 2.3.5. No TensorFlow, PyTorch, scikit-learn, SciPy, or GPU is required.

```sh
python -m pip install -r requirements.txt
python download_data.py
python train.py --benchmark
python train.py
python verify.py
python audit_metrics.py
python -m pip install matplotlib==3.11.2
python plot_results.py
```

`download_data.py` checks the archive against the hash published in the Keras MNIST loader and skips downloading a valid existing copy. `--benchmark` runs only one baseline epoch and validation, without reading official test arrays or saving weights; it is for timing, not model selection. The full training protocol remains 10 epochs.

To verify the released results without retraining, run only `python verify.py`. It checks the file manifest; reconstructs the training/validation split; tests shift boundaries, finite-difference gradients, and corruption generation under different batch sizes; reloads all 20 models; regenerates every test input; and checks hashes, predictions, confusion matrices and all 140 metric rows. Model/data `.npz` files load with `allow_pickle=False`.

For the additional independent implementation checks, run `python audit_metrics.py`. Both verification programs are read-only for core experimental data; they only write their own report. Install Matplotlib 3.11.2 separately, then run `python plot_results.py` to regenerate the seven figures and derived analysis records. The minimal `requirements.txt` contains NumPy for training and numerical verification; plotting versions are recorded in `results/figure_environment.json`.

To regenerate the seven figures and additional analyses from the released predictions, weights and metrics:

```sh
python -m pip install -r requirements-figures.txt
python plot_results.py
```

The plotting script saves aggregate accuracy, noise severity, validation curves and confusion matrices, plus actual MNIST inputs and seed-11 predictions. It reloads both seed-11 checkpoints and checks every displayed prediction. `results/visualization_selection.json` documents exact source/view indices and selection rules. Its four success/failure categories are deliberately shown with three examples each; this is an explanation layout, not an estimate of the categories' prevalence. The actual full-condition category counts are also recorded.

The scripts fix relevant BLAS thread environment variables to two before importing NumPy. Exact replay is expected in the recorded environment; library, CPU or floating-point implementation changes can alter fitting or input generation. Verification deliberately reports such differences instead of silently accepting them. `protocol.json` is a fixed specification of this study, not an interface for arbitrary tasks. Most numeric settings are read by the code, while the MNIST shape/classes and the requirement of 10 epochs and two BLAS threads are explicitly checked.

Rerunning `train.py` overwrites the same-named experiment artifacts. Preserve an archival copy first. No pretrained external model, external training image, test-time ensemble, or hyperparameter search is used.

## Files

- `data/mnist.npz`: original, unmodified compressed archive (11,490,434 bytes).
- `data/split_indices.npz`: the 55,000 training and 5,000 validation indices into the official training arrays.
- `weights/{method}_seed{seed}.npz`: 20 validation-selected checkpoints, including five baseline and five proposed joint-method checkpoints.
- `results/logs/`: 10 per-epoch training/validation records for each of 20 models.
- `results/predictions/`: 20 compact archives containing every predicted test label across seven conditions.
- `results/confusions/`: per-model, per-condition 10×10 confusion matrices.
- `results/metrics.csv` and `.json`: 140 unrounded accuracy/macro-F1 records, including correct counts and evaluated-view counts.
- `results/summary.csv` and `.json`: means and sample standard deviations across seeds.
- `results/test_input_hashes.json`: hashes of deterministically regenerated test images and labels.
- `results/environment.json`: actual environment, timestamps, runtimes, source hashes, split counts, parameter count, selected epochs and validation-input hashes.
- `SHA256SUMS.txt`: manifest covering the immutable core experimental files. Derived figures, analysis files and verification reports are excluded so that regenerating them does not invalidate the manifest.
- `verification.json`: actual outcome of checkpoint/input/metric verification.
- `audit_metrics.py` and `independent_audit.json`: independently implemented consistency checks and the actual audit report.
- `plot_results.py` and `figures/`: seven reproducible figures, including real recognition successes and failures.
- `results/visualization_selection.json`: sample-selection rules, all displayed predictions and source indices.
- `results/visualization_audit.json`: independent checks of the 22 selected records, model predictions and figure hashes.
- `results/extended_analysis.json`: paired seed-wise differences, category confusion and example-category counts.
- `figures/01_examples.png` through `figures/07_mixed_cases.png`: all seven plots, regenerated from this MNIST run only.
- `results/extended_analysis.json`: paired seed-level differences, per-class recalls/confusion rates, checkpoint epochs and actual category counts.
- `results/visualization_selection.json`: every displayed sample, its predictions, weight hashes and deterministic selection rules.

Large float32 corruption arrays are deliberately **not stored**. For a shifted or joint test condition, view index `i` has `direction = i // 10000`, `official_test_index = i % 10000`; the direction order is up, down, left, right. There are 40,000 views. Pure-noise and clean conditions have 10,000 views. Corrupted inputs can be regenerated with `condition_batches` in `train.py`; the data hash and protocol determine them completely.

## Attribution

The MNIST dataset is credited to Yann LeCun, Corinna Cortes and Christopher J. C. Burges. This repository obtains the unmodified archive from the official Keras data mirror. Dataset attribution and the CC BY-SA 3.0 terms are detailed in `DATA_LICENSE.md`.

Primary data documentation: [Keras MNIST API](https://keras.io/api/datasets/mnist/) and [official loader and archive checksum](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py).
