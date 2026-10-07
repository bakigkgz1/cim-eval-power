# cim-eval-power

Data and analysis code for

> B. Gökgöz, **Evaluation-Set Size Decides Accuracy-Budget Verdicts in Compute-in-Memory Inference: A Paired Non-Inferiority Analysis** (submitted).

The repository contains the raw logits behind every accuracy comparison in the paper, per-image correctness tables derived from them, the locked experimental protocol, and the scripts that reproduce the paired verdicts, the sample-size rule and the subsampling analysis.

## Simulation setting

All results were produced with DNN+NeuroSim V1.5 at one fixed operating point:

| Parameter | Value |
|---|---|
| Kernel path | `hardware = 0` (digital) |
| Subarray / parallel read rows | 128 × 128 / 64 |
| Cell / DAC | binary resistive / 1 bit |
| Weight / input precision | 8 bit / 8 bit, bit-sliced |
| Read noise, drift, stuck-at, variation | off |
| Swept field | ADC precision b (acts as a partial-sum clip at 2^b; b ≥ 6 is identical to no clipping) |
| Compared arms | floating point (FP) vs. 2^5 clip |
| Calibration set | 256 images |
| Seed | 1234 |

## Repository layout

```
data/
  protocol/                      locked parameters (locks.json), gate log (gates.csv), session record
  results/
    fp32_ceilings/               FP accuracy of each checkpoint on the full test set
    clip_sweep_512/              clip-ceiling sweeps (b = off, 4 ... 12) on the 512-image window
    analog_kernel_check/         hardware = 1 (noise off) vs. hardware = 0 check
    window512/                   FP and 2^5-clip logits on the first 512 test images
    fulltest10k/c10/             FP and 2^5-clip logits, full CIFAR-10 test set (4 checkpoints)
    fulltest10k/c100/            FP and 2^5-clip logits, full CIFAR-100 test set (6 checkpoints)
    screens/                     pre-clip overflow rates and stage-wise residual ratios (Table 7)
    training/                    training summaries of the five matched-seed CIFAR-100 weights
    supplementary/               full-test CIFAR-100 logits under a 50 000-image calibration set
power_audit.py                   paired verdict, sample-size rule, subsampling reliability
verification.json                every Table 4 entry recomputed from the raw logits
MANIFEST.csv                     every file with size and SHA-256 checksum
```

### Logit files

Each `logits*.pt` file is a PyTorch dictionary with three tensors over the same images:

| Key | Shape | Meaning |
|---|---|---|
| `y` | (n,) | ground-truth label |
| `z_fp` | (n, C) | logits of the floating-point arm |
| `z_5` | (n, C) | logits of the 2^5 partial-sum clip arm |

Next to every `.pt` file there is a `*_per_image.csv` with the columns
`image_id, label, pred_fp, pred_clip, correct_fp, correct_clip`, so the data can be used without PyTorch.

### Checkpoint naming

| File name | Paper name |
|---|---|
| `vgg8_p3` / `window512/vgg8_a` | VGG8/C10 (a) |
| `vgg8_2nd` | VGG8/C10 (b) |
| `resnet_c10_p3` / `window512/resnet18_c10_a` | ResNet-18/C10 (a) |
| `resnet_c10_5678` | ResNet-18/C10 s5678 |
| `s13`, `s21`, `s34`, `s55`, `s89` | ResNet-18/C100 matched-recipe seeds |
| `P4` | ResNet-18/C100 s5678 |

## Reproducing the paper's numbers

```bash
pip install numpy scipy
python power_audit.py data/results/fulltest10k/c100/logits_s13_n10000_per_image.csv --tau 1.0
```

The script prints the paired accuracy change, its 95% bootstrap interval (B = 10 000, seed 1234), the hurt/helped counts, the exact McNemar p-value, the verdict against the allowance τ, the evaluation size required by the closed-form rule, and the share of random n-image subsets that reproduce the full-test verdict.

### Verification summary (full 10 000-image test set, 2^5 clip, τ = 1 pp)

| Checkpoint | FP (%) | Clip (%) | Δ (pp) | 95% CI | hurt/helped | Verdict |
|---|---|---|---|---|---|---|
| VGG8/C10 (a) | 90.19 | 90.06 | −0.13 | [−0.31, 0.04] | 47/34 | hold |
| VGG8/C10 (b) | 90.31 | 90.19 | −0.12 | [−0.31, 0.07] | 54/42 | hold |
| ResNet-18/C10 (a) | 90.04 | 89.94 | −0.10 | [−0.30, 0.10] | 59/49 | hold |
| ResNet-18/C10 s5678 | 95.05 | 94.99 | −0.06 | [−0.22, 0.10] | 35/29 | hold |
| ResNet-18/C100 s34 | 77.68 | 77.37 | −0.31 | [−0.62, 0.01] | 146/115 | hold |
| ResNet-18/C100 s55 | 77.31 | 76.25 | −1.06 | [−1.45, −0.67] | 249/143 | unresolved |
| ResNet-18/C100 s13 | 77.58 | 75.89 | −1.69 | [−2.09, −1.30] | 292/123 | fail |
| ResNet-18/C100 s21 | 77.66 | 75.73 | −1.93 | [−2.36, −1.52] | 326/133 | fail |
| ResNet-18/C100 s89 | 78.03 | 75.66 | −2.37 | [−2.79, −1.96] | 344/107 | fail |
| ResNet-18/C100 s5678 | 78.02 | 74.32 | −3.70 | [−4.21, −3.21] | 516/146 | fail |

## Data sources

CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) are publicly available and are not redistributed here. DNN+NeuroSim V1.5 is available from its authors.

## Citation

If you use these data, please cite the paper above.

## License

Data: CC BY 4.0. Code: MIT.
