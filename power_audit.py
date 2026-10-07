"""Verdict reliability and sample-size audit for CIM accuracy budgets.

Usage:
    python power_audit.py data/results/fulltest10k/c100/logits_s13_n10000_per_image.csv --tau 1.0
    python power_audit.py fp_correct.npy clip_correct.npy --tau 1.0
Input is either one per-image CSV with columns correct_fp and correct_clip, or two
.npy 0/1 vectors of per-image correctness over the SAME test images.
Reproduces Tables IV-V and Figs. 3-5 of the manuscript for one checkpoint.
"""
import argparse, numpy as np
from scipy.stats import binomtest

def paired_ci(n01, n10, n, B=10000, rng=None):
    rng = rng or np.random.default_rng(1234)
    c = rng.multinomial(n, np.array([n01, n10, n - n01 - n10]) / n, size=B)
    d = 100 * (c[:, 1] - c[:, 0]) / n
    return np.percentile(d, [2.5, 97.5])

def verdict(lo, hi, tau):
    return np.where(lo >= -tau, 'hold', np.where(hi < -tau, 'fail', 'unresolved'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fp', help='per-image CSV, or FP correctness .npy')
    ap.add_argument('clip', nargs='?', help='clip correctness .npy (omit when a CSV is given)')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--draws', type=int, default=20000)
    a = ap.parse_args()
    if a.fp.endswith('.csv'):
        import csv
        with open(a.fp) as fh:
            rows = list(csv.DictReader(fh))
        fp = np.array([int(r['correct_fp']) for r in rows]); cl = np.array([int(r['correct_clip']) for r in rows])
    else:
        fp, cl = np.load(a.fp).astype(int), np.load(a.clip).astype(int)
    assert fp.shape == cl.shape
    N = len(fp); n01 = int(((fp == 1) & (cl == 0)).sum()); n10 = int(((fp == 0) & (cl == 1)).sum())
    D = 100 * (n10 - n01) / N; d = (n01 + n10) / N
    lo, hi = paired_ci(n01, n10, N)
    p = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
    print(f"N={N} delta={D:+.2f} pp CI=[{lo:.2f},{hi:.2f}] hurt/helped={n01}/{n10} "
          f"McNemar p={p:.3g} verdict={verdict(np.array(lo), np.array(hi), a.tau)}")
    gap = abs(D / 100 + a.tau / 100)
    nreq = (1.959964 + 0.841621) ** 2 * (d - (D / 100) ** 2) / gap ** 2 if gap > 0 else np.inf
    print(f"discordance d={d:.4f}  n_req(80% power)={nreq:.0f}")
    full = verdict(np.array(lo), np.array(hi), a.tau)
    rng = np.random.default_rng(1234)
    for n in [256, 400, 512, 1000, 2000, 3000, 5000, 7500]:
        if n >= N: break
        s = rng.multivariate_hypergeometric([n01, n10, N - n01 - n10], n, size=a.draws)
        dd = (s[:, 1] - s[:, 0]) / n; se = np.sqrt(np.maximum((s[:, 0] + s[:, 1]) / n - dd ** 2, 0) / n)
        v = verdict(100 * (dd - 1.96 * se), 100 * (dd + 1.96 * se), a.tau)
        pe = np.mean((100 * dd >= -a.tau) != (D >= -a.tau))
        print(f"n={n:5d} P(hold)={np.mean(v=='hold'):.3f} P(unres)={np.mean(v=='unresolved'):.3f} "
              f"P(fail)={np.mean(v=='fail'):.3f} P(agree full)={np.mean(v==full):.3f} P(point-estimate flip)={pe:.3f}")

if __name__ == '__main__':
    main()
