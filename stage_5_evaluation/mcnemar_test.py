"""
McNemar's exact paired test (Chapter 3 §3.7.1, Table 3.6).

Given two paired Boolean vectors c1, c2 (correctness on the same
probes):

    b = # probes where c2 correct but c1 wrong
    c = # probes where c1 correct but c2 wrong

Under the null (the two matchers have equal accuracy on this probe
set), min(b, c) ~ Binomial(b + c, 0.5).  We report the exact two-
sided binomial p-value, which is preferred over the chi-square
approximation when b + c is small.
"""

from dataclasses import dataclass
from scipy import stats


@dataclass
class McNemarResult:
    b: int                 # only matcher B correct
    c: int                 # only matcher A correct
    p_value: float
    significance: str      # "***", "**", "*", or "ns"


def _sig_label(p):
    return ("***" if p < 0.001 else
            "**"  if p < 0.01  else
            "*"   if p < 0.05  else "ns")


def mcnemar_test(correct_A, correct_B) -> McNemarResult:
    b = int(((~correct_A) &  correct_B).sum())
    c = int(( correct_A  & ~correct_B).sum())
    if b + c == 0:
        return McNemarResult(b=b, c=c, p_value=1.0, significance="ns")
    p = float(stats.binomtest(min(b, c), b + c, 0.5,
                              alternative="two-sided").pvalue)
    return McNemarResult(b=b, c=c, p_value=p, significance=_sig_label(p))
