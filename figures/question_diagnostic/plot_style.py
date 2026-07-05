"""Shared matplotlib/seaborn style for all question_diagnostic figures.

Usage: import plot_style  (at top of each plotting script, before any plt calls)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# 10pt Times New Roman throughout
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "mathtext.fontset": "stix",  # Times-compatible math
})
