"""Plotting utilities for Part 2 training curves."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_metric(train_vals, val_vals, ylabel, save_path, title=None):
    epochs = list(range(1, len(train_vals) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_vals, marker='o', label=f'Train {ylabel}', linewidth=2)
    ax.plot(epochs, val_vals,   marker='s', label=f'Val {ylabel}',   linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Train vs Validation {ylabel}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_single(vals, ylabel, save_path, title=None):
    steps = list(range(1, len(vals) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, vals, linewidth=2, color='steelblue')
    ax.set_xlabel('Step / Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_lr(lr_history, save_path):
    plot_single(lr_history, 'Learning Rate', save_path, title='Learning Rate Schedule')
