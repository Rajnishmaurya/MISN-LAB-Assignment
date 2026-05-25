import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server
import matplotlib.pyplot as plt


def plot_metric(train_values, val_values, ylabel, save_path, title=None):
    epochs = list(range(1, len(train_values) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train_values, marker='o', linewidth=2,
            markersize=4, label='Train', color='#2196F3')
    ax.plot(epochs, val_values,   marker='s', linewidth=2,
            markersize=4, label='Validation', color='#F44336')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title if title else ylabel, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_lr(lr_values, save_path):
    epochs = list(range(1, len(lr_values) + 1))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, lr_values, marker='o', linewidth=2,
            markersize=4, color='#4CAF50')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Learning Rate', fontsize=13)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
