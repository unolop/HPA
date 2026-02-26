import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
 

def draw_ecdf_embedding_similarity(human, model, save=None):
    plt.figure(figsize=(6, 4))

    # ECDF curves
    sns.ecdfplot(human, label="Human")
    sns.ecdfplot(model, label="Model")

    # Mean lines
    plt.axvline(
        human.mean(), ls="--", lw=1,
        label=f"Human μ={human.mean():.1f}"
    )
    plt.axvline(
        model.mean(), ls="--", lw=1, color="orange",
        label=f"Model μ={model.mean():.1f}"
    )

    plt.xlabel("Embedding similarity (cosine, ×100)")
    plt.ylabel("ECDF")
    plt.title("Answer embedding similarity (ECDF)")
    plt.legend(frameon=False)

    sns.despine()
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    else:
        plt.show()
        
def draw_histogram_embedding_similarity(human, model, save=None):
    plt.figure(figsize=(6, 4))

    # Shared bins for fair comparison
    bins = np.linspace(
        min(human.min(), model.min()),
        max(human.max(), model.max()),
        40
    )

    plt.hist(
        human, bins=bins, alpha=0.6, density=True,
        label="Human"
    )
    plt.hist(
        model, bins=bins, alpha=0.6, density=True,
        label="Model"
    )

    # Mean lines
    plt.axvline(
        human.mean(), ls="--", lw=1,
        label=f"Human μ={human.mean():.1f}"
    )
    plt.axvline(
        model.mean(), ls="--", lw=1, color="orange",
        label=f"Model μ={model.mean():.1f}"
    )

    plt.xlabel("Embedding similarity (cosine, ×100)")
    plt.ylabel("Density")
    plt.title("Answer embedding similarity distribution")
    plt.legend(frameon=False)

    sns.despine()
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    else:
        plt.show()