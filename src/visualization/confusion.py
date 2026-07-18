"""
Confusion Matrix Visualization
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import PLOT_DIR


def plot_confusion_matrix(
    confusion_matrix,
    model_name,
    cmap="Blues",
    save=True,
):
    """
    Plot and optionally save a confusion matrix.

    Parameters
    ----------
    confusion_matrix : ndarray
        Confusion matrix computed using sklearn.metrics.confusion_matrix.

    model_name : str
        Model name for plot title and filename.

    cmap : str, default="Blues"
        Matplotlib colormap.

    save : bool, default=True
        Whether to save the figure.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=["Normal", "Attack"],
    )

    display.plot(
        ax=ax,
        cmap=cmap,
        colorbar=False,
        values_format="d",
    )

    ax.set_title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    if save:

        PLOT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            PLOT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_all_confusion_matrices(results):
    """
    Plot confusion matrices for multiple models.

    Parameters
    ----------
    results : dict

    Example
    -------
    {
        "Random Forest": rf_results,
        "Logistic Regression": lr_results,
        "Hybrid IDS": hybrid_results,
    }
    """

    for model_name, metrics in results.items():

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            model_name,
        )