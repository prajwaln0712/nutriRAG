"""Food class labels and label-index mappings."""

LABELS = []


def label_to_index(label: str) -> int:
    """Return the integer index for a label."""
    return LABELS.index(label)


def index_to_label(index: int) -> str:
    """Return the label for an integer index."""
    return LABELS[index]
