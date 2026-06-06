"""Training entry point for the food classifier."""


def train(data_dir: str, epochs: int = 10, lr: float = 1e-3):
    """Train the food classifier on labeled image data."""
    raise NotImplementedError


if __name__ == "__main__":
    train("data/processed")
