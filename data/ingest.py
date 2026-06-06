"""Data ingestion: load raw documents and prepare them for indexing."""


def ingest(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """Read raw data, process it, and write to the processed directory."""
    raise NotImplementedError


if __name__ == "__main__":
    ingest()
