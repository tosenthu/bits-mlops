"""Download and clean the Heart Disease UCI dataset."""

from __future__ import annotations

from heart_disease_mlops.data import prepare_dataset


def main() -> None:
    processed_path = prepare_dataset(download=True)
    print(f"Processed dataset written to {processed_path}")


if __name__ == "__main__":
    main()
