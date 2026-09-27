from src.data.data_loader import load_dataset


def test_loader_function_exists():
    assert callable(load_dataset)
