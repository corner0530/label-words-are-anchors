import os.path

from datasets import Dataset, load_dataset, load_from_disk

ROOT_FOLEDER = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def load_from_local(task_name: str, splits: list[str]) -> list[Dataset]:
    """
    Load a Hugging Face dataset from the local file system.

    Args:
        task_name (str): The name of the task or dataset.
        splits (list[str]): A list of split names to load.

    Returns:
        list[Dataset]: A list of loaded datasets corresponding to the specified splits.
    """
    dataset_path = os.path.join(ROOT_FOLEDER, "datasets", task_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset_path: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    dataset = [dataset[split] for split in splits]
    return dataset


def load_huggingface_dataset_train_and_test(
    task_name: str,
) -> dict[str, Dataset]:
    """
    Loads the Hugging Face dataset for training and testing.

    Args:
        task_name (str): The name of the task/dataset to load.

    Returns:
        dict[str, Dataset]: A dictionary containing the train and test datasets.

    Raises:
        FileNotFoundError: If the dataset files are not found.
        NotImplementedError: If the specified task_name is not implemented.

    """
    dataset: dict[str, Dataset] = {}
    if task_name == "sst2":
        try:
            dataset = load_from_local(task_name, ["train", "validation"])
        except FileNotFoundError:
            dataset = load_dataset(
                "glue", "sst2", split=["train", "validation"]
            )
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column("sentence", "text")
        # rename validation to test
    elif task_name == "agnews":
        try:
            dataset = load_from_local(task_name, ["train", "test"])
        except FileNotFoundError:
            dataset = load_dataset("ag_news", split=["train", "test"])
    elif task_name == "trec":
        try:
            dataset = load_from_local(task_name, ["train", "test"])
        except FileNotFoundError:
            dataset = load_dataset("trec", split=["train", "test"])
        coarse_label_name = (
            "coarse_label"
            if "coarse_label" in dataset[0].column_names
            else "label-coarse"
        )
        for i, _ in enumerate(dataset):
            dataset[i] = dataset[i].rename_column(coarse_label_name, "label")
    elif task_name == "emo":
        try:
            dataset = load_from_local(task_name, ["train", "test"])
        except FileNotFoundError:
            dataset = load_dataset("emo", split=["train", "test"])
    if dataset is None:
        raise NotImplementedError(f"task_name: {task_name}")
    dataset = {"train": dataset[0], "test": dataset[1]}
    return dataset
