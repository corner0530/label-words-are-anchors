import datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

format_s_dict: dict[str, str] = {
    "sst2": "Review: {text}\nSentiment:{label}",
    "agnews": "Article: {text}\nAnswer:{label}",
    "trec": "Question: {question}\nAnswer Type:{label}",
    "emo": "Dialogue: {text}\nEmotion:{label}",
}


def sst2_wrap_data(
    demonstrations: list[dict[str, any]],
    input_sample: dict[str, any],
    label_dict: dict[str, str],
) -> str:
    """
    Wraps the data for the SST-2 task by formatting the input samples and labels.

    Args:
        demonstrations (list[dict[str, any]]): A list of dictionaries representing the demonstration samples.
        input_sample (dict[str, any]): A dictionary representing the input sample.
        label_dict (dict[str, str]): A dictionary mapping labels to their corresponding formatted strings.

    Returns:
        str: The wrapped data as a string.

    """
    format_s: str = format_s_dict["sst2"]
    prompts: list[str] = [
        format_s.format(text=sample["text"], label=label_dict[sample["label"]])
        for sample in demonstrations
    ]
    inputs: str = format_s.format(text=input_sample["text"], label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def trec_wrap_data(
    demonstrations: list[dict[str, any]],
    input_sample: dict[str, any],
    label_dict: dict[str, str],
) -> str:
    """
    Wraps the data in TREC format.

    Args:
        demonstrations (list[dict[str, any]]): A list of dictionaries representing the demonstrations.
        input_sample (dict[str, any]): A dictionary representing the input sample.
        label_dict (dict[str, str]): A dictionary mapping label names to label values.

    Returns:
        str: The wrapped data in TREC format.
    """
    format_s: str = format_s_dict["trec"]
    prompts: list[str] = [
        format_s.format(
            question=sample["text"], label=label_dict[sample["label"]]
        )
        for sample in demonstrations
    ]
    inputs: str = format_s.format(question=input_sample["text"], label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def emo_wrap_data(
    demonstrations: list[dict[str, any]],
    input_sample: dict[str, any],
    label_dict: dict[str, str],
) -> str:
    """
    Wraps the data for emotion classification by formatting the input samples and labels.

    Args:
        demonstrations (list[dict[str, any]]): A list of dictionaries representing the demonstration samples.
        input_sample (dict[str, any]): A dictionary representing the input sample.
        label_dict (dict[str, str]): A dictionary mapping label keys to label values.

    Returns:
        str: The wrapped data with formatted prompts and inputs.
    """
    format_s: str = format_s_dict["emo"]
    prompts: list[str] = [
        format_s.format(text=sample["text"], label=label_dict[sample["label"]])
        for sample in demonstrations
    ]
    inputs: str = format_s.format(text=input_sample["text"], label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def agnews_wrap_data(
    demonstrations: list[dict[str, any]],
    input_sample: dict[str, any],
    label_dict: dict[str, str],
) -> str:
    """
    Wraps the data for the AG News dataset by formatting the input samples and labels.

    Args:
        demonstrations (list[dict[str, any]]): A list of dictionaries representing the demonstration samples.
        input_sample (dict[str, any]): A dictionary representing the input sample.
        label_dict (dict[str, str]): A dictionary mapping label names to label values.

    Returns:
        str: The wrapped data as a string.

    """
    format_s: str = format_s_dict["agnews"]
    prompts: list[str] = [
        format_s.format(text=sample["text"], label=label_dict[sample["label"]])
        for sample in demonstrations
    ]
    inputs: str = format_s.format(text=input_sample["text"], label="")
    if len(prompts) > 0:
        inputs = "\n".join(prompts + [inputs])
    return inputs


def wrap_data(
    demonstrations: list[dict[str, any]],
    input_sample: dict[str, any],
    label_dict: dict[str, str],
    task_name: str,
) -> str:
    """
    Wraps the data based on the task name.

    Args:
        demonstrations (list[dict[str, any]]): List of demonstration data.
        input_sample (dict[str, any]): Input sample data.
        label_dict (dict[str, str]): Dictionary mapping labels to their corresponding values.
        task_name (str): Name of the task.

    Returns:
        str: Wrapped data.

    Raises:
        NotImplementedError: If the task name is not supported.
    """
    if task_name == "sst2":
        return sst2_wrap_data(demonstrations, input_sample, label_dict)
    elif task_name == "agnews":
        return agnews_wrap_data(demonstrations, input_sample, label_dict)
    elif task_name == "trec":
        return trec_wrap_data(demonstrations, input_sample, label_dict)
    elif task_name == "emo":
        return emo_wrap_data(demonstrations, input_sample, label_dict)
    else:
        raise NotImplementedError(f"task_name: {task_name}")


def instruct_wrapper(
    instruct: str,
    input_sample: dict[str, any],
    label_dict: dict[str, str],
    task_name: str,
) -> str:
    """
    Wraps the input data and formats it with the given instruction.

    Args:
        instruct (str): The instruction to be included in the formatted output.
        input_sample (dict[str, any]): The input data sample.
        label_dict (dict[str, str]): The label dictionary.
        task_name (str): The name of the task.

    Returns:
        str: The formatted input data with the instruction.

    """
    inputs: str = wrap_data(
        demonstrations=[],
        input_sample=input_sample,
        label_dict=label_dict,
        task_name=task_name,
    )
    format_s: str = "{instruct}\n{text}"
    inputs: str = format_s.format(text=inputs, instruct=instruct)
    return inputs


def wrap_dataset(
    dataset: datasets.arrow_dataset.Dataset,
    demonstration: list[dict[str, any]],
    label_dict: dict[str, str],
    task_name: str,
) -> datasets.arrow_dataset.Dataset:
    """
    Wraps the dataset by modifying the examples.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to be wrapped.
        demonstration (list[dict[str, any]]): The list of demonstration data.
        label_dict (dict[str, str]): The dictionary mapping label names to label values.
        task_name (str): The name of the task.

    Returns:
        datasets.arrow_dataset.Dataset: The wrapped dataset.
    """

    def wrap(example: dict[str, any]) -> dict[str, any]:
        example["sentence"] = wrap_data(
            demonstrations=demonstration,
            input_sample=example,
            label_dict=label_dict,
            task_name=task_name,
        )
        example["labels"] = example["label"]
        return example

    dataset = dataset.map(wrap)
    return dataset


def wrap_dataset_with_instruct(
    dataset: datasets.arrow_dataset.Dataset,
    instruct: str,
    label_dict: dict[str, str],
    task_name: str,
) -> datasets.arrow_dataset.Dataset:
    """
    Wraps the given dataset with instructions by modifying the 'sentence' and 'labels' fields of each example.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to be wrapped.
        instruct (str): The instruction to be applied to each example.
        label_dict (dict[str, str]): A dictionary mapping label keys to label values.
        task_name (str): The name of the task.

    Returns:
        datasets.arrow_dataset.Dataset: The wrapped dataset.
    """

    def wrap(example: dict[str, any]) -> dict[str, any]:
        example["sentence"] = instruct_wrapper(
            instruct=instruct,
            input_sample=example,
            label_dict=label_dict,
            task_name=task_name,
        )
        example["labels"] = example["label"]
        return example

    dataset = dataset.map(wrap)
    return dataset


# you may add your tokenizer's name or local path (corresponding to tokenizer.name_or_path)
# to this dict, and the corresponding model max length
default_max_length_dict = {
    "gpt2": 1024,
}


def get_max_length(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> int:
    """
    Get the maximum length for tokenization.

    Args:
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer object.

    Returns:
        int: The maximum length for tokenization.
    """
    if tokenizer.name_or_path in default_max_length_dict:
        return default_max_length_dict[tokenizer.name_or_path]
    max_length = tokenizer.max_len_single_sentence
    if max_length > 10000000:
        max_length = tokenizer.model_max_length
    if max_length > 10000000:
        raise ValueError(
            f"Your tokenizer has a very large `max_len_single_sentence` value: {max_length}, "
            f"you may add this to tokenizer's config, or add it to `default_max_length_dict` above"
        )
    return max_length


def tokenize_dataset(
    dataset: datasets.arrow_dataset.Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> datasets.arrow_dataset.Dataset:
    """
    Tokenizes the dataset using the provided tokenizer.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to be tokenized.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to be used for tokenization.

    Returns:
        datasets.arrow_dataset.Dataset: The tokenized dataset.
    """

    def tokenize_function(examples: dict[str, any]) -> dict[str, any]:
        return tokenizer(
            examples["sentence"],
            padding=True,
            max_length=get_max_length(tokenizer),
            truncation=True,
            return_tensors="pt",
        )

    tokenized_datasets: datasets.arrow_dataset.Dataset = dataset.map(
        tokenize_function, batched=True
    )
    return tokenized_datasets


def remove_str_columns(
    dataset: datasets.arrow_dataset.Dataset,
) -> datasets.arrow_dataset.Dataset:
    """
    Removes string columns from the given dataset.

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to remove string columns from.

    Returns:
        datasets.arrow_dataset.Dataset: The dataset with string columns removed.
    """
    remove_keys: set[str] = {
        k for k, v in dataset.features.items() if v.dtype == "string"
    }
    dataset = dataset.remove_columns(list(remove_keys))
    return dataset
