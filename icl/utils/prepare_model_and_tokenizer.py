import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..util_classes.arg_classes import DeepArgs
from .load_local import load_local_model_or_tokenizer


def load_model_and_tokenizer(
    args: DeepArgs,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer based on the provided arguments.

    Args:
        args (DeepArgs): The arguments specifying the model name.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the loaded model and tokenizer.
    """
    if args.model_name in ["gpt2-xl", "gpt-j-6b"]:
        tokenizer = load_local_model_or_tokenizer(args.model_name, "tokenizer")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = load_local_model_or_tokenizer(args.model_name, "model")
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"model_name: {args.model_name}")
    return model, tokenizer


def get_label_id_dict_for_args(
    args: DeepArgs, tokenizer: AutoTokenizer
) -> dict[str, int]:
    """
    Returns a dictionary mapping label names to their corresponding label IDs.

    Args:
        args (DeepArgs): The arguments object containing the label dictionary.
        tokenizer (AutoTokenizer): The tokenizer used to encode the label names.

    Returns:
        dict[str, int]: A dictionary mapping label names to their corresponding label IDs.
    """
    label_id_dict: dict[str, int] = {
        k: tokenizer.encode(v, add_special_tokens=False)[0]
        for k, v in args.label_dict.items()
    }
    for v in args.label_dict.values():
        token_num = len(tokenizer.encode(v, add_special_tokens=False))
        if token_num != 1:
            warnings.warn(
                f"{v} in {args.task_name} has token_num: {token_num} which is not 1"
            )
    return label_id_dict
