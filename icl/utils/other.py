import functools
import os
import warnings

import numpy as np
import torch
import transformers
from transformers import HfArgumentParser

from .random_utils import np_temp_random

REDUCE_FN_MAPPINGS = {
    "sum": torch.sum,
    "mean": torch.mean,
    "none": lambda x: x,
}


def apply_on_element(l, fn=None) -> any:
    """
    Recursively applies a function to each element in a tensor, list, or dictionary.

    Args:
        l (torch.Tensor | list | dict | any): The input tensor, list, or dictionary.
        fn (callable[[any], any] | None): The function to be applied to each element. Defaults to None.

    Returns:
        any: The result after applying the function to each element.

    """
    if isinstance(l, torch.Tensor):
        l = l.tolist()
    if isinstance(l, list):
        return [apply_on_element(_, fn) for _ in l]
    elif isinstance(l, dict):
        return {k: apply_on_element(v, fn) for k, v in l.items()}
    else:
        return fn(l)


def show_words(
    logits: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    topk: int = 5,
) -> None:
    """
    Prints the top-k words based on the given logits.

    Args:
        logits (torch.Tensor): The logits tensor.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to convert token IDs to words.
        topk (int, optional): The number of top words to display. Defaults to 5.

    Returns:
        None
    """
    token_ids = logits.topk(topk)[1]
    words = apply_on_element(token_ids, tokenizer.convert_ids_to_tokens)
    print(words)


def load_args(args_type: type, is_ipynb: bool = False) -> any:
    """
    Load arguments of the specified type.

    Args:
        args_type (type): The type of the arguments to load.
        is_ipynb (bool, optional): Indicates whether the code is running in an IPython notebook. Defaults to False.

    Returns:
        any: The loaded arguments.
    """
    if not is_ipynb:
        parser = HfArgumentParser((args_type,))
        (args,) = parser.parse_args_into_dataclasses()
    else:
        args = args_type()
    return args


def sample_two_set_with_shot_per_class(
    ori_data: list[dict],
    a_shot: int,
    b_shot: int,
    seed: int,
    label_name: str = "labels",
    a_total_shot=None,
    b_total_shot=None,
) -> tuple[list[dict], list[dict]]:
    """
    Samples two sets of data with a specified number of shots per class.

    Args:
        ori_data (list[dict]): The original data to sample from.
        a_shot (int): The number of shots to sample from class A.
        b_shot (int): The number of shots to sample from class B.
        seed (int): The seed value for random number generation.
        label_name (str, optional): The key name for the label in the data dictionary. Defaults to "labels".
        a_total_shot (int | None, optional): The total number of shots to sample from class A. Defaults to None.
        b_total_shot (int | None, optional): The total number of shots to sample from class B. Defaults to None.

    Returns:
        tuple[list[dict], list[dict]]: A tuple containing two lists of dictionaries, representing the sampled data for class A and class B, respectively.
    """
    a_label_count = {}
    b_label_count = {}
    a_data_idx = []
    b_data_idx = []
    all_indices = [_ for _ in range(len(ori_data))]
    np_temp_random(seed=seed)(np.random.shuffle)(all_indices)

    a_total_cnt = 0
    b_total_cnt = 0
    for index in all_indices:
        label = ori_data[index][label_name]
        if label < 0:
            continue

        if label not in a_label_count.keys():
            a_label_count[label] = 0
        if label not in b_label_count.keys():
            b_label_count[label] = 0

        if a_label_count[label] < a_shot:
            a_data_idx.append(index)
            a_label_count[label] += 1
            a_total_cnt += 1
        elif b_label_count[label] < b_shot:
            b_data_idx.append(index)
            b_label_count[label] += 1
            b_total_cnt += 1

        a_cond = a_total_shot is not None and a_total_cnt >= a_total_shot
        b_cond = (
            b_total_shot is not None and b_total_cnt >= b_total_shot
        ) or (b_shot == 0)
        if a_cond and b_cond:
            warnings.warn(
                f"sampled {a_total_shot} and {b_total_shot} samples, "
            )

    a_data = ori_data.select(a_data_idx)
    b_data = ori_data.select(b_data_idx)
    return a_data, b_data


def dict_to(
    d: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Move all torch.Tensor values in the input dictionary to the specified device.

    Args:
        d (dict[str, torch.Tensor]): The input dictionary.
        device (torch.device): The target device.

    Returns:
        dict[str, torch.Tensor]: The modified dictionary with tensor values moved to the target device.
    """
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d


def set_gpu(gpu_id: str | int) -> None:
    """
    Sets the CUDA_VISIBLE_DEVICES environment variable to the specified GPU ID.

    Args:
        gpu_id (str | int): The ID of the GPU to be used. Can be either a string or an integer.

    Returns:
        None
    """
    if isinstance(gpu_id, int):
        gpu_id = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


class TensorStrFinder:
    """
    Utility class for finding strings or tensors within a given tensor.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode strings into tokens.

    Methods:
        find_tensor_in_tensor: Finds a tensor within another tensor and returns the positions where the match occurs.
        find_str_in_tensor: Finds a string within a tensor and returns the positions where the match occurs.
        get_strs_mask_in_tensor: Finds multiple strings within a tensor and returns a mask indicating the positions where any of the strings match.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def find_tensor_in_tensor(
        self,
        a_tensor: torch.Tensor | list,
        b_tensor: torch.Tensor,
        return_mask: bool = True,
        match_before=None,
    ) -> torch.Tensor | torch.BoolTensor:
        """
        Finds a tensor within another tensor and returns the positions where the match occurs.

        Args:
            a_tensor (torch.Tensor | list): The tensor or list to search for.
            b_tensor (torch.Tensor): The tensor to search within.
            return_mask (bool, optional): Whether to return a mask indicating the positions where the match occurs. Defaults to True.
            match_before (int | None, optional): The maximum position where a match is allowed. Defaults to None.

        Returns:
            torch.Tensor | torch.BoolTensor: The positions where the match occurs, or a mask indicating the positions.
        """
        if len(b_tensor.shape) == 2:
            assert b_tensor.shape[0] == 1
            b_tensor = b_tensor[0]
        if isinstance(a_tensor, list):
            a_tensor = torch.tensor(a_tensor)
        if a_tensor.device != b_tensor.device:
            a_tensor = a_tensor.to(b_tensor.device)

        window_size = len(a_tensor)
        b_windows = b_tensor.unfold(0, window_size, 1)

        matches = torch.all(b_windows == a_tensor, dim=1)

        positions = torch.nonzero(matches, as_tuple=True)[0]

        if return_mask:
            mask = torch.zeros_like(b_tensor, dtype=torch.bool)
            for pos in positions:
                if match_before is None or pos + window_size <= match_before:
                    mask[pos : pos + window_size] = True
            return mask

        return positions

    def find_str_in_tensor(
        self,
        s: str,
        t: torch.Tensor,
        return_mask: bool = True,
        match_before=None,
    ) -> torch.Tensor | torch.BoolTensor:
        """
        Finds a string within a tensor and returns the positions where the match occurs.

        Args:
            s (str): The string to search for.
            t (torch.Tensor): The tensor to search within.
            return_mask (bool, optional): Whether to return a mask indicating the positions where the match occurs. Defaults to True.
            match_before (int | None, optional): The maximum position where a match is allowed. Defaults to None.

        Returns:
            torch.Tensor | torch.BoolTensor: The positions where the match occurs, or a mask indicating the positions.
        """
        s_tokens = self.tokenizer.encode(s, add_special_tokens=False)
        s_tensor = torch.LongTensor(s_tokens)
        return self.find_tensor_in_tensor(
            s_tensor, t, return_mask=return_mask, match_before=match_before
        )

    def get_strs_mask_in_tensor(
        self,
        list_s: list[str],
        t: torch.Tensor,
        match_before=None,
    ) -> torch.BoolTensor:
        """
        Finds multiple strings within a tensor and returns a mask indicating the positions where any of the strings match.

        Args:
            list_s (list[str]): The list of strings to search for.
            t (torch.Tensor): The tensor to search within.
            match_before (int | None, optional): The maximum position where a match is allowed. Defaults to None.

        Returns:
            torch.BoolTensor: A mask indicating the positions where any of the strings match.
        """
        list_s_tokens = [
            self.tokenizer.encode(s, add_special_tokens=False) for s in list_s
        ]
        list_s_tensor = [
            torch.LongTensor(s_tokens) for s_tokens in list_s_tokens
        ]
        mask_tensor_list = [
            self.find_tensor_in_tensor(
                s_tensor, t, return_mask=True, match_before=match_before
            )
            for s_tensor in list_s_tensor
        ]
        mask_tensor = functools.reduce(torch.logical_or, mask_tensor_list)
        return mask_tensor
