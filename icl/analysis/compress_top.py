import pickle
import random
import warnings
from dataclasses import dataclass, field
from typing import List
import os

import numpy as np
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from .prefixier import Prefixer
from ..lm_apis.lm_api_base import LMForwardAPI
from ..util_classes.context_solver import ContextSolver
from ..util_classes.predictor_classes import Predictor
from ..utils.data_wrapper import (
    wrap_dataset,
    tokenize_dataset,
    wrap_dataset_with_instruct,
    remove_str_columns,
)
from ..utils.load_huggingface_dataset import (
    load_huggingface_dataset_train_and_test,
)
from ..utils.random_utils import set_seed
from ..utils.other import (
    load_args,
    set_gpu,
    sample_two_set_with_shot_per_class,
)
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from ..utils.load_local import get_model_layer_num
from ..util_classes.arg_classes import CompressTopArgs
from ..utils.prepare_model_and_tokenizer import (
    load_model_and_tokenizer,
    get_label_id_dict_for_args,
)


class TruncatingDataCollator(DataCollatorWithPadding):
    def __init__(
        self, tokenizer, max_length: int, padding=True, pad_to_multiple_of=None
    ):
        super().__init__(
            tokenizer=tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.max_length = max_length

    def __call__(self, features: List[dict]):
        batch = super().__call__(features)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                batch[key] = value[:, : self.max_length]
        return batch


def get_label(y):
    return y.predictions[0].argmax(-1)


def get_logits(y):
    if y.predictions[2].shape[-1] > 30000:
        return y.predictions[2]
    else:
        return y.predictions[3]


def get_topk(y, k):
    logits = get_logits(y)
    indices = np.argpartition(logits, -k, axis=1)[:, -k:]
    return indices


def jaccard(a, b):
    scores = []
    for single_a, single_b in zip(a, b):
        set_a = set(single_a)
        set_b = set(single_b)
        score = len(set_a.intersection(set_b)) / len(set_a.union(set_b))
        scores.append(score)
    return np.array(scores).mean()


def compress(args: CompressTopArgs):
    if os.path.exists(args.save_file_name):
        return
    # set_gpu(args.gpu)
    if args.sample_from == "test":
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)

    model = LMForwardAPI(
        model=model,
        model_name=args.model_name,
        tokenizer=tokenizer,
        device="cuda:0",
        label_dict=args.label_dict,
    )

    set_seed(args.seeds[0])

    data_collator = TruncatingDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=1,
        max_length=tokenizer.max_len_single_sentence,
    )
    training_args = TrainingArguments(
        "./output_dir",
        remove_unused_columns=False,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
    )

    num_layer = get_model_layer_num(
        model=model.model, model_name=args.model_name
    )
    predictor = Predictor(
        label_id_dict=args.label_id_dict,
        pad_token_id=tokenizer.pad_token_id,
        task_name=args.task_name,
        tokenizer=tokenizer,
        layer=num_layer,
    )
    context_solver = ContextSolver(
        task_name=args.task_name, tokenizer=tokenizer
    )

    def prepare_analysis_dataset(seed):
        demonstration, _ = sample_two_set_with_shot_per_class(
            dataset["train"],
            args.demonstration_shot,
            0,
            seed,
            label_name="label",
            a_total_shot=args.demonstration_total_shot,
        )
        if args.sample_from == "test":
            if len(dataset["test"]) < args.actual_sample_size:
                args.actual_sample_size = len(dataset["test"])
                warnings.warn(
                    f"sample_size: {args.sample_size} is larger than test set size: {len(dataset['test'])},"
                    f"actual_sample_size is {args.actual_sample_size}"
                )
            test_sample = (
                dataset["test"]
                .shuffle(seed=seed)
                .select(range(args.actual_sample_size))
            )
            demo_dataset = wrap_dataset(
                test_sample, demonstration, args.label_dict, args.task_name
            )
            demo_dataset = tokenize_dataset(demo_dataset, tokenizer)

            context = demo_dataset[0]["sentence"]
            instruct = context_solver.get_empty_demo_context(
                context, only_demo_part=True
            )

            empty_demo_dataset = wrap_dataset_with_instruct(
                test_sample, instruct, args.label_dict, args.task_name
            )
            empty_demo_dataset = tokenize_dataset(
                empty_demo_dataset, tokenizer
            )

            no_demo_dataset = wrap_dataset(
                test_sample, [], args.label_dict, args.task_name
            )
            no_demo_dataset = tokenize_dataset(no_demo_dataset, tokenizer)
        else:
            raise NotImplementedError(f"sample_from: {args.sample_from}")

        return demo_dataset, empty_demo_dataset, no_demo_dataset

    results = []

    for seed in args.seeds:
        (
            analysis_dataset,
            analysis_empty_demo_dataset,
            analysis_no_demo_dataset,
        ) = prepare_analysis_dataset(seed)

        model.use_past_key_values = False
        model.past_key_values = None
        model.results_args = {}
        trainer = Trainer(
            model=model, args=training_args, data_collator=data_collator
        )
        data = remove_str_columns(analysis_dataset)
        normal_y = trainer.predict(data, ignore_keys=["results"])

        model.results_args = {
            "output_hidden_states": True,
            "output_attentions": True,
            "use_cache": True,
        }
        trainer = Trainer(
            model=model, args=training_args, data_collator=data_collator
        )
        data = analysis_dataset.select([0])
        data = remove_str_columns(data)
        _ = trainer.predict(data)
        past_key_values = _.predictions[2].past_key_values
        past_key_values = tuple(
            tuple(torch.tensor(t) for t in tup) for tup in past_key_values
        )
        mask = context_solver.get_mask(data[0]["input_ids"])

        label_past_key_values = tuple(
            tuple(t[:, :, mask, :] for t in tup) for tup in past_key_values
        )

        model.use_past_key_values = True
        model.past_key_values = label_past_key_values
        model.results_args = {}
        model.probs_from_results_fn = None

        offset = torch.where(mask)[0][-1] + 1
        model.position_offset = offset

        trainer = Trainer(
            model=model, args=training_args, data_collator=data_collator
        )
        n_data = remove_str_columns(analysis_no_demo_dataset)
        label_y = trainer.predict(n_data, ignore_keys=["results"])

        true_label = n_data["labels"]
        normal_label = get_label(normal_y)
        l_label = get_label(label_y)
        wl = jaccard(get_topk(label_y, 5), get_topk(normal_y, 5))
        ll = accuracy_score(normal_label, l_label)
        acc = accuracy_score(true_label, l_label)

        os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
        with open(args.save_file_name + "label", "w") as f:
            f.write(f"{wl},{ll},{acc}, label")

        print(f"{wl},{ll},{acc}, label")

        data_collator = TruncatingDataCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=1,
            max_length=tokenizer.max_len_single_sentence - mask.sum(),
        )

        model.use_past_key_values = True
        model.results_args = {}
        model.probs_from_results_fn = None

        random_mask = torch.zeros_like(mask)
        selected_idxs = torch.where(mask)[0]
        other_idxs = set(list(range(selected_idxs[-1]))) - set(selected_idxs)
        class_poss, final_pos = predictor.get_pos(
            {"input_ids": torch.tensor(data[0]["input_ids"]).unsqueeze(0)}
        )
        class_poss = [_.unsqueeze(0) for _ in class_poss]
        class_poss_flatten = torch.flatten(
            torch.cat(class_poss, dim=-1)
        ).tolist()
        class_poss_flatten = sorted(class_poss_flatten)

        single_other_idxs_sets = []
        for i in range(len(class_poss_flatten)):
            before_idx = -1 if i == 0 else class_poss_flatten[i - 1]
            single_other_idxs = [
                j
                for j in other_idxs
                if (j > before_idx and j < class_poss_flatten[i])
            ]
            single_other_idxs = set(single_other_idxs) - set(
                selected_idxs.tolist()
            )
            single_other_idxs_sets.append(list(single_other_idxs))

        prod = 1
        for _ in single_other_idxs_sets:
            prod *= len(_)

        ks = list(range(prod))
        random.shuffle(ks)

        if args.ks_num != -1:
            ks = ks[: args.ks_num]

        for k in ks:
            single_other_idx_list = []
            t_k = k
            for _ in single_other_idxs_sets:
                cur_i = t_k % len(_)
                t_k = t_k // len(_)
                single_other_idx_list.append(_[cur_i])

            new_selected_idxs = list(
                (set(selected_idxs.tolist()) - set(class_poss_flatten))
                | set(single_other_idx_list)
            )

            random_mask[new_selected_idxs] = True
            random_past_key_values = tuple(
                tuple(t[:, :, random_mask, :] for t in tup)
                for tup in past_key_values
            )

            model.use_past_key_values = True
            model.past_key_values = random_past_key_values
            offset = torch.where(mask)[0][-1] + 1
            model.position_offset = offset
            trainer = Trainer(
                model=model, args=training_args, data_collator=data_collator
            )
            n_data = remove_str_columns(analysis_no_demo_dataset)
            compress_y = trainer.predict(n_data, ignore_keys=["results"])

            true_label = n_data["labels"]
            normal_label = get_label(normal_y)
            compress_label = get_label(compress_y)
            wl = jaccard(get_topk(compress_y, 5), get_topk(normal_y, 5))
            ll = accuracy_score(normal_label, compress_label)
            acc = accuracy_score(true_label, compress_label)

            results.append((wl, ll, acc, k))

            os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
            with open(args.save_file_name + str(k), "w") as f:
                f.write(f"{wl},{ll},{acc},{k}")

    os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
    with open(args.save_file_name, "wb") as f:
        pickle.dump(results, f)
