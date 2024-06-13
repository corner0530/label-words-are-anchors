import warnings

import torch


class Predictor:
    """
    Class representing a predictor for a specific task.

    Args:
        label_id_dict (dict): A dictionary mapping label names to their corresponding IDs.
        pad_token_id (int): The ID of the padding token in the tokenizer.
        task_name (str): The name of the task.
        tokenizer (any): The tokenizer used for tokenizing input sequences.
        layer (int): The number of layers in the model.
        naive_class_embs (any | None, optional): Naive class embeddings. Defaults to None.
        naive_final_emb (any | None, optional): Naive final embedding. Defaults to None.

    Attributes:
        naive_class_embs (any | None): Naive class embeddings.
        naive_final_emb (any | None): Naive final embedding.
        label_id_dict (dict): A dictionary mapping label names to their corresponding IDs.
        pad_token_id (int): The ID of the padding token in the tokenizer.
        task_name (str): The name of the task.
        tokenizer (any): The tokenizer used for tokenizing input sequences.
        layer (int): The number of layers in the model.
        prefix_idxs (list[int]): The indices of prefix tokens specific to each task.

    Raises:
        NotImplementedError: If the task name is not supported.

    """

    def __init__(
        self,
        label_id_dict: dict,
        pad_token_id: int,
        task_name: str,
        tokenizer: any,
        layer: int,
        naive_class_embs = None,
        naive_final_emb = None,
    ) -> None:
        self.naive_class_embs = naive_class_embs
        self.naive_final_emb = naive_final_emb
        self.label_id_dict = label_id_dict
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.layer = layer

        if task_name == "sst2":
            self.prefix_idxs = [
                tokenizer.encode("Sentiment", add_special_tokens=False)[-1],
                tokenizer.encode(":", add_special_tokens=False)[0],
            ]
        elif task_name == "agnews":
            self.prefix_idxs = [
                tokenizer.encode("Answer", add_special_tokens=False)[-1],
                tokenizer.encode(":", add_special_tokens=False)[0],
            ]
        elif task_name == "trec":
            self.prefix_idxs = [
                tokenizer.encode(" Type", add_special_tokens=False)[-1],
                tokenizer.encode(":", add_special_tokens=False)[0],
            ]
        elif task_name == "emo":
            self.prefix_idxs = [
                tokenizer.encode("Emotion", add_special_tokens=False)[-1],
                tokenizer.encode(":", add_special_tokens=False)[0],
            ]
        else:
            raise NotImplementedError(f"task_name: {task_name}")

    def get_pos(self, inputs: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Get the positions of class tokens and the final token in the input sequence.

        Args:
            inputs (dict): The input dictionary containing the "input_ids" tensor.

        Returns:
            tuple[list[torch.Tensor], torch.Tensor]: A tuple containing a list of class positions
                for each label and the position of the final token.

        """
        label_id_dict = self.label_id_dict
        pad_token_id = self.pad_token_id
        final_pos = (inputs["input_ids"] != pad_token_id).int().sum(-1) - 1
        device = inputs["input_ids"].device
        bsz, sql = inputs["input_ids"].shape
        class_poss = []
        for idx in label_id_dict.values():
            class_idx = idx
            for offset, prefix_idx in enumerate(reversed(self.prefix_idxs)):
                class_idx += prefix_idx * 100000 ** (offset + 1)
            input_ids = inputs["input_ids"].detach().clone()
            input_ids[:, 1:] += inputs["input_ids"][:, :-1] * 100000
            input_ids[:, 2:] += inputs["input_ids"][:, :-2] * 100000 * 100000
            class_pos = (
                torch.arange(sql, device=device)
                .unsqueeze(0)
                .repeat(bsz, 1)[input_ids == class_idx]
                .squeeze()
            )
            class_poss.append(class_pos)
        return class_poss, final_pos

    def _cal_all_key_and_values_of_class(
        self,
        inputs: dict,
        past_key_values: tuple[tuple[torch.Tensor]],
        one_class_one_list: bool = False,
        include_final: bool = False,
    ) -> tuple[tuple[torch.Tensor]]:
        """
        Calculate the key and value tensors for each class.

        Args:
            inputs (dict): The input dictionary containing the "input_ids" tensor.
            past_key_values (tuple[tuple[torch.Tensor]]): The past key and value tensors.
            one_class_one_list (bool, optional): Whether to return one class per list. Defaults to False.
            include_final (bool, optional): Whether to include the final token. Defaults to False.

        Returns:
            tuple[tuple[torch.Tensor]]: A tuple containing the key and value tensors for each class.

        """
        class_poss, final_pos = self.get_pos(inputs)

        if include_final:
            class_poss.append(final_pos)

        def get_vecs(
            ker_or_value: torch.Tensor, class_poss: list[torch.Tensor]
        ) -> torch.Tensor | list[torch.Tensor]:
            batch_idx = torch.arange(inputs["input_ids"].shape[0])
            class_vecs = []
            for poss in class_poss:
                class_vec = ker_or_value[batch_idx, :, poss, :]
                class_vecs.append(class_vec.unsqueeze(-2))
            if not one_class_one_list:
                class_vecs = torch.cat(class_vecs, dim=-2)
            return class_vecs

        key_and_values = []
        for layer in range(0, self.layer):
            key_and_values.append(
                tuple(
                    [get_vecs(_, class_poss) for _ in past_key_values[layer]]
                )
            )
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def cal_all_key_and_values_of_class(
        self,
        inputs: dict,
        results: any,
        one_class_one_list: bool = False,
        include_final: bool = False,
    ) -> tuple[tuple[torch.Tensor]]:
        """
        Calculate the key and value tensors for each class.

        Args:
            inputs (dict): The input dictionary containing the "input_ids" tensor.
            results (any): The results object containing the past key and value tensors.
            one_class_one_list (bool, optional): Whether to return one class per list. Defaults to False.
            include_final (bool, optional): Whether to include the final token. Defaults to False.

        Returns:
            tuple[tuple[torch.Tensor]]: A tuple containing the key and value tensors for each class.

        """
        past_key_values = results.past_key_values
        key_and_values = self._cal_all_key_and_values_of_class(
            inputs,
            past_key_values,
            one_class_one_list=one_class_one_list,
            include_final=include_final,
        )
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def get_attention(
        self,
        inputs: dict,
        results: any,
        layer: int,
    ) -> torch.Tensor:
        """
        Get the attention scores for each class at a specific layer.

        Args:
            inputs (dict): The input dictionary containing the "input_ids" tensor.
            results (any): The results object containing the attention scores.
            layer (int): The layer index.

        Returns:
            torch.Tensor: The attention scores for each class.

        """
        class_poss, final_pos = self.get_pos(inputs)
        batch_idx = torch.arange(inputs["input_ids"].shape[0])
        scores = []
        for class_pos in class_poss:
            attention = results.attentions[layer][
                batch_idx, :, final_pos, class_pos
            ]
            score = attention
            if class_pos.numel() == 1:
                score = score.sum(-1)
            else:
                score = score.sum()
            if inputs["input_ids"].shape[0] != 1:
                warnings.warn(f"Only support batch_size=1 now!")
            scores.append(score.unsqueeze(0))
        scores = torch.cat(scores, dim=0)
        return scores

    def cal_all_sim_attn(
        self,
        inputs: dict,
        results: any,
    ) -> torch.Tensor:
        """
        Calculate the similarity attention scores for each class at each layer.

        Args:
            inputs (dict): The input dictionary containing the "input_ids" tensor.
            results (any): The results object containing the attention scores.

        Returns:
            torch.Tensor: The similarity attention scores for each class at each layer.

        """
        sims = []
        for layer in range(0, self.layer):
            sim = self.get_attention(
                inputs=inputs, results=results, layer=layer
            )
            sims.append(sim.unsqueeze(1))
        sims = torch.cat(sims, dim=1)
        sims = sims.reshape(inputs["input_ids"].shape[0], -1)
        return sims
