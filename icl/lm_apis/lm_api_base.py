import torch
import torch.nn as nn
import torch.nn.functional as F

from icl.utils.other import dict_to


class LMForwardAPI(nn.Module):
    """
    LMForwardAPI is a class that provides forward pass functionality for language models.

    Args:
        model (nn.Module): The language model.
        model_name (str): The name of the language model.
        tokenizer (any): The tokenizer used for tokenizing input.
        label_dict (dict[int, str]): A dictionary mapping label indices to label names.
        device (str, optional): The device to run the model on. Defaults to "cuda:0".

    Attributes:
        _use_past_key_values (bool): Flag indicating whether past key values are used.
        _past_key_values (tuple[tuple[torch.Tensor, torch.Tensor]] | None): The past key values.
        model (nn.Module): The language model.
        model_name (str): The name of the language model.
        tokenizer (any): The tokenizer used for tokenizing input.
        device (str): The device the model is running on.
        calibration_probs (torch.Tensor | None): Calibration probabilities.
        use_calibration_probs (bool): Flag indicating whether to use calibration probabilities.
        probs_from_results_fn (callable | None): Function to calculate probabilities from results.
        results_args (dict[str, any]): Additional arguments for the model results.
        label_map (dict[int, int]): A dictionary mapping tokenized label indices to label indices.
        position_offset (int): The position offset for position IDs.

    Properties:
        device (str): The device the model is running on.
        past_key_values (tuple[tuple[torch.Tensor, torch.Tensor]] | None): The past key values.
        use_past_key_values (bool): Flag indicating whether past key values are used.

    Methods:
        cal_logits(inputs: dict[str, torch.Tensor], **kwargs: any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            Calculates the logits for the given inputs.
        _cal_probs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            Calculates the probabilities from the logits.
        cal_probs(inputs: dict[str, torch.Tensor], **kwargs: any) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
            Calculates the probabilities and logits for the given inputs.
        cal_probs_from_results(inputs: dict[str, torch.Tensor], results: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            Calculates the probabilities from the results.
        get_mask_with_past_key_values(mask: torch.Tensor) -> torch.Tensor:
            Gets the mask with past key values.
        get_past_key_values(inputs: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
            Gets the past key values.
        forward_no_grad(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            Performs a forward pass without gradients.
        forward(**kwargs: any) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
            Performs a forward pass with gradients.

    Raises:
        ValueError: If past_key_values is None and it is required.

    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        tokenizer,
        label_dict: dict[int, str],
        device: str = "cuda:0",
    ) -> None:
        """
        Initializes the LMForwardAPI class.

        Args:
            model (nn.Module): The language model.
            model_name (str): The name of the language model.
            tokenizer (any): The tokenizer used for tokenizing input.
            label_dict (dict[int, str]): A dictionary mapping label indices to label names.
            device (str, optional): The device to run the model on. Defaults to "cuda:0".

        """
        super().__init__()
        self._use_past_key_values: bool = False
        self._past_key_values: (
            tuple[tuple[torch.Tensor, torch.Tensor]] | None
        ) = None
        self.model: nn.Module = model
        self.model_name: str = model_name
        self.tokenizer: any = tokenizer
        self.device: str = device
        self.model.eval()
        self.calibration_probs: torch.Tensor | None = None
        self.use_calibration_probs: bool = False
        self.probs_from_results_fn = None
        self.results_args: dict[str, any] = {}
        self.label_map: dict[int, int] = {
            tokenizer.encode(v, add_special_tokens=False)[0]: k
            for k, v in label_dict.items()
        }
        self.position_offset: int = 0

        assert model_name in ["gpt2-xl", "gpt-j-6b"]

    @property
    def device(self) -> str:
        """
        Returns the device on which the model is currently running.

        Returns:
            str: The device name.
        """
        return self.model.device

    @device.setter
    def device(self, device: str) -> None:
        """
        Sets the device for the LMForwardAPI.

        Args:
            device (str): The device to set.

        Returns:
            None
        """
        print(f"LMForwardAPI: set device to {device}")
        self.model = self.model.to(device)
        if self.past_key_values:
            self.past_key_values = self.past_key_values  # will reset device

    def cal_logits(
        self, inputs: dict[str, torch.Tensor], **kwargs: any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculates the logits for the given inputs.

        Args:
            inputs (dict[str, torch.Tensor]): The input tensors.
            **kwargs (any): Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the logits and additional results.
        """
        self.model.eval()
        inputs = dict_to(inputs, self.device)

        if self.use_past_key_values:
            past_key_values = self.get_past_key_values(inputs)
            kwargs["past_key_values"] = past_key_values
            inputs["attention_mask"] = self.get_mask_with_past_key_values(
                inputs["attention_mask"]
            )
            if self.model_name in ["gpt-j-6b", "gpt2-xl"]:
                bsz, sql = inputs["input_ids"].shape
                position_ids = torch.arange(
                    sql, dtype=torch.long, device=self.device
                ).repeat(bsz, 1)
                position_ids = position_ids + self.position_offset
                kwargs["position_ids"] = position_ids

        results = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs,
        )
        logits = results["logits"]
        # find last position before pad tokens
        input_ids = inputs["input_ids"]
        eos_token_id: int = self.tokenizer.eos_token_id
        is_not_eos = input_ids != eos_token_id
        prediction_pos = is_not_eos.sum(dim=1) - 1
        is_not_eos = is_not_eos.float()
        # check all eos_tokens are at the end
        assert (is_not_eos[:, :-1] - is_not_eos[:, 1:] >= 0).all()
        # get logits for the last position
        logits = logits[torch.arange(input_ids.shape[0]), prediction_pos, :]
        return logits, results

    def _cal_probs(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate probabilities and logits for the given logits tensor.

        Args:
            logits (torch.Tensor): The input tensor containing logits.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the calculated probabilities
            and logits.

        Raises:
            AssertionError: If `self.use_calibration_probs` is True but `self.calibration_probs`
            is None.
        """
        interest_index = list(self.label_map.keys())
        logits = logits[:, interest_index]
        probs = F.softmax(logits, dim=-1)
        if self.use_calibration_probs:
            assert self.calibration_probs is not None
            probs = probs / self.calibration_probs
        return probs, logits

    def cal_probs(
        self, inputs: dict[str, torch.Tensor], **kwargs: any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculates the probabilities of the inputs using the model.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary of input tensors.
            **kwargs (any): Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the probabilities,
            logits, and additional results.

        """
        logits, results = self.cal_logits(inputs, **kwargs)
        probs, logits = self._cal_probs(logits)
        return probs, logits, results

    def cal_probs_from_results(
        self, inputs: dict[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Calculate probabilities from the given inputs and results.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.
            results (dict[str, torch.Tensor]): A dictionary containing result tensors.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing probability tensors.
        """
        return self.probs_from_results_fn(inputs, results)

    @property
    def past_key_values(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor]] | None:
        """
        Get the past key values.

        Returns:
            A tuple of tuples containing torch.Tensor objects representing the past key values.
            If there are no past key values, None is returned.
        """
        return self._past_key_values

    @past_key_values.setter
    def past_key_values(
        self, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]] | None
    ) -> None:
        """
        Sets the past key values for the language model.

        Args:
            past_key_values (tuple[tuple[torch.Tensor, torch.Tensor]] | None):
                The past key values to be set. It should be a tuple of tuples,
                where each inner tuple contains two torch.Tensor objects.

        Returns:
            None
        """
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple)
            assert isinstance(past_key_values[0], tuple)
            assert len(past_key_values[0]) == 2
            assert isinstance(past_key_values[0][0], torch.Tensor)
            assert past_key_values[0][0].shape[0] == 1
            self._past_key_values = tuple(
                tuple(t.to(self.device) for t in tup)
                for tup in past_key_values
            )
        else:
            self._past_key_values = None

    @property
    def use_past_key_values(self) -> bool:
        """
        Returns a boolean value indicating whether past key values are being used.

        Returns:
            bool: True if past key values are being used, False otherwise.
        """
        return self._use_past_key_values

    @use_past_key_values.setter
    def use_past_key_values(self, use_past_key_values: bool) -> None:
        """
        Set whether to use past key values.

        Args:
            use_past_key_values (bool): A boolean value indicating whether to use past key values.
        """
        self._use_past_key_values = use_past_key_values

    def get_mask_with_past_key_values(
        self, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns a mask tensor with past key values.

        Args:
            mask (torch.Tensor): The input mask tensor.

        Returns:
            torch.Tensor: The mask tensor with past key values.
        """
        if self.past_key_values is None:
            raise ValueError("past_key_values is None, please set it first")
        batch_size = mask.shape[0]
        past_key_values_len = self.past_key_values[0][0].shape[2]
        mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    past_key_values_len,
                    dtype=torch.bool,
                    device=self.device,
                ),
                mask,
            ],
            dim=1,
        )
        return mask

    def get_past_key_values(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieves the past key values for the given inputs.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.

        Returns:
            tuple[tuple[torch.Tensor, torch.Tensor]]: A tuple of past key values, where each element is a tuple
            containing the layer key and layer value tensors.

        Raises:
            ValueError: If the past_key_values is None.

        """
        if self.past_key_values is None:
            raise ValueError("past_key_values is None, please set it first")
        batch_size = inputs["input_ids"].shape[0]
        past_key_values = ()
        for layer_key, layer_value in self.past_key_values:
            past_key_values += (
                (
                    layer_key.expand(batch_size, -1, -1, -1),
                    layer_value.expand(batch_size, -1, -1, -1),
                ),
            )

        return past_key_values

    @torch.no_grad()
    def forward_no_grad(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Perform a forward pass through the model without gradient computation.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary of input tensors.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the output probabilities and additional results.

        """
        ori_logits, results = self.cal_logits(inputs, **self.results_args)
        probs, logits = self._cal_probs(ori_logits)
        probs_from_results = self.cal_probs_from_results(inputs, results)
        probs_from_results["ori_logits"] = ori_logits
        return probs, probs_from_results

    def forward(
        self, **kwargs: any
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Forward pass of the LM API model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: A dictionary containing the following keys:
                - "probs": The predicted probabilities.
                - "logits": The predicted logits.
                - "results": The results obtained from calculating logits.
                - "probs_from_results" (optional): The predicted probabilities calculated from the results.
                - "ori_logits": The original logits before calculating probabilities.

        """
        ori_logits, results = self.cal_logits(kwargs, **self.results_args)
        probs, logits = self._cal_probs(ori_logits)
        result = {"probs": probs, "logits": logits, "results": results}
        if self.probs_from_results_fn:
            probs_from_results = self.cal_probs_from_results(kwargs, results)
            result["probs_from_results"] = probs_from_results
        result["ori_logits"] = ori_logits
        return result
