# import warnings
from functools import partial, wraps

import torch
from torch import nn

# from torch.nn import functional as F
from transformers import PreTrainedModel

from icl.analysis.attentioner_for_attribution import AttentionAdapter
from icl.util_classes.predictor_classes import Predictor

# from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class AttentionAdapterBase(nn.Module):
    """
    Base class for attention adapters.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        use_flag (bool): Flag indicating whether to use the attention adapter.
        input_ids (torch.Tensor): Tensor containing input IDs.

    Methods:
        forward(attn_weights): Performs the forward pass of the attention adapter.
        _forward(attn_weights): Abstract method to be implemented by subclasses.
        register_input_ids(input_ids): Registers the input IDs for the attention adapter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag: bool = True

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the attention adapter.

        Args:
            attn_weights (torch.Tensor): Tensor containing attention weights.

        Returns:
            torch.Tensor: Tensor containing modified attention weights if `use_flag` is True, else returns the original attention weights.
        """
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses.

        Args:
            attn_weights (torch.Tensor): Tensor containing attention weights.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor) -> None:
        """
        Registers the input IDs for the attention adapter.

        Args:
            input_ids (torch.Tensor): Tensor containing input IDs.
        """
        self.input_ids = input_ids


def gpt2_attn(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor = None,
    head_mask: torch.Tensor = None,
    attention_adapter: torch.nn.Module = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the attention output and attention weights for the GPT-2 model.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, num_heads, seq_length, head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, num_heads, seq_length, head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, num_heads, seq_length, head_dim).
        attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, 1, seq_length, seq_length). Defaults to None.
        head_mask (torch.Tensor, optional): The head mask tensor of shape (num_heads, seq_length, seq_length). Defaults to None.
        attention_adapter (torch.nn.Module, optional): The attention adapter module. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The attention output tensor of shape (batch_size, num_heads, seq_length, head_dim) and the attention weights tensor of shape (batch_size, num_heads, seq_length, seq_length).
    """
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ].bool()
        attn_weights = torch.where(
            causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
        )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    if attention_adapter is not None:
        attn_weights = attention_adapter(attn_weights)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


class AttentionerManagerBase:
    """
    Base class for managing attention adapters in a model.

    Args:
        model (PreTrainedModel): The pre-trained model.
        predictor (Predictor): The predictor object.
        n_demo (int): The number of demos.
        device (str): The device to run the model on.
        n_head (int): The number of attention heads.

    Attributes:
        n_demo (int): The number of demos.
        n_head (int): The number of attention heads.
        device (str): The device to run the model on.
        model (PreTrainedModel): The pre-trained model.
        attention_adapters (list[AttentionAdapter]): List of attention adapters.
        predictor (Predictor): The predictor object.

    """

    def __init__(
        self,
        model: PreTrainedModel,
        predictor: Predictor,
        n_demo: int,
        device: str,
        n_head: int,
    ) -> None:
        self.n_demo = n_demo
        self.n_head = n_head
        self.device = device
        self.model = model
        self.attention_adapters = self.register_attentioner_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)
        self.predictor = predictor

    @property
    def input_ids(self) -> torch.Tensor:
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids: torch.Tensor) -> None:
        self._input_ids = input_ids
        class_poss, final_poss = self.predictor.get_pos(
            {"input_ids": input_ids}
        )
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)
            attention_adapter.class_poss = class_poss
            attention_adapter.final_poss = final_poss

    def register_input_ids(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids

    def register_attentioner_to_model(self) -> list[AttentionAdapter]:
        """
        Abstract method to register attention adapters to the model.
        Subclasses should implement this method.

        Returns:
            list[AttentionAdapter]: List of attention adapters.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Sets gradients of attention adapters to zero.

        Args:
            set_to_none (bool, optional): Whether to set gradients to None instead of zero. Defaults to True.
        """
        if set_to_none:
            for attention_adapter in self.attention_adapters:
                attention_adapter.params = None
        else:
            for attention_adapter in self.attention_adapters:
                attention_adapter.zero_grad(set_to_none=True)

    def grad_process(
        self, grad: torch.Tensor, use_abs: bool = True
    ) -> torch.Tensor:
        """
        Processes the gradients of attention adapters.

        Args:
            grad (torch.Tensor): The gradients to be processed.
            use_abs (bool, optional): Whether to use absolute values of gradients. Defaults to True.

        Returns:
            torch.Tensor: The processed gradients.
        """
        assert len(grad.shape) == 4
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self, *args, **kwargs) -> list[torch.Tensor]:
        """
        Computes the gradients of attention adapters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[torch.Tensor]: List of gradients.
        """
        grads = []
        for attention_adapter in self.attention_adapters:
            grads.append(
                self.grad_process(
                    attention_adapter.params.grad, *args, **kwargs
                )
            )
        return grads

    def params(self) -> list[torch.Tensor]:
        """
        Returns the parameters of attention adapters.

        Returns:
            list[torch.Tensor]: List of parameters.
        """
        params = []
        for attention_adapter in self.attention_adapters:
            params.append(attention_adapter.weight)
        return params


def manager_decoractor(manager):
    """
    Decorator function that registers the input_ids with the given AttentionerManagerBase instance.

    Args:
        manager (AttentionerManagerBase): The AttentionerManagerBase instance to register the input_ids with.

    Returns:
        callable: The decorated function.

    """

    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator


class GPT2AttentionerManager(AttentionerManagerBase):
    """
    Manager class for GPT-2 attentioner.

    Args:
        model (PreTrainedModel): The GPT-2 model.
        n_demo (int): The number of demonstrations.
        predictor (Predictor): The predictor object.
        device (str): The device to run the model on.
        n_head (int, optional): The number of attention heads. Defaults to 1.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        n_demo: int,
        predictor: Predictor,
        device: str,
        n_head: int = 1,
    ) -> None:
        super().__init__(model, predictor, n_demo, device, n_head=n_head)

    def register_attentioner_to_model(self) -> list[AttentionAdapter]:
        """
        Registers the attentioner to the GPT-2 model.

        Returns:
            list[AttentionAdapter]: The list of attention adapters.
        """
        attention_adapters = []
        for i, layer in enumerate(self.model.transformer.h):
            attention_adapter = AttentionAdapter(
                n_demo=self.n_demo, device=self.device, n_head=self.n_head
            )
            layer.attn._attn = partial(
                gpt2_attn, layer.attn, attention_adapter=attention_adapter
            )
            attention_adapters.append(attention_adapter)
        return attention_adapters


class AttentionAdapter(AttentionAdapterBase):
    """
    Adapter class for attention mechanism.

    Args:
        n_demo (int): Number of demos.
        n_head (int): Number of attention heads.
        device (str): Device to be used.

    Attributes:
        n_demo (int): Number of demos.
        n_head (int): Number of attention heads.
        weight (torch.nn.Parameter): Weight parameter for attention.
        class_poss (torch.Tensor | None): Class positions.
        final_poss (torch.Tensor | None): Final positions.
    """

    def __init__(self, n_demo: int, n_head: int, device: str) -> None:
        super().__init__()
        self.n_demo = n_demo
        self.n_head = n_head
        self.weight = torch.nn.Parameter(
            torch.zeros((n_head, n_demo), requires_grad=True, device=device)
        )
        self.class_poss: torch.Tensor | None = None
        self.final_poss: torch.Tensor | None = None

    def _forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to the given attention weights.

        Args:
            attn_weights (torch.Tensor): Attention weights.

        Returns:
            torch.Tensor: Attention weights after applying the attention mechanism.
        """
        class_poss = self.class_poss
        final_poss = self.final_poss
        weight = self.weight.exp()
        bsz, n_head, seq_len, _ = attn_weights.shape
        assert bsz == 1
        mask_mat = torch.ones(
            (1, n_head, seq_len, seq_len), device=attn_weights.device
        )
        mask_mat[:, :, final_poss, class_poss] = weight.reshape(
            1, self.n_head, self.n_demo
        )
        return attn_weights * mask_mat

    @property
    def grad(self) -> torch.Tensor | None:
        """
        Get the gradient of the weight parameter.

        Returns:
            torch.Tensor | None: Gradient of the weight parameter.
        """
        return self.weight.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Reset the gradient of the weight parameter.

        Args:
            set_to_none (bool, optional): Whether to set the gradient to None. Defaults to False.
        """
        if self.weight.grad is not None:
            if set_to_none:
                self.weight.grad = None
            else:
                self.weight.grad = torch.zeros_like(self.weight.grad)
