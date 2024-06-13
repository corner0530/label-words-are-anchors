from functools import partial, wraps

import torch
from torch import nn
from transformers import PreTrainedModel


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
        forward(attn_weights: torch.Tensor) -> torch.Tensor:
            Forward pass of the attention adapter.
        _forward(attn_weights: torch.Tensor) -> torch.Tensor:
            Abstract method to be implemented by subclasses.
        register_input_ids(input_ids: torch.Tensor) -> None:
            Registers the input IDs for the attention adapter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag: bool = True

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention adapter.

        Args:
            attn_weights (torch.Tensor): Tensor containing attention weights.

        Returns:
            torch.Tensor: Tensor containing modified attention weights.
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

        Returns:
            torch.Tensor: Tensor containing modified attention weights.
        """
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor) -> None:
        """
        Registers the input IDs for the attention adapter.

        Args:
            input_ids (torch.Tensor): Tensor containing input IDs.

        Returns:
            None
        """
        self.input_ids = input_ids


class AttentionAdapter(AttentionAdapterBase):
    """
    Adapter class for attention mechanism.

    This class extends the base class `AttentionAdapterBase` and provides
    functionality to modify attention weights using trainable parameters.

    Attributes:
        params (torch.Tensor): Trainable parameters used to modify attention weights.

    Methods:
        _forward(attn_weights: torch.Tensor) -> torch.Tensor:
            Applies the modification to the attention weights.

        grad() -> torch.Tensor:
            Returns the gradient of the trainable parameters.

        zero_grad(set_to_none: bool = False) -> None:
            Zeros out the gradient of the trainable parameters.

    """

    def __init__(self) -> None:
        super().__init__()
        self.params: torch.Tensor = None

    def _forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Applies the modification to the attention weights.

        Args:
            attn_weights (torch.Tensor): Attention weights to be modified.

        Returns:
            torch.Tensor: Modified attention weights.

        """
        if self.params is None:
            self.params = torch.ones_like(attn_weights, requires_grad=True)
        else:
            self.params.data = torch.ones_like(attn_weights)
        return attn_weights * self.params

    @property
    def grad(self) -> torch.Tensor:
        """
        Returns the gradient of the trainable parameters.

        Returns:
            torch.Tensor: Gradient of the trainable parameters.

        """
        return self.params.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Zeros out the gradient of the trainable parameters.

        Args:
            set_to_none (bool, optional): If True, sets the gradient to None instead of zeroing it out.
                Defaults to False.

        """
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                self.params.grad = torch.zeros_like(self.params.grad)


def gpt2_attn(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor = None,
    head_mask: torch.Tensor = None,
    attention_adapter: AttentionAdapter = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the attention mechanism for GPT-2 model.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, num_heads, query_length, head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, num_heads, key_length, head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, num_heads, value_length, head_dim).
        attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, num_heads, query_length, key_length). Defaults to None.
        head_mask (torch.Tensor, optional): The head mask tensor of shape (num_heads, head_dim). Defaults to None.
        attention_adapter (AttentionAdapter, optional): The attention adapter. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The attention output tensor of shape (batch_size, num_heads, query_length, head_dim) and the attention weights tensor of shape (batch_size, num_heads, query_length, key_length).
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

    Attributes:
        model (PreTrainedModel): The pre-trained model.
        attention_adapters (list[AttentionAdapter]): List of attention adapters.
        _input_ids (torch.Tensor): Input IDs for the model.

    Methods:
        input_ids (property): Getter method for input_ids.
        input_ids (setter): Setter method for input_ids.
        register_input_ids: Registers input_ids for attention adapters.
        register_attentioner_to_model: Abstract method to register attention adapters to the model.
        zero_grad: Sets gradients to zero for attention adapters.
        grad_process: Processes gradients for attention adapters.
        grad: Computes gradients for attention adapters.

    """

    def __init__(self, model: PreTrainedModel):
        self.model: PreTrainedModel = model
        self.attention_adapters: list[AttentionAdapter] = (
            self.register_attentioner_to_model()
        )
        self.model.forward = manager_decoractor(self)(self.model.forward)

    @property
    def input_ids(self) -> torch.Tensor:
        """
        Getter method for input_ids.

        Returns:
            torch.Tensor: The input IDs for the model.

        """
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids: torch.Tensor) -> None:
        """
        Setter method for input_ids.

        Args:
            input_ids (torch.Tensor): The input IDs for the model.

        """
        self._input_ids = input_ids
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids: torch.Tensor) -> None:
        """
        Registers input_ids for attention adapters.

        Args:
            input_ids (torch.Tensor): The input IDs for the model.

        """
        self.input_ids = input_ids

    def register_attentioner_to_model(self) -> list[AttentionAdapter]:
        """
        Abstract method to register attention adapters to the model.

        Returns:
            list[AttentionAdapter]: List of attention adapters.

        """
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Sets gradients to zero for attention adapters.

        Args:
            set_to_none (bool, optional): Whether to set gradients to None. Defaults to True.

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
        Processes gradients for attention adapters.

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
        Computes gradients for attention adapters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[torch.Tensor]: List of computed gradients.

        """
        grads = []
        for attention_adapter in self.attention_adapters:
            grads.append(
                self.grad_process(
                    attention_adapter.params.grad, *args, **kwargs
                )
            )
        return grads


def manager_decoractor(manager):
    """
    Decorator function that registers the input_ids with the given AttentionerManager.

    Args:
        manager (AttentionerManagerBase): The AttentionerManager instance to register the input_ids with.

    Returns:
        callable: The decorated function.
    """

    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs) -> any:
            input_ids: torch.Tensor = kwargs.get("input_ids", None)
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

    Attributes:
        model (PreTrainedModel): The GPT-2 model.

    Methods:
        register_attentioner_to_model: Registers attention adapters to the GPT-2 model.

    """

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)

    def register_attentioner_to_model(self) -> list[AttentionAdapter]:
        """
        Registers attention adapters to the GPT-2 model.

        Returns:
            list[AttentionAdapter]: List of attention adapters.

        """
        attention_adapters: list[AttentionAdapter] = []
        for i, layer in enumerate(self.model.transformer.h):
            attention_adapter: AttentionAdapter = AttentionAdapter()
            layer.attn._attn = partial(
                gpt2_attn, layer.attn, attention_adapter=attention_adapter
            )
            attention_adapters.append(attention_adapter)
        return attention_adapters
