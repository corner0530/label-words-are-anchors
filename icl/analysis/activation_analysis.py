import torch

from .numpy_writer import CPUTensorBufferDict, NumpyWriter

_save_dict: CPUTensorBufferDict | None = None
_save_activation: bool = False
_save_activation_grad: bool = False
_save_debug: bool = False
_writer_np: NumpyWriter | None = None


def set_save_activation(value: bool) -> None:
    """
    Set whether to save activation.

    Args:
        value (bool): Whether to save activation.

    Returns:
        None
    """
    global _save_activation
    _save_activation = value


def get_save_activation() -> bool:
    """
    Get whether to save activation.

    Returns:
        bool: Whether to save activation.
    """
    return _save_activation


def set_save_activation_grad(value: bool) -> None:
    """
    Set whether to save activation grad.

    Args:
        value (bool): Whether to save activation grad.

    Returns:
        None
    """
    global _save_activation_grad
    _save_activation_grad = value


def get_save_activation_grad() -> bool:
    """
    Get whether to save activation grad.

    Returns:
        bool: Whether to save activation grad.
    """
    return _save_activation_grad


def set_debug(value: bool) -> None:
    """
    Set whether to debug.

    Args:
        value (bool): Whether to debug.

    Returns:
        None
    """
    global _save_debug
    _save_debug = value


def get_debug() -> bool:
    """
    Get whether to debug.

    Returns:
        bool: Whether to debug.
    """
    return _save_debug


def clear_save_dict() -> None:
    """
    Clear save dict.

    Returns:
        None
    """
    _save_dict.clear()


def set_save_dict(writer_np: NumpyWriter | None) -> None:
    """
    Set save dict.

    Args:
        writer_np (NumpyWriter): Numpy writer.

    Returns:
        None

    Raises:
        RuntimeError: If save dict already exists.
    """
    global _save_dict
    if _save_dict is not None and writer_np is not None:
        raise RuntimeError("save_dict already exists")
    if writer_np is None:
        _save_dict = None
    else:
        _save_dict = CPUTensorBufferDict(writer_np=writer_np)


def set_writer_np(writer_np: NumpyWriter | None) -> None:
    """
    Set writer np.

    Args:
        writer_np (NumpyWriter): Numpy writer.

    Returns:
        None

    Raises:
        RuntimeError: If writer_np already exists.
    """
    global _writer_np
    if _writer_np is not None and writer_np is not None:
        raise RuntimeError("writer_np already exists")
    _writer_np = writer_np
    set_save_dict(writer_np)


def get_writer_np() -> NumpyWriter | None:
    """
    Get writer np.

    Returns:
        NumpyWriter: Numpy writer.
    """
    return _writer_np


def debug_fn(func):
    """
    Debug function.

    Args:
        func (function): Function.

    Returns:
        function: Wrapper function.
    """

    def wrapper(*args, **kwargs):
        if get_debug():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


debug_print = debug_fn(print)


def _add_tensor(
    name: str,
    value: torch.Tensor | list[torch.Tensor],
    save_type: str = "activation",
    mode: str | None = None,
    log_interval=None,
) -> None:
    """
    Add tensor.

    Args:
        name (str): Name.
        value (torch.Tensor): Value.
        save_type (str): Save type.
        mode (str): Mode.
        log_interval (int): Log interval.

    Returns:
        None

    Raises:
        RuntimeError: If unsupported type.
    """
    global _save_dict
    save_dict = _save_dict
    # Maybe more processing is needed, like processing bf16
    debug_print(
        f"add_{save_type} {name} {value.shape} {value.dtype}", flush=True
    )
    if isinstance(value, torch.Tensor):
        value = value.detach().clone()

        if (
            mode == "ijk->(i*j)k" or "ij->ij"
        ):  # Here you can splice the data from a few batch together before turning it into (num_token, dim) (doesn't work for ATTENTION, use it for HIDDEN)
            value = value.reshape(-1, value.shape[-1])
        else:
            raise RuntimeError(f"Unsupported mode {mode}")

        if (
            log_interval is not None
        ):  # You can skip storing every few tokens (doesn't work for attention, use it for hidden)
            assert mode is not None, "log_interval require mode"
            value = value[::log_interval, ...]

        if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
            value = value.float()
        value = value.cpu()
        value = value.numpy()
        save_dict[name].append(value)
    else:
        raise RuntimeError(f"Unsupported type {type(value)}")


def add_activation(
    input: torch.Tensor,
    name: str,
    mode: str | None = None,
    log_interval=None,
) -> torch.Tensor:
    """
    Add activation.

    Args:
        input (torch.Tensor): Input.
        name (str): Name.
        mode (str): Mode.
        log_interval (int): Log interval.

    Returns:
        torch.Tensor: Input.
    """
    if not get_save_activation():
        return input
    _add_tensor(name, input, "activation", mode, log_interval)
    return input


def _add_activation_grad(
    input: torch.Tensor,
    name: str,
    mode: str | None = None,
    log_interval=None,
) -> None:
    """
    Add activation grad.

    Args:
        input (torch.Tensor): Input.
        name (str): Name.
        mode (str): Mode.
        log_interval (int): Log interval.

    Returns:
        None
    """
    if not get_save_activation_grad():
        return
    _add_tensor(name, input, "activation_grad", mode, log_interval)


class IdentityToCatchGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        name: str,
        mode: str | None,
        log_interval,
    ) -> torch.Tensor:
        """
        Forward.

        Args:
            ctx (object): Context.
            input (torch.Tensor): Input.
            name (str): Name.
            mode (str): Mode.
            log_interval (int): Log interval.

        Returns:
            torch.Tensor: Input.
        """
        if not get_save_activation_grad():
            return input
        ctx.name = name
        ctx.mode = mode
        ctx.log_interval = log_interval
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward.

        Args:
            ctx (object): Context.
            grad_output (torch.Tensor): Grad output.

        Returns:
            torch.Tensor: Grad output.
        """
        if not get_save_activation_grad():
            return grad_output
        _add_activation_grad(grad_output, ctx.name, ctx.mode, ctx.log_interval)
        return grad_output, None, None, None


def add_activation_grad(
    input: torch.Tensor,
    name: str,
    mode: str | None = None,
    log_interval=None,
) -> torch.Tensor:
    """
    Add activation grad.

    Args:
        input (torch.Tensor): Input.
        name (str): Name.
        mode (str): Mode.
        log_interval (int): Log interval.

    Returns:
        torch.Tensor: Input.
    """
    if not name.endswith("_grad"):
        name = name + "_grad"
    if not get_save_activation_grad():
        return input
    input.requires_grad_(True)
    input = IdentityToCatchGrad.apply(
        input, name, mode, log_interval
    )  # can not set requires_grad=True in this
    return input


def force_save_write_clear() -> None:
    """
    Force save write clear.

    Returns:
        None
    """
    debug_print("force_activation_write_clear")
    debug_print("save_dict", _save_dict.keys())
    for name, buffer in _save_dict.items():
        buffer._write()
    clear_save_dict()


def start_save(
    log_dir: str,
    save_activation: bool = False,
    save_activation_grad: bool = False,
    debug: bool = False,
    continue_run: bool = False,
    cover: bool = False,
) -> None:
    """
    Start save.

    Args:
        log_dir (str): Log dir.
        save_activation (bool): Save activation.
        save_activation_grad (bool): Save activation grad.
        debug (bool): Debug.
        continue_run (bool): Continue run.
        cover (bool): Cover.

    Returns:
        None

    Raises:
        AssertionError: If nothing to save.
    """
    assert save_activation or save_activation_grad, "Nothing to save."
    mode = "a" if continue_run else "w"
    writer_np = NumpyWriter(log_dir=log_dir, mode=mode, cover=cover)
    set_writer_np(writer_np)
    set_save_activation(save_activation)
    set_save_activation_grad(save_activation_grad)
    set_debug(debug)


def end_save() -> None:
    """
    End save.

    Returns:
        None
    """
    set_save_activation(False)
    set_save_activation_grad(False)
    force_save_write_clear()
    set_writer_np(None)


def get_result(
    log_dir: str,
    name: str,
    idxs: list[int] | None = None,
    condition: dict[str, int | float | str] | None = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """
    Get result.

    Args:
        log_dir (str): Log dir.
        name (str): Name.
        idxs (list): Idxs.
        condition (dict): Condition.

    Returns:
        dict: Result.
    """
    writer_np = NumpyWriter(log_dir=log_dir, mode="r")
    return writer_np.read(name, idxs, condition)
