import os.path

from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_PATH_LIST = [
    "Your_path"
]  # add your model path if you want to load local models


def convert_path_old(path: str, ROOT_PATH: str, load_type: str) -> str:
    """
    Converts a given path to a new path based on the load type.

    Args:
        path (str): The original path.
        ROOT_PATH (str): The root path.
        load_type (str): The type of load (either "tokenizer" or "model").

    Returns:
        str: The converted path.

    Raises:
        AssertionError: If the load_type is not "tokenizer" or "model".
    """
    assert load_type in ["tokenizer", "model"]
    return os.path.join(ROOT_PATH, load_type + "s", path)


def convert_path(path: str, ROOT_PATH: str, load_type: str) -> str:
    """
    Converts a relative path to an absolute path by joining it with the ROOT_PATH.

    Args:
        path (str): The relative path to be converted.
        ROOT_PATH (str): The root path to be joined with the relative path.
        load_type (str): The type of load operation ("tokenizer" or "model").

    Returns:
        str: The absolute path obtained by joining the ROOT_PATH and the relative path.
    """
    assert load_type in ["tokenizer", "model"]
    return os.path.join(ROOT_PATH, path)


def load_local_model_or_tokenizer(
    model_name: str, load_type: str
) -> AutoModelForCausalLM | AutoTokenizer | None:
    """
    Load a local model or tokenizer based on the specified model name and load type.

    Args:
        model_name (str): The name of the model to load.
        load_type (str): The type of object to load, either 'tokenizer' or 'model'.

    Returns:
        AutoModelForCausalLM | AutoTokenizer | None: The loaded model or tokenizer object, or None if the object could not be loaded.

    Raises:
        ValueError: If the specified load type is not supported.

    """
    if load_type in "tokenizer":
        LoadClass = AutoTokenizer
    elif load_type in "model":
        LoadClass = AutoModelForCausalLM
    else:
        raise ValueError(f"load_type: {load_type} is not supported")

    model = None
    for ROOT_PATH in ROOT_PATH_LIST:
        try:
            folder_path = convert_path_old(model_name, ROOT_PATH, load_type)
            if not os.path.exists(folder_path):
                continue
            print(f"loading {model_name} {load_type} from {folder_path} ...")
            model = LoadClass.from_pretrained(folder_path)
            print("finished loading")
            break
        except:
            continue
    if model is not None:
        return model
    for ROOT_PATH in ROOT_PATH_LIST:
        try:
            folder_path = convert_path(model_name, ROOT_PATH, load_type)
            if not os.path.exists(folder_path):
                continue
            print(f"loading {model_name} {load_type} from {folder_path} ...")
            model = LoadClass.from_pretrained(folder_path)
            print("finished loading")
            break
        except:
            continue
    return model


def get_model_layer_num(
    model: AutoModelForCausalLM | AutoTokenizer | None = None,
    model_name: str | None = None,
) -> int | None:
    """
    Get the number of layers in a model.

    Args:
        model (AutoModelForCausalLM | AutoTokenizer | None): The model object.
        model_name (str | None): The name of the model.

    Returns:
        int | None: The number of layers in the model.

    Raises:
        ValueError: If the number of layers cannot be obtained from the model or model_name.
    """
    num_layer = None
    if model is not None:
        if hasattr(model.config, "num_hidden_layers"):
            num_layer = model.config.num_hidden_layers
        elif hasattr(model.config, "n_layers"):
            num_layer = model.config.n_layers
        elif hasattr(model.config, "n_layer"):
            num_layer = model.config.n_layer
        else:
            pass
    elif model_name is not None:
        pass
    if num_layer is None:
        raise ValueError(
            f"cannot get num_layer from model: {model} or model_name: {model_name}"
        )
    return num_layer
