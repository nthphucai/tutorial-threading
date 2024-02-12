import datetime
import glob
import json
import logging
import os
import pickle
import shutil
from typing import List, Optional

import pandas as pd

import yaml


__all__ = [
    "logger",
    "read_yaml_file",
    "read_text_file",
    "write_text_file",
    "remove_files",
    "write_pickle_file",
    "read_pickle_file",
    "write_json_file",
    "load_json_file",
    "unzip",
    "format_arg_str",
    "get_time",
    "log_func",
]

# Initialize the logger
logger = logging.getLogger(__name__)  # Create a logger with the function's name

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
# Create a handler to write logs to a file
file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(logging.INFO)  # Set the log level for the file handler
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)  # Add the file handler to the logger


def log_func(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.info(f"{result}")
        return result

    return wrapper


# Access to Yaml File
def read_yaml_file(file_path: str):
    with open(file_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


# Access to text files
def read_text_file(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        context = [item.strip() for item in f.readlines()]
    return context


def write_text_file(context: List[str], file_path: str):
    with open(file_path, "w") as f:
        for text in context:
            f.write(text + "\n")


# Remove all files in a directory
def remove_files(path):
    files = glob.glob(f"{path}/*")
    for f in files:
        os.remove(f)


# Access Pickle File
def write_pickle_file(data: dict, path: str, name: Optional[str] = None) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "wb")
    pickle.dump(data, f)
    f.close()


def read_pickle_file(path, name: Optional[str] = None) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "rb")
    pickle_file = pickle.load(f)
    return pickle_file


# Access JSON File
def write_json_file(
    data: dict, path: str, name: Optional[str] = None, **kwargs
) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path
    with open(save_path, "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4, **kwargs)


def load_json_file(path: str, name: Optional[str] = None, **kwargs) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path
    with open(save_path, encoding="utf-8") as outfile:
        data = json.load(outfile, **kwargs)
        return data


# Unzip Function
def unzip(zip_path, extract_path):
    shutil.unpack_archive(filename=zip_path, extract_dir=extract_path)


# Get current time
def get_time() -> str:
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def format_arg_str(args, max_len: int = 50) -> str:
    """
    Beauty arguments.
    Args:
        args: Input arguments.
        max_len (int): Max length of printing string.

    Returns:
        str: Output beautied arguments.
    """
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys()]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = "Arguments", "Values"
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = (
        max([len(key_title), key_max_len]),
        max([len(value_title), value_max_len]),
    )
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + "=" * horizon_len + linesep
    res_str += (
        " "
        + key_title
        + " " * (key_max_len - len(key_title))
        + " | "
        + value_title
        + " " * (value_max_len - len(value_title))
        + " "
        + linesep
        + "=" * horizon_len
        + linesep
    )
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace("\t", "\\t")
            value = value[: max_len - 3] + "..." if len(value) > max_len else value
            res_str += (
                " "
                + key
                + " " * (key_max_len - len(key))
                + " | "
                + value
                + " " * (value_max_len - len(value))
                + linesep
            )
    res_str += "=" * horizon_len
    return res_str
