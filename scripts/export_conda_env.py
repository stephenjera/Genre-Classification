import os
from typing import TypedDict

import chardet


class EncodingResult(TypedDict):
    encoding: str | None
    confidence: float
    language: str | None


def export_conda_env(path: str) -> None:
    """Exports the current conda environment to a YAML file."""
    os.system(command=f"conda env export > {path}")


def check_encoding(file_path: str) -> str | None:
    """Detects the encoding of a file."""
    with open(file=file_path, mode="rb") as f:
        result: EncodingResult = chardet.detect(byte_str=f.read())
    return result["encoding"]


def convert_to_utf8(file_path: str, encoding: str | None) -> None:
    """Converts a file to UTF-8 encoding."""
    # Open the file with the detected encoding
    with open(file=file_path, mode="r", encoding=encoding) as f:
        lines: list[str] = f.readlines()

    # Remove the line that starts with 'prefix:'
    lines = [line for line in lines if not line.startswith("prefix:")]

    # Write the lines back out in UTF-8
    with open(file=file_path, mode="w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    file_path = "../.devcontainer/environment.yaml"
    export_conda_env(path=file_path)
    encoding: str | None = check_encoding(file_path=file_path)
    print(f"The encoding of {file_path} is {encoding}.")
    convert_to_utf8(file_path=file_path, encoding=encoding)
