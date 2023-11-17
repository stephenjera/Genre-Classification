import os
import chardet


def export_conda_env(path):
    os.system(f"conda env export > {path}")


def check_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


def convert_to_utf8(file_path, encoding):
    # Open the file with the detected encoding
    with open(file_path, "r", encoding=encoding) as f:
        lines = f.readlines()

    # Remove the line that starts with 'prefix:'
    lines = [line for line in lines if not line.startswith("prefix:")]

    # Write the lines back out in UTF-8
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    file_path = "../environment.yaml"
    export_conda_env(file_path)
    encoding = check_encoding(file_path)
    print(f"The encoding of {file_path} is {encoding}.")
    convert_to_utf8(file_path, encoding)
