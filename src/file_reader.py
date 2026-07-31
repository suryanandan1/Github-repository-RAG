import os


SUPPORTED_EXTENSIONS = [".py"]


def get_python_files(repo_path):

    python_files = []

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if d not in [
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "node_modules"
            ]
        ]

        for file in files:

            if file.endswith(".py"):

                python_files.append(
                    os.path.join(root, file)
                )

    return python_files


def read_python_file(file_path):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            content = f.read()

        return content

    except Exception:

        return ""


def create_documents(file_paths):

    documents = []

    for file_path in file_paths:

        content = read_python_file(
            file_path
        )

        line_count = len(
            content.splitlines()
        )

        document = {
            "file_name":
                os.path.basename(file_path),

            "file_path":
                file_path,

            "content":
                content,

            "line_count":
                line_count,

            "size":
                len(content)
        }

        documents.append(
            document
        )

    return documents