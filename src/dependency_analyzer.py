import ast
import os


def extract_imports(file_path):

    imports = []

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            tree = ast.parse(f.read())

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for name in node.names:
                    imports.append(name.name)

            elif isinstance(node, ast.ImportFrom):

                if node.module:
                    imports.append(node.module)

    except Exception:
        pass

    return imports


def build_dependency_graph(documents):

    graph = {}

    for doc in documents:

        graph[
            doc["file_name"]
        ] = extract_imports(
            doc["file_path"]
        )

    return graph