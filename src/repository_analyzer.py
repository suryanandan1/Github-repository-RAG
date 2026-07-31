import os


def calculate_repository_stats(documents):

    total_files = len(documents)

    total_lines = sum(
        doc["line_count"]
        for doc in documents
    )

    largest_file = max(
        documents,
        key=lambda x: x["line_count"],
        default=None
    )

    return {
        "total_files": total_files,
        "total_lines": total_lines,
        "largest_file": (
            largest_file["file_name"]
            if largest_file else "N/A"
        ),
        "largest_file_lines": (
            largest_file["line_count"]
            if largest_file else 0
        )
    }