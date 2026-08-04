def calculate_health_score(
    stats
):

    score = 100

    if stats["total_files"] < 3:
        score -= 20

    if stats["total_lines"] < 200:
        score -= 20

    return max(score, 0)