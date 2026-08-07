import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

# Kept independent from src/llm_service.py on purpose, same as
# src/commit_service.py - Repository Activity Intelligence should
# not depend on (or risk destabilizing) the existing RAG chat
# client/prompting.
_ACTIVITY_MODEL_NAME = "codestral-latest"
_activity_client = None


class GitHubActivityError(Exception):
    """
    Raised for any user-facing Repository Activity Intelligence
    failure (e.g. AI summary generation failure).
    """
    pass


def _get_activity_client():

    global _activity_client

    if _activity_client is None:

        _activity_client = Mistral(
            api_key=os.getenv("MISTRALAI_API_KEY")
        )

    return _activity_client


def _parse_date(date_str):

    if not date_str:
        return None

    try:
        return datetime.strptime(
            date_str,
            "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def get_developer_activity(commits):
    """
    Aggregate commits per contributor: commit count and most
    recent commit date, sorted by commit count descending (most
    active contributor first).
    """

    if not commits:
        return []

    per_author_count = Counter()
    per_author_last_date = {}

    for commit in commits:

        name = commit.get("author_name", "Unknown")
        per_author_count[name] += 1

        date = commit.get("date", "")

        if (
            name not in per_author_last_date
            or date > per_author_last_date[name]
        ):
            per_author_last_date[name] = date

    activity = [
        {
            "name": name,
            "commits": count,
            "last_commit_date": per_author_last_date.get(name, "")
        }
        for name, count in per_author_count.items()
    ]

    activity.sort(
        key=lambda x: x["commits"],
        reverse=True
    )

    return activity


def get_commit_timeline(commits, granularity="day"):
    """
    Bucket commits into counts per day / week / month.
    Returns a dict of {bucket_label: count}, sorted chronologically.
    """

    buckets = defaultdict(int)

    for commit in commits:

        date = _parse_date(
            commit.get("date")
        )

        if not date:
            continue

        if granularity == "day":
            key = date.strftime("%Y-%m-%d")
        elif granularity == "week":
            key = date.strftime("%Y-W%W")
        else:
            key = date.strftime("%Y-%m")

        buckets[key] += 1

    return dict(
        sorted(buckets.items())
    )


def filter_commits_by_days(commits, days):
    """
    Keep only commits within the last `days` days.
    """

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(days=days)
    )

    filtered = []

    for commit in commits:

        date = _parse_date(
            commit.get("date")
        )

        if date and date >= cutoff:
            filtered.append(commit)

    return filtered


def get_last_updated_status(commits):
    """
    Determine repository activity status from the most recent
    commit date.

    Rules:
    < 7 days  -> Active
    < 30 days -> Moderately Active
    >= 30 days -> Inactive
    """

    if not commits:
        return {
            "last_commit_date": None,
            "days_since_last_commit": None,
            "status": "Unknown"
        }

    dated_commits = [
        (commit, _parse_date(commit.get("date")))
        for commit in commits
    ]

    dated_commits = [
        (commit, date)
        for commit, date in dated_commits
        if date is not None
    ]

    if not dated_commits:
        return {
            "last_commit_date": None,
            "days_since_last_commit": None,
            "status": "Unknown"
        }

    latest_commit, latest_date = max(
        dated_commits,
        key=lambda pair: pair[1]
    )

    days_since = (
        datetime.now(timezone.utc) - latest_date
    ).days

    if days_since < 7:
        status = "Active"
    elif days_since < 30:
        status = "Moderately Active"
    else:
        status = "Inactive"

    return {
        "last_commit_date": latest_commit.get("date"),
        "days_since_last_commit": days_since,
        "status": status
    }


# Keyword fast-path: catches the vast majority of activity
# questions without an extra LLM round-trip. The LLM fallback only
# runs when the keyword check is inconclusive.
_ACTIVITY_KEYWORDS = [
    "commit", "commits", "committed", "who committed",
    "recent change", "recent changes", "recently changed",
    "most active contributor", "active contributor",
    "contributor", "contributors",
    "last updated", "when was this", "when was the repo",
    "development activity", "recent development",
    "recent activity", "activity summary",
    "commit history", "changelog"
]


def is_activity_question(question, chat_history=""):
    """
    Decide whether a question is about repository ACTIVITY
    (commits, contributors, recent changes, last updated,
    development activity) as opposed to the repository's
    code/content (which should use ChromaDB retrieval) or general
    knowledge.
    """

    if not question:
        return False

    question_lower = question.lower()

    if any(
        keyword in question_lower
        for keyword in _ACTIVITY_KEYWORDS
    ):
        return True

    prompt = f"""
Conversation History:
{chat_history}

Question:
{question}

Is this question asking about the repository's commit history,
contributors, recent code changes, development activity, or when
it was last updated? Reply with exactly one word: YES or NO.
"""

    try:

        client = _get_activity_client()

        response = client.chat.complete(
            model=_ACTIVITY_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
            .upper()
        )

        return "YES" in answer

    except Exception:
        return False


def build_activity_context(commits, developer_activity=None, status=None):
    """
    Build a plain-text context block summarizing recent commit
    activity, fed into the main repository chat LLM call in place
    of ChromaDB retrieval results for activity-related questions.
    """

    if not commits:
        return (
            "No commit history is available for this repository."
        )

    lines = ["Recent Commit Activity:"]

    for commit in commits[:30]:

        date = (commit.get("date") or "")[:10]

        lines.append(
            f"- {commit.get('short_sha', commit.get('sha', ''))} | "
            f"{commit.get('author_name', 'Unknown')} | {date} | "
            f"{commit.get('message_title', '')}"
        )

    if developer_activity:

        lines.append("\nContributor Activity:")

        for dev in developer_activity[:10]:

            lines.append(
                f"- {dev['name']}: {dev['commits']} commits "
                f"(last: {(dev['last_commit_date'] or '')[:10]})"
            )

    if status:

        lines.append(
            f"\nRepository Status: {status['status']} "
            f"(last commit {status['days_since_last_commit']} "
            f"days ago)"
        )

    return "\n".join(lines)


def generate_activity_summary(commits, days=30):
    """
    Generate a professional AI summary of recent development
    activity, e.g.:

    "This repository has received 24 commits during the last 30
    days. Most changes were made by X. Development activity mainly
    focused on ..."
    """

    if not commits:
        raise GitHubActivityError(
            "No commit history available to summarize."
        )

    developer_activity = get_developer_activity(
        commits
    )

    status = get_last_updated_status(
        commits
    )

    context = build_activity_context(
        commits,
        developer_activity,
        status
    )

    prompt = f"""
You are a GitHub repository analyst.

Using the activity data below, write a concise, professional
summary (3-5 sentences) of development activity over the last
{days} days. Mention the total number of commits, the most active
contributor(s), and what the changes appear to have focused on.

{context}
"""

    try:

        client = _get_activity_client()

        response = client.chat.complete(
            model=_ACTIVITY_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        raise GitHubActivityError(
            f"Failed to generate activity summary: {e}"
        )