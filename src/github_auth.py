import os
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = os.getenv(
    "GITHUB_REDIRECT_URI",
    "http://localhost:8501"
)

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API_URL = "https://api.github.com/user"

# "repo" grants access to public AND private repositories.
# "read:user" grants access to profile fields (name, followers, etc).
GITHUB_SCOPES = "read:user repo"


def get_login_url():
    """
    Build the GitHub OAuth authorization URL. Redirecting (or
    linking) the user here starts the login flow: GitHub will show
    its authorization screen and, once approved, redirect back to
    GITHUB_REDIRECT_URI with a `code` query parameter.
    """

    if not GITHUB_CLIENT_ID:
        raise RuntimeError(
            "GITHUB_CLIENT_ID is not set. Add it to your .env file."
        )

    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": GITHUB_SCOPES,
        "allow_signup": "true"
    }

    return f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_token(code):
    """
    Exchange the temporary `code` GitHub redirected back with for a
    long-lived access token.
    """

    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise RuntimeError(
            "GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET are not set. "
            "Add them to your .env file."
        )

    response = requests.post(
        GITHUB_TOKEN_URL,
        headers={
            "Accept": "application/json"
        },
        data={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
            "redirect_uri": GITHUB_REDIRECT_URI
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            data.get(
                "error_description",
                data["error"]
            )
        )

    access_token = data.get("access_token")

    if not access_token:
        raise RuntimeError(
            "GitHub did not return an access token."
        )

    return access_token


def get_user_profile(access_token):
    """
    Fetch the authenticated user's GitHub profile using their
    access token.

    Returns a dict with username, name, avatar, followers,
    public_repos, and access_token.
    """

    response = requests.get(
        GITHUB_USER_API_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json"
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return {
        "username": data.get("login"),
        "name": data.get("name") or data.get("login"),
        "avatar": data.get("avatar_url"),
        "followers": data.get("followers", 0),
        "public_repos": data.get("public_repos", 0),
        "profile_url": data.get("html_url"),
        "access_token": access_token
    }