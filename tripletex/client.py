"""
Thin HTTP client for the Tripletex v2 API.

All tools receive a TripletexClient instance — auth and base URL are
handled here so no tool needs to know about credentials.
"""

import requests


class TripletexError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        # Strip trailing slash for clean path concatenation
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _raise_for_status(self, resp: requests.Response) -> None:
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise TripletexError(resp.status_code, str(detail))

    def get(self, path: str, **kwargs) -> dict:
        resp = requests.get(self._url(path), auth=self.auth, **kwargs)
        self._raise_for_status(resp)
        return resp.json()

    def post(self, path: str, **kwargs) -> dict:
        resp = requests.post(self._url(path), auth=self.auth, **kwargs)
        self._raise_for_status(resp)
        return resp.json()

    def put(self, path: str, **kwargs) -> dict:
        resp = requests.put(self._url(path), auth=self.auth, **kwargs)
        self._raise_for_status(resp)
        return resp.json()

    def delete(self, path: str, **kwargs) -> None:
        resp = requests.delete(self._url(path), auth=self.auth, **kwargs)
        self._raise_for_status(resp)
