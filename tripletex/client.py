"""
Thin HTTP client for the Tripletex v2 API.

All tools receive a TripletexClient instance — auth and base URL are
handled here so no tool needs to know about credentials.
"""

import json
import logging

import requests

logger = logging.getLogger(__name__)


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

    def _log_request(self, method: str, path: str, kwargs: dict) -> None:
        body = kwargs.get("json")
        body_str = json.dumps(body, ensure_ascii=False) if body else ""
        logger.info("→ %s %s %s", method, path, body_str)

    def _log_response(self, resp: requests.Response) -> None:
        try:
            body = resp.json()
            # For large list responses, just log count
            if isinstance(body, dict) and "values" in body:
                logger.info("← %s (count=%s)", resp.status_code, len(body["values"]))
            else:
                logger.debug("← %s %s", resp.status_code, json.dumps(body, ensure_ascii=False)[:300])
        except Exception:
            logger.debug("← %s %s", resp.status_code, resp.text[:200])

    def get(self, path: str, **kwargs) -> dict:
        logger.debug("→ GET %s params=%s", path, kwargs.get("params"))
        resp = requests.get(self._url(path), auth=self.auth, **kwargs)
        self._raise_for_status(resp)
        return resp.json()

    def post(self, path: str, **kwargs) -> dict:
        self._log_request("POST", path, kwargs)
        resp = requests.post(self._url(path), auth=self.auth, **kwargs)
        self._log_response(resp)
        self._raise_for_status(resp)
        return resp.json() if resp.content else {"ok": True, "status": resp.status_code}

    def put(self, path: str, **kwargs) -> dict:
        self._log_request("PUT", path, kwargs)
        resp = requests.put(self._url(path), auth=self.auth, **kwargs)
        self._log_response(resp)
        self._raise_for_status(resp)
        return resp.json() if resp.content else {"ok": True, "status": resp.status_code}

    def delete(self, path: str, **kwargs) -> None:
        logger.info("→ DELETE %s", path)
        resp = requests.delete(self._url(path), auth=self.auth, **kwargs)
        logger.info("← %s", resp.status_code)
        self._raise_for_status(resp)
