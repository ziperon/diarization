import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class DionApiError(Exception):
    def __init__(self, status_code: int, message: str, response: Optional[Response] = None):
        super().__init__(f"DION API error {status_code}: {message}")
        self.status_code = status_code
        self.response = response


class DionApiClient:
    """
    Reusable client for DION IAPI (https://faq.dion.vc/ru/api).
    - Auth: X-Client-Access-Token header
    - Transport: HTTPS with optional mTLS (client certificate)
    - Built-in retries for transient errors
    - Simple rate limiting: 100 req / 60s, max 10 concurrent
    """

    def __init__(
        self,
        access_token: str,
        base_url: str = "https://api-integration.dion.vc/v1",
        timeout_seconds: int = 30,
        client_cert: Optional[Union[str, Tuple[str, str]]] = None,
        verify_ssl: Union[bool, str] = True,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        per_minute_limit: int = 100,
        concurrent_limit: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.client_cert = client_cert
        self.verify_ssl = verify_ssl

        self.session: Session = requests.Session()
        self.session.headers.update({"X-Client-Access-Token": access_token})

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=concurrent_limit, pool_maxsize=concurrent_limit)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Concurrency limiter
        self._concurrent_semaphore = threading.Semaphore(concurrent_limit)

        # Simple token bucket for per-minute limit
        self._lock = threading.Lock()
        self._window_seconds = 60
        self._max_in_window = per_minute_limit
        self._calls_window: deque[float] = deque()

    # --------------- Internal helpers ---------------
    def _rate_limit_blocking(self) -> None:
        """
        Blocks until a slot is available per 100 requests/60s. Thread-safe.
        """
        while True:
            with self._lock:
                now = time.time()
                # Drop timestamps older than window
                while self._calls_window and now - self._calls_window[0] > self._window_seconds:
                    self._calls_window.popleft()
                if len(self._calls_window) < self._max_in_window:
                    self._calls_window.append(now)
                    return
                # Calculate sleep until earliest timestamp leaves the window
                sleep_for = self._window_seconds - (now - self._calls_window[0])
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.25))
            else:
                # Just loop again; another thread may have freed space
                time.sleep(0.01)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self._rate_limit_blocking()
        with self._concurrent_semaphore:
            resp = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                timeout=self.timeout,
                cert=self.client_cert,
                verify=self.verify_ssl,
                stream=stream,
            )
        if 200 <= resp.status_code < 300:
            return resp
        # Surface API errors with content if available
        message = ""
        try:
            message = resp.text or ""
        except Exception:
            message = ""
        raise DionApiError(resp.status_code, message, resp)

    # --------------- Public API methods ---------------
    # Technical: GET /events/params
    def get_events_params(self) -> List[Dict[str, Any]]:
        resp = self._request("GET", "events/params")
        return resp.json()

    def get_event_data_by_id(self, event_id: str) -> List[Dict[str, Any]]:
        resp = self._request("GET", f"events/{event_id}")
        return resp.json()

    def get_event_users(
        self,
        event_id: str,
        *,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получить список пользователей события за период.

        Endpoint:
            GET /events/{event_id}/users?time_start=...&time_end=...

        Args:
            event_id: UUID события
            time_start: ISO8601 (например '2025-11-10T11:00:00Z')
            time_end: ISO8601 (например '2025-11-10T15:00:00Z')

        Returns:
            Список пользователей с данными их участия.
        """
        params: Dict[str, Any] = {}

        if time_start is not None:
            params["time_start"] = time_start

        if time_end is not None:
            params["time_end"] = time_end

        path = f"events/{event_id}/users"
        resp = self._request("GET", path, params=params)
        return resp.json()

    # Conferences: POST /events
    def create_event(
        self,
        *,
        owner_email: str,
        title: Optional[str] = None,
        event_params: Optional[List[str]] = None,
        moderators_emails: Optional[List[str]] = None,
        use_org_servers: Optional[bool] = None,
        personalized_room_name: Optional[str] = None,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"owner_email": owner_email}
        if title is not None:
            payload["title"] = title
        if event_params:
            payload["event_params"] = event_params
        if moderators_emails:
            payload["moderators_emails"] = moderators_emails
        if use_org_servers is not None:
            payload["use_org_servers"] = use_org_servers
        if personalized_room_name is not None:
            payload["personalized_room_name"] = personalized_room_name
        if start_at is not None:
            payload["start_at"] = start_at
        if end_at is not None:
            payload["end_at"] = end_at
        if extra_payload:
            payload.update(extra_payload)

        resp = self._request("POST", "events", json_body=payload)
        return resp.json()

    # Video: GET /video/records/{record_id}/summarization → returns file
    def get_record_summarization(self, record_id: str) -> bytes:
        path = f"video/records/{record_id}/summarization"
        resp = self._request("GET", path, stream=True)
        # Return raw bytes (caller decides how to persist)
        return resp.content

    # Audit: GET /audits
    def get_audits(
        self,
        *,
        limit: Optional[int] = None,
        last_audit_id: Optional[str] = None,
        message_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if last_audit_id is not None:
            params["last_audit_id"] = last_audit_id
        if message_types:
            # API expects message_type[] as repeated query params
            for idx, value in enumerate(message_types):
                params[f"message_type[{idx}]"] = value
        resp = self._request("GET", "audits", params=params)
        return resp.json()

    # Audit: GET /audits/types
    def get_audit_types(self) -> Dict[str, Any]:
        resp = self._request("GET", "audits/types")
        return resp.json()

    # Audit: GET /audits/platforms
    def get_audit_platforms(self) -> Dict[str, Any]:
        resp = self._request("GET", "audits/platforms")
        return resp.json()

    # Users: GET /user/:user_id
    def get_user_by_id(self, user_id: str) -> Dict[str, Any]:
        """
        Получить информацию о пользователе по ID.
        Endpoint: GET api-integration.dion.vc/v1/user/:user_id
        
        Args:
            user_id: UUID пользователя
            
        Returns:
            Словарь с информацией о пользователе (email, name, и т.д.)
        """
        path = f"user/{user_id}"
        resp = self._request("GET", path)
        return resp.json()

    # --------------- Convenience ---------------
    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def __enter__(self) -> "DionApiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


