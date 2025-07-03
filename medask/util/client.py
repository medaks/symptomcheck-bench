"""
Utility functions for making http requests.
"""

import json
import requests
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from requests.models import Response

_logger = getLogger("medask.util.client")

# Using self-signed certs in VM.
requests.packages.urllib3.disable_warnings()

_session = requests.session()


def _decode(resp: "Response") -> Dict[str, Any]:
    """Unmarshal the response object content."""
    content = resp.content.decode("utf-8")
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed. {resp.status_code} - {content}")
    return json.loads(content)


def get(
    path: str, url: str, timeout: Optional[float] = None, params: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """Make GET request to the server."""
    resp = _session.get(
        f"{url}/{path}",
        timeout=timeout,
        verify=False,
        params=params,
    )
    return _decode(resp)


def post(path: str, body: str, url: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Make POST request to the server."""
    resp = _session.post(
        f"{url}/{path}",
        body.encode("utf-8"),
        timeout=timeout,
        verify=False,
    )
    return _decode(resp)
