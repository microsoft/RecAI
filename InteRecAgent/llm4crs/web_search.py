# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Keyless web evidence over MCP Streamable HTTP, compatible with Python 3.9."""

import json
import os
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4


SEARCH_MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "RecAI-InteRecAgent/ParallelSearch (https://github.com/microsoft/RecAI)"
MAX_RESPONSE_BYTES = 1024 * 1024


def new_search_session():
    """Create an opaque identity for one conversation, never for a connection."""
    return str(uuid4())


class WebSearchError(Exception):
    """An unavailable web tool must not be mistaken for item evidence."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward optional credentials to a redirected endpoint.
        return None


class ParallelSearchClient:
    """Small synchronous MCP client for the two read-only Parallel web tools.

    The lock protects transport initialization, not conversation identity. Callers
    supply their conversation identity on every call, including retries.
    """

    def __init__(self, timeout=30, api_key=None):
        if timeout <= 0:
            raise ValueError("Web search timeout must be positive")
        self.timeout = timeout
        self.api_key = os.environ.get("PARALLEL_API_KEY", "") if api_key is None else api_key
        self._opener = build_opener(_NoRedirect())
        self._lock = threading.RLock()
        self._initialized = False
        self._transport_session = None
        self._protocol = "2025-03-26"

    def _request(self, method, params=None, notification=False):
        request_id = str(uuid4())
        payload = {"jsonrpc": "2.0", "method": method}
        if not notification:
            payload["id"] = request_id
        if params is not None:
            payload["params"] = params
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if self._initialized:
            headers["MCP-Protocol-Version"] = self._protocol
        if self._transport_session:
            headers["Mcp-Session-Id"] = self._transport_session
        request = Request(SEARCH_MCP_URL, data=json.dumps(payload).encode(), headers=headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if method == "initialize":
                    self._transport_session = response.headers.get("Mcp-Session-Id")
                if notification:
                    return None
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                if content_type == "text/event-stream":
                    size, data = 0, []
                    for raw_line in response:
                        size += len(raw_line)
                        if size > MAX_RESPONSE_BYTES:
                            raise WebSearchError("Web response exceeds the size limit")
                        line = raw_line.decode("utf-8").rstrip("\r\n")
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip(" "))
                        elif not line and data:
                            message = json.loads("\n".join(data))
                            data = []
                            if message.get("id") == request_id:
                                break
                    else:
                        raise WebSearchError("Web response ended without a matching MCP result")
                elif content_type == "application/json":
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise WebSearchError("Web response exceeds the size limit")
                    message = json.loads(raw)
                else:
                    raise WebSearchError("Unexpected web response content type")
        except HTTPError as exc:
            raise WebSearchError("Web service returned HTTP " + str(exc.code)) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise WebSearchError("Web service is unavailable or timed out") from exc
        except (ValueError, UnicodeError, AttributeError) as exc:
            raise WebSearchError("Invalid MCP web response") from exc
        if not isinstance(message, dict) or message.get("id") != request_id or message.get("jsonrpc") != "2.0":
            raise WebSearchError("Invalid MCP response identity")
        if "error" in message:
            raise WebSearchError("Web service returned an MCP error")
        if not isinstance(message.get("result"), dict):
            raise WebSearchError("Missing MCP web result")
        return message["result"]

    def call(self, tool_name, arguments, session_id):
        if not session_id:
            raise ValueError("Web tools require a conversation identity")
        with self._lock:
            if not self._initialized:
                self._transport_session = None
                result = self._request("initialize", {
                    "protocolVersion": self._protocol,
                    "capabilities": {},
                    "clientInfo": {"name": "RecAI-InteRecAgent", "version": "1"},
                })
                if result.get("protocolVersion") not in ("2025-03-26", "2025-06-18", "2025-11-25"):
                    raise WebSearchError("Unsupported MCP protocol version")
                self._protocol = result["protocolVersion"]
                self._initialized = True
                try:
                    self._request("notifications/initialized", notification=True)
                    tools = self._request("tools/list").get("tools", [])
                    if not isinstance(tools, list) or any(not isinstance(t, dict) or not isinstance(t.get("name"), str) for t in tools):
                        raise WebSearchError("Invalid MCP tool discovery response")
                    if not {"web_search", "web_fetch"}.issubset({t["name"] for t in tools}):
                        raise WebSearchError("Parallel web tools were not discovered")
                except BaseException:
                    self._initialized = False
                    raise
            result = self._request("tools/call", {
                "name": tool_name,
                "arguments": {**arguments, "session_id": session_id},
            })
        if result.get("isError"):
            raise WebSearchError("Web tool failed; no evidence is available. Retry later.")
        content = result.get("content", [])
        if not isinstance(content, list) or any(not isinstance(c, dict) for c in content):
            raise WebSearchError("Invalid MCP tool content")
        text = "\n".join(c["text"] for c in content
                         if c.get("type") == "text" and isinstance(c.get("text"), str))
        if not text:
            raise WebSearchError("Web tool returned no text evidence")
        return text


class ParallelWebTool:
    """An information tool, separate from the shared catalog candidate buffer."""

    external_evidence = True

    def __init__(self, domain, client, fetch=False):
        self.client = client
        self.fetch = fetch
        self.name = domain.capitalize() + (" Web Fetch Tool" if fetch else " Web Search Tool")
        self.desc = (
            "Fetch public HTTP(S) pages for current item information. Input is a JSON string "
            "with urls (one to twenty URLs), optional objective (up to 200 characters) and "
            "optional search_queries (keyword strings)."
            if fetch else
            "Find current item information missing from the catalog, such as game updates or "
            "official documentation. Input is a JSON string with objective (a focused goal) "
            "and search_queries (a nonempty list of short keyword strings)."
        ) + " Returns web evidence and source URLs, never catalog candidates. Do not send user profiles or private data."

    def run(self, inputs, session_id):
        arguments = json.loads(inputs)
        allowed = {"urls", "objective", "search_queries"} if self.fetch else {"objective", "search_queries"}
        if not isinstance(arguments, dict) or set(arguments) - allowed:
            raise ValueError("Unexpected web tool fields")
        queries = arguments.get("search_queries")
        if queries is not None and (not isinstance(queries, list) or not queries or
                                    any(not isinstance(q, str) or not q.strip() for q in queries)):
            raise ValueError("search_queries must contain nonempty strings")
        objective = arguments.get("objective")
        if objective is not None and (not isinstance(objective, str) or not objective.strip()):
            raise ValueError("objective must be nonempty text")
        if self.fetch:
            urls = arguments.get("urls")
            if not isinstance(urls, list) or not 1 <= len(urls) <= 20:
                raise ValueError("urls must contain one to twenty public URLs")
            for url in urls:
                if not isinstance(url, str):
                    raise ValueError("Invalid URL")
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("Only HTTP(S) URLs without credentials are supported")
            if objective is not None and len(objective) > 200:
                raise ValueError("Fetch objective exceeds 200 characters")
        elif objective is None or queries is None:
            raise ValueError("Search requires objective and search_queries")
        return self.client.call("web_fetch" if self.fetch else "web_search", arguments, session_id)


def resolve_web_tools(tools, domain, provider="parallel", client=None):
    """Preserve supplied tools; disabled means no built-in web tools or requests."""
    result = dict(tools)
    if provider == "disabled":
        result.pop("WebSearchTool", None)
        result.pop("WebFetchTool", None)
    elif provider == "parallel":
        client = client if client is not None else ParallelSearchClient()
        result.setdefault("WebSearchTool", ParallelWebTool(domain, client))
        result.setdefault("WebFetchTool", ParallelWebTool(domain, client, fetch=True))
    else:
        raise ValueError("Unknown web search provider")
    return result
