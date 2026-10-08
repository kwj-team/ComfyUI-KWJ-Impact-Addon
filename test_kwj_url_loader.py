import io
import socket
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: "/tmp"
    sys.modules["folder_paths"] = folder_paths

if "PIL" not in sys.modules:
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace()
    pil.ImageOps = types.SimpleNamespace()
    pil.ImageSequence = types.SimpleNamespace()
    pil.UnidentifiedImageError = type("UnidentifiedImageError", (Exception,), {})
    sys.modules["PIL"] = pil

import kwj_url_loader


def _response(payload: bytes):
    response = mock.MagicMock()
    response.read.return_value = payload
    response.__enter__.return_value = response
    return response


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://childbook-b2.b-cdn.net/a.webp",
        code,
        "error",
        hdrs=None,
        fp=io.BytesIO(b""),
    )


class DownloadRetryTests(unittest.TestCase):
    def test_connection_error_retries_and_resolves_again(self):
        payload = b"ok-image"
        server = _listening_server(payload)
        refused_port = _closed_port()
        lookups = []

        def fake_getaddrinfo(host, port, *args, **kwargs):
            if host != "cdn.test":
                return _real_getaddrinfo(host, port, *args, **kwargs)
            lookups.append((host, port))
            chosen = refused_port if len(lookups) == 1 else server.server_address[1]
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", chosen),
                )
            ]

        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            kwj_url_loader._KWJHTTPHandler,
        )
        try:
            with (
                mock.patch("socket.getaddrinfo", side_effect=fake_getaddrinfo),
                mock.patch.object(kwj_url_loader, "_URL_OPENER", opener),
                mock.patch.object(kwj_url_loader.time, "sleep") as sleep,
            ):
                data = kwj_url_loader._download_url("http://cdn.test/photo.webp")
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(data, payload)
        self.assertEqual(lookups, [("cdn.test", 80), ("cdn.test", 80)])
        sleep.assert_called_once_with(1)

    def test_connection_errors_stop_after_the_retry_budget(self):
        with (
            mock.patch.object(
                kwj_url_loader,
                "_open_url",
                side_effect=urllib.error.URLError(TimeoutError("handshake timed out")),
            ) as open_url,
            mock.patch.object(kwj_url_loader.time, "sleep") as sleep,
        ):
            with self.assertRaises(urllib.error.URLError):
                kwj_url_loader._download_url("https://childbook-b2.b-cdn.net/a.webp")

        self.assertEqual(open_url.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 3])
        hosts = [call.args[0].host for call in open_url.call_args_list]
        self.assertEqual(hosts, ["childbook-b2.b-cdn.net"] * 4)

    def test_http_502_is_retried(self):
        with (
            mock.patch.object(
                kwj_url_loader,
                "_open_url",
                side_effect=[_http_error(502), _response(b"img")],
            ) as open_url,
            mock.patch.object(kwj_url_loader.time, "sleep") as sleep,
        ):
            data = kwj_url_loader._download_url("https://childbook-b2.b-cdn.net/a.webp")

        self.assertEqual(data, b"img")
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_other_http_errors_are_not_retried(self):
        with mock.patch.object(
            kwj_url_loader,
            "_open_url",
            side_effect=_http_error(404),
        ) as open_url:
            with self.assertRaises(urllib.error.HTTPError):
                kwj_url_loader._download_url("https://images.childbook.ai/a.webp")

        self.assertEqual(open_url.call_count, 1)

    def test_read_timeout_is_not_retried(self):
        response = _response(b"")
        response.read.side_effect = TimeoutError("timed out")
        with mock.patch.object(kwj_url_loader, "_open_url", return_value=response) as open_url:
            with self.assertRaises(TimeoutError):
                kwj_url_loader._download_url("https://images.childbook.ai/a.webp")

        self.assertEqual(open_url.call_count, 1)

    def test_dropped_connection_while_reading_is_retried(self):
        failed = _response(b"")
        failed.read.side_effect = ConnectionResetError("reset")
        with (
            mock.patch.object(
                kwj_url_loader,
                "_open_url",
                side_effect=[failed, _response(b"img")],
            ) as open_url,
            mock.patch.object(kwj_url_loader.time, "sleep"),
        ):
            data = kwj_url_loader._download_url("https://images.childbook.ai/a.webp")

        self.assertEqual(data, b"img")
        self.assertEqual(open_url.call_count, 2)


class ConnectTimeoutTests(unittest.TestCase):
    def test_timeouts_are_split_between_handshake_and_read(self):
        self.assertEqual(kwj_url_loader._CONNECT_TIMEOUT_SECONDS, 10)
        self.assertEqual(kwj_url_loader._READ_TIMEOUT_SECONDS, 120)

    def test_https_handshake_uses_connect_timeout(self):
        conn = kwj_url_loader._KWJHTTPSConnection("childbook-b2.b-cdn.net", timeout=10)
        raw = mock.Mock()
        wrapped = mock.Mock()
        events = []

        def wrap_socket(sock, server_hostname):
            events.append(("wrap", sock, server_hostname))
            return wrapped

        wrapped.settimeout.side_effect = lambda value: events.append(("read_timeout", value))
        conn._create_connection = mock.Mock(return_value=raw)
        conn._context = mock.Mock()
        conn._context.wrap_socket.side_effect = wrap_socket

        conn.connect()

        address, timeout, _source = conn._create_connection.call_args.args
        self.assertEqual(address, ("childbook-b2.b-cdn.net", 443))
        self.assertEqual(timeout, 10)
        self.assertEqual(
            events,
            [("wrap", raw, "childbook-b2.b-cdn.net"), ("read_timeout", 120)],
        )

    def test_http_connect_uses_connect_timeout_then_read_timeout(self):
        conn = kwj_url_loader._KWJHTTPConnection("images.childbook.ai", timeout=10)
        raw = mock.Mock()
        conn._create_connection = mock.Mock(return_value=raw)

        conn.connect()

        address, timeout, _source = conn._create_connection.call_args.args
        self.assertEqual(address, ("images.childbook.ai", 80))
        self.assertEqual(timeout, 10)
        raw.settimeout.assert_called_once_with(120)


_real_getaddrinfo = socket.getaddrinfo


def _closed_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _listening_server(payload: bytes) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    unittest.main()
