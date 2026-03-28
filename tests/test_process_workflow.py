"""
Tests for the async workflow processing pipeline:
  - _monitor_execution (WebSocket monitoring)
  - _wait_for_comfyui (startup polling)
  - process_workflow (full end-to-end)
"""

import os
import sys
import json
import base64
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app
from app import GenerateRequest


# ---------------------------------------------------------------------------
# Helper for mocking websockets.connect as an async context manager
# ---------------------------------------------------------------------------

class MockWebSocket:
    """A mock WebSocket that yields pre-configured messages."""

    def __init__(self, messages):
        self._messages = messages
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._index]
        self._index += 1
        if isinstance(msg, str):
            return msg
        if isinstance(msg, bytes):
            return msg
        return json.dumps(msg)


class MockWebSocketConnect:
    """Mock for websockets.connect() as an async context manager."""

    def __init__(self, messages):
        self.ws = MockWebSocket(messages)

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *args):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# _monitor_execution
# ═══════════════════════════════════════════════════════════════════════════════


class TestMonitorExecution(unittest.TestCase):
    """Tests for _monitor_execution() async WebSocket monitoring."""

    def test_execution_complete(self):
        """Normal flow: executing messages, then node=None signals completion."""

        async def _run():
            prompt_id = "prompt-123"
            messages = [
                {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}},
                {"type": "executing", "data": {"node": "6", "prompt_id": prompt_id}},
                {"type": "executing", "data": {"node": "9", "prompt_id": prompt_id}},
                {"type": "executing", "data": {"node": None, "prompt_id": prompt_id}},
            ]

            with patch(
                "app.websockets.connect",
                return_value=MockWebSocketConnect(messages),
            ):
                done, errors = await app._monitor_execution(prompt_id, "client-1")

            self.assertTrue(done)
            self.assertEqual(errors, [])

        asyncio.run(_run())

    def test_execution_error(self):
        """WebSocket receives an execution_error message."""

        async def _run():
            prompt_id = "prompt-456"
            messages = [
                {
                    "type": "execution_error",
                    "data": {
                        "prompt_id": prompt_id,
                        "node_type": "KSampler",
                        "node_id": "3",
                        "exception_message": "OOM",
                    },
                },
            ]

            with patch(
                "app.websockets.connect",
                return_value=MockWebSocketConnect(messages),
            ):
                done, errors = await app._monitor_execution(prompt_id, "client-1")

            self.assertFalse(done)
            self.assertEqual(len(errors), 1)
            self.assertIn("OOM", errors[0])

        asyncio.run(_run())

    def test_ignores_other_prompt_ids(self):
        """Messages for different prompt_ids should be ignored."""

        async def _run():
            my_prompt = "mine"
            messages = [
                {"type": "executing", "data": {"node": None, "prompt_id": "other"}},
                {"type": "executing", "data": {"node": "6", "prompt_id": my_prompt}},
                {"type": "executing", "data": {"node": None, "prompt_id": my_prompt}},
            ]

            with patch(
                "app.websockets.connect",
                return_value=MockWebSocketConnect(messages),
            ):
                done, errors = await app._monitor_execution(my_prompt, "client-1")

            self.assertTrue(done)

        asyncio.run(_run())

    def test_ignores_binary_messages(self):
        """Binary WebSocket frames should be skipped."""

        async def _run():
            prompt_id = "prompt-bin"
            # Mix of binary (raw bytes) and JSON completion message
            messages = [
                b"\x00\x01\x02",
                {"type": "executing", "data": {"node": None, "prompt_id": prompt_id}},
            ]

            with patch(
                "app.websockets.connect",
                return_value=MockWebSocketConnect(messages),
            ):
                done, errors = await app._monitor_execution(prompt_id, "client-1")

            self.assertTrue(done)

        asyncio.run(_run())

    def test_ignores_invalid_json(self):
        """Invalid JSON frames should be skipped without crashing."""

        async def _run():
            prompt_id = "prompt-json"
            messages = [
                "not valid json {",  # raw string, not dict
                {"type": "executing", "data": {"node": None, "prompt_id": prompt_id}},
            ]

            # MockWebSocket passes strings as-is, dicts get json.dumps'd
            ws_messages = []
            for m in messages:
                if isinstance(m, str):
                    ws_messages.append(m)
                else:
                    ws_messages.append(json.dumps(m))

            mock_ws = MockWebSocket.__new__(MockWebSocket)
            mock_ws._messages = ws_messages
            mock_ws._index = 0

            class _Connect:
                async def __aenter__(self):
                    return mock_ws

                async def __aexit__(self, *a):
                    pass

            with patch("app.websockets.connect", return_value=_Connect()):
                done, errors = await app._monitor_execution(prompt_id, "client-1")

            self.assertTrue(done)

        asyncio.run(_run())

    def test_reconnect_on_connection_closed_server_unreachable(self):
        """When WS closes and ComfyUI HTTP is down, raise ConnectionError immediately."""

        async def _run():
            import websockets

            with patch(
                "app.websockets.connect",
                side_effect=websockets.ConnectionClosed(None, None),
            ):
                with patch(
                    "app._comfy_server_status",
                    return_value={"reachable": False, "error": "refused"},
                ):
                    with self.assertRaises(ConnectionError):
                        await app._monitor_execution("prompt-x", "client-1")

        asyncio.run(_run())

    def test_reconnect_exhausted_raises(self):
        """After max reconnect attempts, raise ConnectionError."""

        async def _run():
            import websockets

            with patch("app.WEBSOCKET_RECONNECT_ATTEMPTS", 2):
                with patch("app.WEBSOCKET_RECONNECT_DELAY_S", 0):
                    with patch(
                        "app.websockets.connect",
                        side_effect=websockets.ConnectionClosed(None, None),
                    ):
                        with patch(
                            "app._comfy_server_status",
                            return_value={"reachable": True, "status_code": 200},
                        ):
                            with self.assertRaises(ConnectionError):
                                await app._monitor_execution("prompt-x", "client-1")

        asyncio.run(_run())

    def test_ignores_execution_error_for_other_prompt(self):
        """execution_error for a different prompt_id should be ignored."""

        async def _run():
            my_prompt = "mine"
            messages = [
                {
                    "type": "execution_error",
                    "data": {
                        "prompt_id": "not-mine",
                        "node_type": "X",
                        "node_id": "1",
                        "exception_message": "err",
                    },
                },
                {"type": "executing", "data": {"node": None, "prompt_id": my_prompt}},
            ]

            with patch(
                "app.websockets.connect",
                return_value=MockWebSocketConnect(messages),
            ):
                done, errors = await app._monitor_execution(my_prompt, "client-1")

            self.assertTrue(done)
            self.assertEqual(errors, [])

        asyncio.run(_run())


# ═══════════════════════════════════════════════════════════════════════════════
# _wait_for_comfyui
# ═══════════════════════════════════════════════════════════════════════════════


class TestWaitForComfyUI(unittest.TestCase):
    """Tests for _wait_for_comfyui() startup polling."""

    def test_sets_ready_when_server_responds_200(self):
        async def _run():
            mock_app = MagicMock()
            mock_app.state = MagicMock()
            mock_app.state.comfyui_ready = False

            with patch("app.requests.get") as mock_get:
                mock_get.return_value = MagicMock(status_code=200)
                with patch("app._is_comfyui_process_alive", return_value=True):
                    with patch("app.is_network_volume_debug_enabled", return_value=False):
                        await app._wait_for_comfyui(mock_app)

            self.assertTrue(mock_app.state.comfyui_ready)

        asyncio.run(_run())

    def test_stays_unhealthy_when_process_exits(self):
        async def _run():
            mock_app = MagicMock()
            mock_app.state = MagicMock()
            mock_app.state.comfyui_ready = False

            with patch("app._is_comfyui_process_alive", return_value=False):
                with patch("app.is_network_volume_debug_enabled", return_value=False):
                    await app._wait_for_comfyui(mock_app)

            self.assertFalse(mock_app.state.comfyui_ready)

        asyncio.run(_run())

    def test_retries_on_connection_error(self):
        async def _run():
            mock_app = MagicMock()
            mock_app.state = MagicMock()
            mock_app.state.comfyui_ready = False

            import requests as req

            call_count = 0

            def mock_get_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise req.ConnectionError("refused")
                return MagicMock(status_code=200)

            with patch("app.requests.get", side_effect=mock_get_side_effect):
                with patch("app._is_comfyui_process_alive", return_value=True):
                    with patch("app.is_network_volume_debug_enabled", return_value=False):
                        with patch("app.COMFY_API_AVAILABLE_INTERVAL_MS", 1):
                            await app._wait_for_comfyui(mock_app)

            self.assertTrue(mock_app.state.comfyui_ready)
            self.assertEqual(call_count, 3)

        asyncio.run(_run())

    def test_gives_up_after_fallback_retries_when_no_pid(self):
        async def _run():
            mock_app = MagicMock()
            mock_app.state = MagicMock()
            mock_app.state.comfyui_ready = False

            import requests as req

            with patch("app.requests.get", side_effect=req.ConnectionError("refused")):
                with patch("app._is_comfyui_process_alive", return_value=None):
                    with patch("app.is_network_volume_debug_enabled", return_value=False):
                        with patch("app.COMFY_API_AVAILABLE_INTERVAL_MS", 1):
                            with patch("app.COMFY_API_AVAILABLE_MAX_RETRIES", 0):
                                with patch("app.COMFY_API_FALLBACK_MAX_RETRIES", 3):
                                    await app._wait_for_comfyui(mock_app)

            self.assertFalse(mock_app.state.comfyui_ready)

        asyncio.run(_run())


# ═══════════════════════════════════════════════════════════════════════════════
# process_workflow — full pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcessWorkflow(unittest.TestCase):
    """Tests for process_workflow() end-to-end async processing."""

    def _make_request(self, workflow=None, images=None, comfy_org_api_key=None):
        return GenerateRequest(
            workflow=workflow or {"6": {"class_type": "CLIPTextEncode"}},
            images=images,
            comfy_org_api_key=comfy_org_api_key,
        )

    def test_successful_base64_output(self):
        """Full success path: queue -> monitor -> history -> base64 image."""

        async def _run():
            prompt_id = "prompt-ok"
            image_bytes = b"\x89PNG fake image data"

            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": prompt_id}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            prompt_id: {
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {
                                                "filename": "ComfyUI_00001_.png",
                                                "subfolder": "",
                                                "type": "output",
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                        with patch("app.get_image_data") as mock_img:
                            mock_img.return_value = image_bytes

                            result = await app.process_workflow(
                                self._make_request(), "req-1"
                            )

            self.assertIn("images", result)
            self.assertEqual(len(result["images"]), 1)
            self.assertEqual(result["images"][0]["type"], "base64")
            self.assertEqual(result["images"][0]["filename"], "ComfyUI_00001_.png")
            expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
            self.assertEqual(result["images"][0]["data"], expected_b64)

        asyncio.run(_run())

    @patch.dict(
        os.environ,
        {
            "BUCKET_ENDPOINT_URL": "https://mybucket.s3.amazonaws.com",
            "BUCKET_ACCESS_KEY_ID": "KEY",
            "BUCKET_SECRET_ACCESS_KEY": "SECRET",
        },
    )
    def test_successful_s3_output(self):
        """Success path with S3 upload."""

        async def _run():
            prompt_id = "prompt-s3"

            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": prompt_id}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            prompt_id: {
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {
                                                "filename": "out.png",
                                                "subfolder": "",
                                                "type": "output",
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                        with patch("app.get_image_data") as mock_img:
                            mock_img.return_value = b"image_bytes"
                            with patch("app.upload_to_s3") as mock_s3:
                                mock_s3.return_value = (
                                    "https://mybucket.s3.amazonaws.com/req-s3/out.png"
                                )

                                result = await app.process_workflow(
                                    self._make_request(), "req-s3"
                                )

            self.assertIn("images", result)
            self.assertEqual(result["images"][0]["type"], "s3_url")
            self.assertIn("mybucket", result["images"][0]["data"])

        asyncio.run(_run())

    def test_upload_images_called_when_provided(self):
        """Input images should be uploaded before queueing workflow."""

        async def _run():
            from app import ImageInput

            request = self._make_request(
                images=[ImageInput(name="input.png", image="base64data")]
            )

            with patch("app.upload_images") as mock_upload:
                mock_upload.return_value = {
                    "status": "success",
                    "message": "ok",
                    "details": [],
                }
                with patch("app.queue_workflow") as mock_queue:
                    mock_queue.return_value = {"prompt_id": "p1"}
                    with patch(
                        "app._monitor_execution", new_callable=AsyncMock
                    ) as mock_monitor:
                        mock_monitor.return_value = (True, [])
                        with patch("app.get_history") as mock_history:
                            mock_history.return_value = {
                                "p1": {"outputs": {}}
                            }

                            result = await app.process_workflow(request, "req-img")

            mock_upload.assert_called_once()
            upload_args = mock_upload.call_args[0][0]
            self.assertEqual(len(upload_args), 1)
            self.assertEqual(upload_args[0]["name"], "input.png")

        asyncio.run(_run())

    def test_image_upload_failure_returns_error(self):
        """If image upload fails, return error immediately."""

        async def _run():
            from app import ImageInput

            request = self._make_request(
                images=[ImageInput(name="bad.png", image="data")]
            )

            with patch("app.upload_images") as mock_upload:
                mock_upload.return_value = {
                    "status": "error",
                    "message": "Upload failed",
                    "details": ["Error uploading bad.png"],
                }

                result = await app.process_workflow(request, "req-fail")

            self.assertIn("error", result)
            self.assertIn("upload", result["error"].lower())

        asyncio.run(_run())

    def test_queue_workflow_failure(self):
        """ValueError from queue_workflow should return error."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.side_effect = ValueError("Workflow validation failed")

                result = await app.process_workflow(self._make_request(), "req-qerr")

            self.assertIn("error", result)
            self.assertIn("validation", result["error"].lower())

        asyncio.run(_run())

    def test_queue_workflow_http_error(self):
        """RequestException from queue_workflow should return error."""

        async def _run():
            import requests as req

            with patch("app.queue_workflow") as mock_queue:
                mock_queue.side_effect = req.ConnectionError("refused")

                result = await app.process_workflow(self._make_request(), "req-http")

            self.assertIn("error", result)

        asyncio.run(_run())

    def test_missing_prompt_id_in_queue_response(self):
        """If queue_workflow returns no prompt_id, return error."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"number": 1}  # no prompt_id

                result = await app.process_workflow(self._make_request(), "req-noid")

            self.assertIn("error", result)
            self.assertIn("prompt_id", result["error"].lower())

        asyncio.run(_run())

    def test_websocket_error_returns_error(self):
        """ConnectionError from _monitor_execution should be caught."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.side_effect = ConnectionError("WS died")

                    result = await app.process_workflow(self._make_request(), "req-ws")

            self.assertIn("error", result)
            self.assertIn("WebSocket", result["error"])

        asyncio.run(_run())

    def test_prompt_not_in_history(self):
        """If prompt_id not found in history, return error."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {}  # p1 not in history

                        result = await app.process_workflow(
                            self._make_request(), "req-nohist"
                        )

            self.assertIn("error", result)
            self.assertIn("not found", result["error"].lower())

        asyncio.run(_run())

    def test_no_outputs_returns_error(self):
        """Workflow with empty outputs returns error (no images produced)."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {"p1": {"outputs": {}}}

                        result = await app.process_workflow(
                            self._make_request(), "req-empty"
                        )

            # Empty outputs triggers a "No outputs found" warning, which
            # combined with no images means the job is treated as failed
            self.assertIn("error", result)

        asyncio.run(_run())

    def test_outputs_with_only_non_image_keys_returns_success_no_images(self):
        """Workflow outputs with no 'images' key returns success_no_images."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            "p1": {"outputs": {"9": {"text": ["hello"]}}}
                        }

                        result = await app.process_workflow(
                            self._make_request(), "req-noimg"
                        )

            self.assertEqual(result.get("status"), "success_no_images")
            self.assertEqual(result["images"], [])

        asyncio.run(_run())

    def test_skips_temp_images(self):
        """Images with type='temp' should be skipped."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            "p1": {
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {
                                                "filename": "temp.png",
                                                "subfolder": "",
                                                "type": "temp",
                                            }
                                        ]
                                    }
                                }
                            }
                        }

                        result = await app.process_workflow(
                            self._make_request(), "req-temp"
                        )

            self.assertEqual(result.get("status"), "success_no_images")

        asyncio.run(_run())

    def test_image_fetch_failure_adds_error(self):
        """If get_image_data returns None, an error should be added."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            "p1": {
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {
                                                "filename": "broken.png",
                                                "subfolder": "",
                                                "type": "output",
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                        with patch("app.get_image_data") as mock_img:
                            mock_img.return_value = None

                            result = await app.process_workflow(
                                self._make_request(), "req-broken"
                            )

            self.assertIn("error", result)
            self.assertIn("details", result)

        asyncio.run(_run())

    def test_multiple_output_nodes(self):
        """Multiple output nodes with images should all be collected."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            "p1": {
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {"filename": "a.png", "subfolder": "", "type": "output"}
                                        ]
                                    },
                                    "12": {
                                        "images": [
                                            {"filename": "b.png", "subfolder": "", "type": "output"}
                                        ]
                                    },
                                }
                            }
                        }
                        with patch("app.get_image_data") as mock_img:
                            mock_img.return_value = b"imgdata"

                            result = await app.process_workflow(
                                self._make_request(), "req-multi"
                            )

            self.assertEqual(len(result["images"]), 2)
            filenames = {img["filename"] for img in result["images"]}
            self.assertEqual(filenames, {"a.png", "b.png"})

        asyncio.run(_run())

    def test_execution_error_with_partial_outputs(self):
        """Execution error but history still has some outputs."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (
                        False,
                        ["Workflow execution error: OOM"],
                    )
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            "p1": {
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {"filename": "partial.png", "subfolder": "", "type": "output"}
                                        ]
                                    }
                                }
                            }
                        }
                        with patch("app.get_image_data") as mock_img:
                            mock_img.return_value = b"partial"

                            result = await app.process_workflow(
                                self._make_request(), "req-partial"
                            )

            self.assertIn("images", result)
            self.assertIn("errors", result)
            self.assertEqual(len(result["images"]), 1)

        asyncio.run(_run())

    def test_missing_filename_in_image_info(self):
        """Images without filename should be skipped with a warning."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {
                            "p1": {
                                "outputs": {
                                    "9": {
                                        "images": [
                                            {"subfolder": "", "type": "output"}
                                            # no filename
                                        ]
                                    }
                                }
                            }
                        }

                        result = await app.process_workflow(
                            self._make_request(), "req-noname"
                        )

            self.assertIn("error", result)
            self.assertTrue(
                any("missing filename" in d.lower() for d in result.get("details", []))
            )

        asyncio.run(_run())

    def test_api_key_passed_to_queue_workflow(self):
        """comfy_org_api_key from request should reach queue_workflow."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.return_value = {"p1": {"outputs": {}}}

                        request = self._make_request(comfy_org_api_key="my-key")
                        await app.process_workflow(request, "req-key")

            # Check that queue_workflow was called with the api key
            call_kwargs = mock_queue.call_args
            self.assertEqual(call_kwargs[0][2], "my-key")  # 3rd positional arg

        asyncio.run(_run())

    def test_get_history_failure(self):
        """Exception from get_history should return error."""

        async def _run():
            with patch("app.queue_workflow") as mock_queue:
                mock_queue.return_value = {"prompt_id": "p1"}
                with patch(
                    "app._monitor_execution", new_callable=AsyncMock
                ) as mock_monitor:
                    mock_monitor.return_value = (True, [])
                    with patch("app.get_history") as mock_history:
                        mock_history.side_effect = Exception("history fetch failed")

                        result = await app.process_workflow(
                            self._make_request(), "req-histfail"
                        )

            self.assertIn("error", result)
            self.assertIn("history", result["error"].lower())

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
