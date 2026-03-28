"""
Tests for FastAPI endpoints: /ping and /generate.

Uses FastAPI's TestClient which properly invokes the lifespan context.
"""

import os
import sys
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app


# ---------------------------------------------------------------------------
# Helper: build a test app with controlled comfyui_ready state
# ---------------------------------------------------------------------------

def _make_client(comfyui_ready=True):
    """
    Build a TestClient around a fresh FastAPI app whose lifespan sets
    comfyui_ready to the requested value. The route handlers delegate to
    the real app module functions so patches on `app.*` work.
    """

    @asynccontextmanager
    async def _test_lifespan(a: FastAPI):
        a.state.comfyui_ready = comfyui_ready
        yield

    test_app = FastAPI(lifespan=_test_lifespan)

    @test_app.get("/ping")
    async def ping():
        if not test_app.state.comfyui_ready:
            return Response(status_code=204)
        return Response(status_code=200, content="OK")

    @test_app.post("/generate")
    async def generate(request: app.GenerateRequest):
        if not test_app.state.comfyui_ready:
            return JSONResponse(
                status_code=503,
                content={"error": "ComfyUI is not ready yet"},
            )

        import uuid
        import traceback

        request_id = str(uuid.uuid4())

        try:
            result = await asyncio.wait_for(
                app.process_workflow(request, request_id),
                timeout=app.PROCESSING_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": f"Workflow processing timed out after {app.PROCESSING_TIMEOUT_S}s"},
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"An unexpected error occurred: {e}"},
            )

        if "error" in result:
            status_code = 422 if "validation" in result.get("error", "").lower() else 500
            return JSONResponse(status_code=status_code, content=result)

        return JSONResponse(status_code=200, content=result)

    return TestClient(test_app)


# ═══════════════════════════════════════════════════════════════════════════════
# /ping endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestPingEndpoint(unittest.TestCase):
    """Tests for GET /ping."""

    def test_ping_returns_200_when_ready(self):
        with _make_client(comfyui_ready=True) as client:
            resp = client.get("/ping")
            self.assertEqual(resp.status_code, 200)

    def test_ping_returns_204_when_not_ready(self):
        with _make_client(comfyui_ready=False) as client:
            resp = client.get("/ping")
            self.assertEqual(resp.status_code, 204)


# ═══════════════════════════════════════════════════════════════════════════════
# /generate endpoint — request validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateValidation(unittest.TestCase):
    """Tests for POST /generate input validation."""

    def test_returns_503_when_comfyui_not_ready(self):
        with _make_client(comfyui_ready=False) as client:
            resp = client.post("/generate", json={"workflow": {"6": {}}})
            self.assertEqual(resp.status_code, 503)
            self.assertIn("not ready", resp.json()["error"])

    def test_returns_422_for_missing_workflow(self):
        with _make_client(comfyui_ready=True) as client:
            resp = client.post("/generate", json={})
            self.assertEqual(resp.status_code, 422)

    def test_returns_422_for_invalid_body(self):
        with _make_client(comfyui_ready=True) as client:
            resp = client.post(
                "/generate",
                content="not json",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(resp.status_code, 422)

    def test_accepts_workflow_only(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {
                    "images": [{"filename": "test.png", "type": "base64", "data": "abc"}]
                }
                resp = client.post("/generate", json={"workflow": {"6": {}}})
                self.assertEqual(resp.status_code, 200)
                mock_pw.assert_called_once()

    def test_accepts_workflow_with_images(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {"images": []}
                resp = client.post(
                    "/generate",
                    json={
                        "workflow": {"6": {}},
                        "images": [{"name": "img.png", "image": "data"}],
                    },
                )
                self.assertEqual(resp.status_code, 200)

    def test_accepts_workflow_with_api_key(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {"images": []}
                resp = client.post(
                    "/generate",
                    json={
                        "workflow": {"6": {}},
                        "comfy_org_api_key": "key-123",
                    },
                )
                self.assertEqual(resp.status_code, 200)


# ═══════════════════════════════════════════════════════════════════════════════
# /generate endpoint — response handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateResponses(unittest.TestCase):
    """Tests for POST /generate response codes and bodies."""

    def test_returns_200_with_images(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {
                    "images": [
                        {"filename": "out.png", "type": "base64", "data": "aGVsbG8="}
                    ]
                }
                resp = client.post("/generate", json={"workflow": {"6": {}}})
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertEqual(len(body["images"]), 1)
                self.assertEqual(body["images"][0]["filename"], "out.png")

    def test_returns_error_from_process_workflow(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {"error": "Something went wrong"}
                resp = client.post("/generate", json={"workflow": {"6": {}}})
                self.assertEqual(resp.status_code, 500)
                self.assertIn("error", resp.json())

    def test_returns_422_for_validation_error(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {
                    "error": "Workflow validation failed: something"
                }
                resp = client.post("/generate", json={"workflow": {"6": {}}})
                self.assertEqual(resp.status_code, 422)

    def test_returns_504_on_timeout(self):
        with _make_client(comfyui_ready=True) as client:

            async def slow_workflow(*args, **kwargs):
                await asyncio.sleep(999)

            with patch("app.process_workflow", side_effect=slow_workflow):
                with patch("app.PROCESSING_TIMEOUT_S", 0.01):
                    resp = client.post("/generate", json={"workflow": {"6": {}}})
                    self.assertEqual(resp.status_code, 504)
                    self.assertIn("timed out", resp.json()["error"])

    def test_returns_500_on_unexpected_exception(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.side_effect = RuntimeError("unexpected crash")
                resp = client.post("/generate", json={"workflow": {"6": {}}})
                self.assertEqual(resp.status_code, 500)
                self.assertIn("unexpected", resp.json()["error"].lower())

    def test_returns_success_no_images(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {
                    "status": "success_no_images",
                    "images": [],
                }
                resp = client.post("/generate", json={"workflow": {"6": {}}})
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertEqual(body["status"], "success_no_images")
                self.assertEqual(body["images"], [])

    def test_returns_images_with_errors(self):
        with _make_client(comfyui_ready=True) as client:
            with patch("app.process_workflow", new_callable=AsyncMock) as mock_pw:
                mock_pw.return_value = {
                    "images": [{"filename": "a.png", "type": "base64", "data": "x"}],
                    "errors": ["Warning: something minor"],
                }
                resp = client.post("/generate", json={"workflow": {"6": {}}})
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertEqual(len(body["images"]), 1)
                self.assertEqual(len(body["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
