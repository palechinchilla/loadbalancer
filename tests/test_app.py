"""
Unit tests for app.py helper functions.

Tests cover: validate_input, upload_images, get_available_models,
queue_workflow, get_history, get_image_data, _comfy_server_status,
_get_comfyui_pid, _is_comfyui_process_alive.
"""

import os
import sys
import json
import base64
import unittest
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app


# ═══════════════════════════════════════════════════════════════════════════════
# validate_input
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateInput(unittest.TestCase):
    """Tests for validate_input()."""

    def test_valid_workflow_only(self):
        data = {"workflow": {"6": {"class_type": "CLIPTextEncode"}}}
        result, err = app.validate_input(data)
        self.assertIsNone(err)
        self.assertEqual(result["workflow"], data["workflow"])
        self.assertIsNone(result["images"])
        self.assertIsNone(result["comfy_org_api_key"])

    def test_valid_workflow_with_images(self):
        data = {
            "workflow": {"6": {}},
            "images": [{"name": "img.png", "image": "base64data"}],
        }
        result, err = app.validate_input(data)
        self.assertIsNone(err)
        self.assertEqual(len(result["images"]), 1)

    def test_valid_workflow_with_api_key(self):
        data = {
            "workflow": {"6": {}},
            "comfy_org_api_key": "key-123",
        }
        result, err = app.validate_input(data)
        self.assertIsNone(err)
        self.assertEqual(result["comfy_org_api_key"], "key-123")

    def test_none_input(self):
        result, err = app.validate_input(None)
        self.assertIsNone(result)
        self.assertEqual(err, "Please provide input")

    def test_missing_workflow(self):
        result, err = app.validate_input({"images": []})
        self.assertIsNone(result)
        self.assertEqual(err, "Missing 'workflow' parameter")

    def test_invalid_images_missing_image_key(self):
        data = {"workflow": {"6": {}}, "images": [{"name": "img.png"}]}
        result, err = app.validate_input(data)
        self.assertIsNone(result)
        self.assertIn("'images' must be a list", err)

    def test_invalid_images_missing_name_key(self):
        data = {"workflow": {"6": {}}, "images": [{"image": "base64data"}]}
        result, err = app.validate_input(data)
        self.assertIsNone(result)
        self.assertIn("'images' must be a list", err)

    def test_invalid_images_not_a_list(self):
        data = {"workflow": {"6": {}}, "images": "not_a_list"}
        result, err = app.validate_input(data)
        self.assertIsNone(result)
        self.assertIn("'images' must be a list", err)

    def test_valid_json_string_input(self):
        json_str = '{"workflow": {"6": {}}}'
        result, err = app.validate_input(json_str)
        self.assertIsNone(err)
        self.assertIsNotNone(result["workflow"])

    def test_invalid_json_string_input(self):
        result, err = app.validate_input("not valid json")
        self.assertIsNone(result)
        self.assertEqual(err, "Invalid JSON format in input")

    def test_empty_workflow_is_valid(self):
        result, err = app.validate_input({"workflow": {}})
        self.assertIsNone(err)
        self.assertEqual(result["workflow"], {})

    def test_multiple_images_all_valid(self):
        data = {
            "workflow": {"6": {}},
            "images": [
                {"name": "a.png", "image": "data1"},
                {"name": "b.png", "image": "data2"},
            ],
        }
        result, err = app.validate_input(data)
        self.assertIsNone(err)
        self.assertEqual(len(result["images"]), 2)

    def test_multiple_images_one_invalid(self):
        data = {
            "workflow": {"6": {}},
            "images": [
                {"name": "a.png", "image": "data1"},
                {"name": "b.png"},  # missing image key
            ],
        }
        result, err = app.validate_input(data)
        self.assertIsNone(result)
        self.assertIn("'images' must be a list", err)


# ═══════════════════════════════════════════════════════════════════════════════
# _comfy_server_status
# ═══════════════════════════════════════════════════════════════════════════════


class TestComfyServerStatus(unittest.TestCase):
    """Tests for _comfy_server_status()."""

    @patch("app.requests.get")
    def test_reachable(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        result = app._comfy_server_status()
        self.assertTrue(result["reachable"])
        self.assertEqual(result["status_code"], 200)

    @patch("app.requests.get")
    def test_non_200_response(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        result = app._comfy_server_status()
        self.assertFalse(result["reachable"])
        self.assertEqual(result["status_code"], 500)

    @patch("app.requests.get", side_effect=ConnectionError("refused"))
    def test_connection_error(self, mock_get):
        result = app._comfy_server_status()
        self.assertFalse(result["reachable"])
        self.assertIn("error", result)


# ═══════════════════════════════════════════════════════════════════════════════
# _get_comfyui_pid / _is_comfyui_process_alive
# ═══════════════════════════════════════════════════════════════════════════════


class TestComfyUIPid(unittest.TestCase):
    """Tests for _get_comfyui_pid() and _is_comfyui_process_alive()."""

    @patch("builtins.open", mock_open(read_data="12345\n"))
    def test_get_pid_success(self):
        self.assertEqual(app._get_comfyui_pid(), 12345)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_get_pid_file_not_found(self, _):
        self.assertIsNone(app._get_comfyui_pid())

    @patch("builtins.open", mock_open(read_data="not_a_number"))
    def test_get_pid_invalid_content(self):
        self.assertIsNone(app._get_comfyui_pid())

    @patch("app._get_comfyui_pid", return_value=None)
    def test_process_alive_no_pid(self, _):
        self.assertIsNone(app._is_comfyui_process_alive())

    @patch("app._get_comfyui_pid", return_value=99999)
    @patch("os.kill")
    def test_process_alive_true(self, mock_kill, _):
        mock_kill.return_value = None  # no exception = process exists
        self.assertTrue(app._is_comfyui_process_alive())

    @patch("app._get_comfyui_pid", return_value=99999)
    @patch("os.kill", side_effect=ProcessLookupError)
    def test_process_dead(self, mock_kill, _):
        self.assertFalse(app._is_comfyui_process_alive())

    @patch("app._get_comfyui_pid", return_value=1)
    @patch("os.kill", side_effect=PermissionError)
    def test_process_permission_error_treated_as_alive(self, mock_kill, _):
        self.assertTrue(app._is_comfyui_process_alive())


# ═══════════════════════════════════════════════════════════════════════════════
# upload_images
# ═══════════════════════════════════════════════════════════════════════════════


class TestUploadImages(unittest.TestCase):
    """Tests for upload_images()."""

    def test_empty_list(self):
        result = app.upload_images([])
        self.assertEqual(result["status"], "success")

    def test_none_input(self):
        result = app.upload_images(None)
        self.assertEqual(result["status"], "success")

    @patch("app.requests.post")
    def test_successful_upload(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        test_data = base64.b64encode(b"PNG_IMAGE_DATA").decode("utf-8")
        images = [{"name": "test.png", "image": test_data}]

        result = app.upload_images(images)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["details"]), 1)
        mock_post.assert_called_once()

    @patch("app.requests.post")
    def test_successful_upload_with_data_uri_prefix(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        raw = base64.b64encode(b"PNG_IMAGE_DATA").decode("utf-8")
        images = [{"name": "test.png", "image": f"data:image/png;base64,{raw}"}]

        result = app.upload_images(images)
        self.assertEqual(result["status"], "success")

    @patch("app.requests.post")
    def test_upload_http_error(self, mock_post):
        import requests as req

        mock_post.return_value = MagicMock(status_code=400)
        mock_post.return_value.raise_for_status.side_effect = req.HTTPError("400")

        test_data = base64.b64encode(b"data").decode("utf-8")
        images = [{"name": "test.png", "image": test_data}]

        result = app.upload_images(images)
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["details"]), 1)

    @patch("app.requests.post", side_effect=Exception("timeout"))
    def test_upload_timeout(self, mock_post):
        test_data = base64.b64encode(b"data").decode("utf-8")
        images = [{"name": "test.png", "image": test_data}]

        result = app.upload_images(images)
        self.assertEqual(result["status"], "error")

    def test_invalid_base64(self):
        images = [{"name": "test.png", "image": "!!!not_base64!!!"}]
        result = app.upload_images(images)
        self.assertEqual(result["status"], "error")
        self.assertTrue(any("decoding" in d.lower() or "error" in d.lower() for d in result["details"]))

    @patch("app.requests.post")
    def test_multiple_images_partial_failure(self, mock_post):
        success_resp = MagicMock(status_code=200)
        success_resp.raise_for_status = MagicMock()

        import requests as req
        fail_resp = MagicMock(status_code=400)
        fail_resp.raise_for_status.side_effect = req.HTTPError("400")

        mock_post.side_effect = [success_resp, fail_resp]

        test_data = base64.b64encode(b"data").decode("utf-8")
        images = [
            {"name": "good.png", "image": test_data},
            {"name": "bad.png", "image": test_data},
        ]

        result = app.upload_images(images)
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["details"]), 1)  # only the failure


# ═══════════════════════════════════════════════════════════════════════════════
# get_available_models
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetAvailableModels(unittest.TestCase):
    """Tests for get_available_models()."""

    @patch("app.requests.get")
    def test_returns_checkpoints(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "CheckpointLoaderSimple": {
                        "input": {
                            "required": {
                                "ckpt_name": [
                                    ["model_a.safetensors", "model_b.safetensors"]
                                ]
                            }
                        }
                    }
                }
            ),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = app.get_available_models()
        self.assertIn("checkpoints", result)
        self.assertEqual(len(result["checkpoints"]), 2)

    @patch("app.requests.get")
    def test_no_checkpoint_loader(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"OtherNode": {}}),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = app.get_available_models()
        self.assertEqual(result, {})

    @patch("app.requests.get", side_effect=ConnectionError("refused"))
    def test_connection_error_returns_empty(self, _):
        result = app.get_available_models()
        self.assertEqual(result, {})


# ═══════════════════════════════════════════════════════════════════════════════
# queue_workflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueueWorkflow(unittest.TestCase):
    """Tests for queue_workflow()."""

    @patch("app.requests.post")
    def test_successful_queue(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompt_id": "abc-123"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = app.queue_workflow({"6": {}}, "client-id-1")
        self.assertEqual(result["prompt_id"], "abc-123")

    @patch("app.requests.post")
    def test_includes_client_id_in_payload(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompt_id": "abc-123"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        app.queue_workflow({"6": {}}, "my-client-id")

        call_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(call_data["client_id"], "my-client-id")

    @patch("app.requests.post")
    def test_includes_per_request_api_key(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompt_id": "abc-123"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        app.queue_workflow({"6": {}}, "cid", comfy_org_api_key="req-key")

        call_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(
            call_data["extra_data"]["api_key_comfy_org"], "req-key"
        )

    @patch.dict(os.environ, {"COMFY_ORG_API_KEY": "env-key"})
    @patch("app.requests.post")
    def test_per_request_key_overrides_env_key(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompt_id": "abc-123"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        app.queue_workflow({"6": {}}, "cid", comfy_org_api_key="req-key")

        call_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(
            call_data["extra_data"]["api_key_comfy_org"], "req-key"
        )

    @patch.dict(os.environ, {"COMFY_ORG_API_KEY": "env-key"})
    @patch("app.requests.post")
    def test_env_key_used_when_no_request_key(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompt_id": "abc-123"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        app.queue_workflow({"6": {}}, "cid")

        call_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(
            call_data["extra_data"]["api_key_comfy_org"], "env-key"
        )

    @patch("app.requests.post")
    def test_no_api_key_means_no_extra_data(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompt_id": "abc-123"}),
        )
        mock_post.return_value.raise_for_status = MagicMock()

        app.queue_workflow({"6": {}}, "cid")

        call_data = json.loads(mock_post.call_args[1]["data"])
        self.assertNotIn("extra_data", call_data)

    @patch("app.get_available_models", return_value={})
    @patch("app.requests.post")
    def test_400_with_node_errors(self, mock_post, _):
        error_body = {
            "error": {"message": "Validation failed", "type": "other"},
            "node_errors": {
                "6": {"ckpt_name": "model_x.safetensors not in list"},
            },
        }
        mock_post.return_value = MagicMock(
            status_code=400,
            text=json.dumps(error_body),
            json=MagicMock(return_value=error_body),
        )

        with self.assertRaises(ValueError) as ctx:
            app.queue_workflow({"6": {}}, "cid")
        self.assertIn("Node 6", str(ctx.exception))

    @patch("app.get_available_models", return_value={"checkpoints": ["model.safetensors"]})
    @patch("app.requests.post")
    def test_400_with_checkpoint_hint(self, mock_post, _):
        error_body = {
            "error": {"message": "Validation failed", "type": "other"},
            "node_errors": {
                "6": {"ckpt_name": "bad_model.safetensors not in list"},
            },
        }
        mock_post.return_value = MagicMock(
            status_code=400,
            text=json.dumps(error_body),
            json=MagicMock(return_value=error_body),
        )

        with self.assertRaises(ValueError) as ctx:
            app.queue_workflow({"6": {}}, "cid")
        self.assertIn("model.safetensors", str(ctx.exception))

    @patch("app.get_available_models", return_value={})
    @patch("app.requests.post")
    def test_400_prompt_outputs_failed_validation(self, mock_post, _):
        error_body = {
            "type": "prompt_outputs_failed_validation",
            "message": "Outputs failed",
        }
        mock_post.return_value = MagicMock(
            status_code=400,
            text=json.dumps(error_body),
            json=MagicMock(return_value=error_body),
        )

        with self.assertRaises(ValueError) as ctx:
            app.queue_workflow({"6": {}}, "cid")
        self.assertIn("not available", str(ctx.exception))

    @patch("app.requests.post")
    def test_400_unparseable_json(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=400,
            text="not json at all",
            json=MagicMock(side_effect=json.JSONDecodeError("", "", 0)),
        )

        with self.assertRaises(ValueError) as ctx:
            app.queue_workflow({"6": {}}, "cid")
        self.assertIn("could not parse", str(ctx.exception).lower())

    @patch("app.requests.post")
    def test_500_raises_http_error(self, mock_post):
        import requests as req

        mock_post.return_value = MagicMock(status_code=500)
        mock_post.return_value.raise_for_status.side_effect = req.HTTPError("500")

        with self.assertRaises(req.HTTPError):
            app.queue_workflow({"6": {}}, "cid")


# ═══════════════════════════════════════════════════════════════════════════════
# get_history
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetHistory(unittest.TestCase):
    """Tests for get_history()."""

    @patch("app.requests.get")
    def test_returns_history(self, mock_get):
        history_data = {"prompt-123": {"outputs": {"9": {"images": []}}}}
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=history_data),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = app.get_history("prompt-123")
        self.assertIn("prompt-123", result)

    @patch("app.requests.get")
    def test_calls_correct_url(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={}),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        app.get_history("my-prompt-id")
        called_url = mock_get.call_args[0][0]
        self.assertIn("/history/my-prompt-id", called_url)

    @patch("app.requests.get")
    def test_raises_on_http_error(self, mock_get):
        import requests as req

        mock_get.return_value = MagicMock(status_code=500)
        mock_get.return_value.raise_for_status.side_effect = req.HTTPError("500")

        with self.assertRaises(req.HTTPError):
            app.get_history("prompt-123")


# ═══════════════════════════════════════════════════════════════════════════════
# get_image_data
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetImageData(unittest.TestCase):
    """Tests for get_image_data()."""

    @patch("app.requests.get")
    def test_returns_image_bytes(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            content=b"\x89PNG\r\n\x1a\nfakedata",
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = app.get_image_data("image.png", "", "output")
        self.assertEqual(result, b"\x89PNG\r\n\x1a\nfakedata")

    @patch("app.requests.get")
    def test_correct_url_params(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, content=b"data")
        mock_get.return_value.raise_for_status = MagicMock()

        app.get_image_data("img.png", "sub", "output")
        called_url = mock_get.call_args[0][0]
        self.assertIn("filename=img.png", called_url)
        self.assertIn("subfolder=sub", called_url)
        self.assertIn("type=output", called_url)

    @patch("app.requests.get", side_effect=Exception("timeout"))
    def test_timeout_returns_none(self, _):
        result = app.get_image_data("img.png", "", "output")
        self.assertIsNone(result)

    @patch("app.requests.get")
    def test_request_exception_returns_none(self, mock_get):
        import requests as req

        mock_get.side_effect = req.RequestException("connection error")
        result = app.get_image_data("img.png", "", "output")
        self.assertIsNone(result)

    @patch("app.requests.get")
    def test_timeout_exception_returns_none(self, mock_get):
        import requests as req

        mock_get.side_effect = req.Timeout("timed out")
        result = app.get_image_data("img.png", "", "output")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
