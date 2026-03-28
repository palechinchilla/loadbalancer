"""Tests for s3_upload module."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from s3_upload import _get_bucket_config, _parse_bucket_name, _get_s3_client, upload_to_s3


class TestGetBucketConfig(unittest.TestCase):
    """Tests for _get_bucket_config()."""

    @patch.dict(
        os.environ,
        {
            "BUCKET_ENDPOINT_URL": "https://mybucket.s3.us-east-1.amazonaws.com",
            "BUCKET_ACCESS_KEY_ID": "AKIATEST",
            "BUCKET_SECRET_ACCESS_KEY": "secret123",
        },
    )
    def test_reads_all_env_vars(self):
        config = _get_bucket_config()
        self.assertEqual(
            config["endpoint_url"], "https://mybucket.s3.us-east-1.amazonaws.com"
        )
        self.assertEqual(config["access_key"], "AKIATEST")
        self.assertEqual(config["secret_key"], "secret123")

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_when_endpoint_url_missing(self):
        with self.assertRaises(ValueError) as ctx:
            _get_bucket_config()
        self.assertIn("BUCKET_ENDPOINT_URL", str(ctx.exception))

    @patch.dict(os.environ, {"BUCKET_ENDPOINT_URL": ""}, clear=True)
    def test_raises_when_endpoint_url_empty(self):
        with self.assertRaises(ValueError):
            _get_bucket_config()

    @patch.dict(
        os.environ,
        {"BUCKET_ENDPOINT_URL": "https://mybucket.s3.amazonaws.com"},
        clear=True,
    )
    def test_defaults_empty_keys_when_not_set(self):
        config = _get_bucket_config()
        self.assertEqual(config["access_key"], "")
        self.assertEqual(config["secret_key"], "")


class TestParseBucketName(unittest.TestCase):
    """Tests for _parse_bucket_name()."""

    def test_aws_virtual_hosted_style(self):
        url = "https://mybucket.s3.us-east-1.amazonaws.com"
        self.assertEqual(_parse_bucket_name(url), "mybucket")

    def test_digitalocean_spaces(self):
        url = "https://mybucket.nyc3.digitaloceanspaces.com"
        self.assertEqual(_parse_bucket_name(url), "mybucket")

    def test_cloudflare_r2_path_style(self):
        url = "https://accountid.r2.cloudflarestorage.com/mybucket"
        self.assertEqual(_parse_bucket_name(url), "mybucket")

    def test_path_style_with_trailing_slash(self):
        url = "https://accountid.r2.cloudflarestorage.com/mybucket/"
        self.assertEqual(_parse_bucket_name(url), "mybucket")

    def test_no_path_falls_back_to_subdomain(self):
        url = "https://mybucket.example.com"
        self.assertEqual(_parse_bucket_name(url), "mybucket")


class TestGetS3Client(unittest.TestCase):
    """Tests for _get_s3_client()."""

    @patch("s3_upload.boto3.client")
    def test_creates_client_with_config(self, mock_boto_client):
        config = {
            "endpoint_url": "https://mybucket.s3.amazonaws.com",
            "access_key": "AKIATEST",
            "secret_key": "secret123",
        }
        _get_s3_client(config)
        mock_boto_client.assert_called_once()
        call_kwargs = mock_boto_client.call_args
        self.assertEqual(call_kwargs[1]["endpoint_url"], config["endpoint_url"])
        self.assertEqual(call_kwargs[1]["aws_access_key_id"], "AKIATEST")
        self.assertEqual(call_kwargs[1]["aws_secret_access_key"], "secret123")


class TestUploadToS3(unittest.TestCase):
    """Tests for upload_to_s3()."""

    @patch.dict(
        os.environ,
        {
            "BUCKET_ENDPOINT_URL": "https://mybucket.s3.us-east-1.amazonaws.com",
            "BUCKET_ACCESS_KEY_ID": "AKIATEST",
            "BUCKET_SECRET_ACCESS_KEY": "secret123",
        },
    )
    @patch("s3_upload._get_s3_client")
    def test_upload_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name

        try:
            url = upload_to_s3("req-123", temp_path)

            mock_client.upload_file.assert_called_once()
            call_args = mock_client.upload_file.call_args[0]
            self.assertEqual(call_args[0], temp_path)  # file path
            self.assertEqual(call_args[1], "mybucket")  # bucket name
            self.assertIn("req-123/", call_args[2])  # key prefix
            self.assertIn("req-123/", url)
            self.assertTrue(url.startswith("https://mybucket.s3.us-east-1.amazonaws.com/"))
        finally:
            os.remove(temp_path)

    @patch.dict(
        os.environ,
        {
            "BUCKET_ENDPOINT_URL": "https://mybucket.s3.us-east-1.amazonaws.com/",
            "BUCKET_ACCESS_KEY_ID": "AKIATEST",
            "BUCKET_SECRET_ACCESS_KEY": "secret123",
        },
    )
    @patch("s3_upload._get_s3_client")
    def test_upload_strips_trailing_slash_from_url(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake")
            temp_path = f.name

        try:
            url = upload_to_s3("req-456", temp_path)
            # Should not have double slashes
            self.assertNotIn("//req-456", url)
            self.assertIn("/req-456/", url)
        finally:
            os.remove(temp_path)

    @patch.dict(os.environ, {}, clear=True)
    def test_upload_fails_without_env_vars(self):
        with self.assertRaises(ValueError):
            upload_to_s3("req-789", "/tmp/fake.png")

    @patch.dict(
        os.environ,
        {
            "BUCKET_ENDPOINT_URL": "https://mybucket.s3.amazonaws.com",
            "BUCKET_ACCESS_KEY_ID": "AKIATEST",
            "BUCKET_SECRET_ACCESS_KEY": "secret123",
        },
    )
    @patch("s3_upload._get_s3_client")
    def test_upload_preserves_filename(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with tempfile.NamedTemporaryFile(
            suffix=".webp", delete=False, prefix="ComfyUI_00001_"
        ) as f:
            f.write(b"fake")
            temp_path = f.name

        try:
            url = upload_to_s3("req-abc", temp_path)
            filename = os.path.basename(temp_path)
            self.assertTrue(url.endswith(filename))
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
