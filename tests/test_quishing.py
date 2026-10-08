from types import SimpleNamespace

import numpy as np
import zxingcpp
from PIL import Image

import app
from app import EXAMPLES, _image_from_data_url, _message_with_qr_payload
from jawbreaker.quishing import extract_qr_payload


def test_extract_qr_payload_accepts_pil_path_and_numpy_array(monkeypatch, tmp_path) -> None:
    seen_images = []

    def fake_read_barcodes(image, *, formats):
        assert formats == zxingcpp.BarcodeFormat.QRCode
        seen_images.append(image)
        return [SimpleNamespace(text="  https://paypa1.com/login\n")]

    monkeypatch.setattr(zxingcpp, "read_barcodes", fake_read_barcodes)
    image = Image.new("RGB", (24, 24), "white")
    image_path = tmp_path / "qr.png"
    image.save(image_path)

    assert extract_qr_payload(image) == "https://paypa1.com/login"
    assert extract_qr_payload(image_path) == "https://paypa1.com/login"
    assert extract_qr_payload(np.asarray(image)) == "https://paypa1.com/login"
    assert len(seen_images) == 3


def test_extract_qr_payload_returns_none_when_no_qr_exists(monkeypatch) -> None:
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda image, **kwargs: [])

    assert extract_qr_payload(Image.new("RGB", (24, 24), "white")) is None
    assert extract_qr_payload(None) is None


def test_bundled_quishing_demo_image_decodes() -> None:
    assert extract_qr_payload(app.SAFE_QR_DEMO_PATH) == (
        "Library hours: weekdays 9 AM to 5 PM. No payment or sign-in is needed."
    )
    assert extract_qr_payload(app.PHISHING_QR_DEMO_PATH) == "https://paypa1.example/login"


def test_extract_qr_payload_returns_all_unique_codes(monkeypatch) -> None:
    monkeypatch.setattr(
        zxingcpp,
        "read_barcodes",
        lambda image, **kwargs: [
            SimpleNamespace(text=" https://safe.example "),
            SimpleNamespace(text=" https://paypa1.example/login "),
            SimpleNamespace(text="https://safe.example"),
        ],
    )

    payload = extract_qr_payload(Image.new("RGB", (24, 24), "white"))

    assert payload == "https://safe.example\nhttps://paypa1.example/login"


def test_ui_keeps_homograph_and_cache_demo_examples() -> None:
    assert any("paypa1.example" in example for example in EXAMPLES)
    assert any("Cache demo" in example for example in EXAMPLES)
    html = app.kitchen_table_html()
    assert "Load safe QR demo" in html
    assert "Load phishing QR demo" in html
    assert html.count("data:image/png;base64,") >= 3


def test_message_helper_appends_qr_payload_and_preserves_text_only(monkeypatch) -> None:
    monkeypatch.setattr("app.extract_qr_payload", lambda image: "https://example.com/verify")

    assert _message_with_qr_payload("Check this message", object()) == (
        "Check this message\n[Detected QR Code Link]: https://example.com/verify",
        "https://example.com/verify",
    )
    assert _message_with_qr_payload("Check this message", None) == ("Check this message", None)


def test_image_data_url_decodes_to_pil_image() -> None:
    import base64
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (12, 10), "white").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    decoded = _image_from_data_url(data_url)

    assert isinstance(decoded, Image.Image)
    assert decoded.size == (12, 10)
    assert _image_from_data_url("not-an-image") is None
