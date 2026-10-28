"""Verify HACS can install repository source without published release assets."""

import json
from pathlib import Path


def test_hacs_source_download_layout() -> None:
    """Keep branch installs independent of release ZIPs and match HACS's layout."""
    root = Path(__file__).parents[1]
    hacs = json.loads((root / "hacs.json").read_text())
    assert not hacs.get("zip_release", False)
    assert "filename" not in hacs
    assert hacs.get("content_in_root", False) is False
    integration = root / "custom_components" / "azure_openai_conversation"
    manifest = json.loads((integration / "manifest.json").read_text())
    assert manifest["domain"] == integration.name
    assert (integration / "__init__.py").is_file()
