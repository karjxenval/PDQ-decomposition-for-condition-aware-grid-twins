import json
from pathlib import Path
from dce_data.core import safe_name

ROOT = Path(__file__).resolve().parents[1]

def test_source_catalog_core_entries():
    c = json.loads((ROOT / "configs" / "sources.json").read_text(encoding="utf-8"))
    assert c["kios_real_pmu_2023"]["record_id"] == "8343635"
    assert c["grideye_real_pmu_2025"]["record_id"] == "17648863"
    assert c["ponte_moesa_2025"]["item_uuid"] == "1d379016-0fc4-49c7-8ff0-85cbc12e624f"

def test_safe_name_preserves_extension():
    assert safe_name("a/b:c?.csv").endswith(".csv")

def test_validation_config_has_modes():
    c = json.loads((ROOT / "configs" / "validation.json").read_text(encoding="utf-8"))
    assert "quick" in c and "publication" in c and "seed" in c
