"""KnowledgeBase against an in-memory Qdrant (the embedding model is real and cached under data/models)."""

import numpy as np
import pytest
import soundfile as sf

from soundkb.config import Settings
from soundkb.kb import KBError, KnowledgeBase, parse_recipe, resolve_allowed
from soundkb.store.schema import PatchRecipe

SR = 44100


@pytest.fixture()
def kb(tmp_path):
    settings = Settings(qdrant_url=":memory:", data_dir=tmp_path / "data", ui_port=0, allowed_ingest_dirs=(tmp_path,))
    return KnowledgeBase(settings), settings


def _wav(path, freq=55.0):
    t = np.arange(SR * 2) / SR
    sf.write(path, (0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.5)).astype(np.float32), SR)
    return path


RECIPE = {"name": "Sub via Operator", "target": {"device": "Operator"}, "params": [{"name": "Osc A Level", "value": 0.8}]}


def test_ingest_search_update_delete(kb, tmp_path):
    base, _ = kb
    entry = base.ingest_audio(_wav(tmp_path / "sub.wav"), "Deep sub", ["808", "Sub Bass"], "dark and long")
    assert entry["tags"] == ["808", "sub-bass"]
    assert base.ingest_audio(_wav(tmp_path / "sub.wav"), "Deep sub")["id"] == entry["id"]  # same content -> same entry
    assert len(base.search()) == 1
    assert base.search("deep sub bass")[0]["id"] == entry["id"]
    assert base.search(tags=["808"])[0]["id"] == entry["id"]
    assert base.search(tags=["nope"]) == []
    base.update(entry["id"], tags=["kick"], notes="changed")
    assert base.get(entry["id"])["tags"] == ["kick"]
    assert base.report_png(entry["id"]).is_file()
    base.delete(entry["id"])
    assert base.search() == []


def test_recipe_roundtrip_and_validation(kb):
    base, _ = kb
    note = base.add_text("note", "808 recipe", "operator sine", ["operator"])
    base.add_recipe(note["id"], PatchRecipe.model_validate(RECIPE))
    assert base.get(note["id"])["patch_recipes"][0]["params"][0]["value"] == 0.8
    with pytest.raises(KBError, match="target.device"):
        parse_recipe('{"name": "x", "target": {}}')
    with pytest.raises(KBError, match="valid JSON"):
        parse_recipe("{")


def test_path_allow_list(kb, tmp_path):
    _, settings = kb
    inside = _wav(tmp_path / "ok.wav")
    assert resolve_allowed(inside, settings) == inside.resolve()
    with pytest.raises(KBError):
        resolve_allowed(tmp_path / ".." / "etc" / "hosts", settings)
    outside = tmp_path.parent / "outside.wav"
    _wav(outside)
    with pytest.raises(KBError, match="outside"):
        resolve_allowed(outside, settings)
