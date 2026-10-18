"""Tests for the voice catalogue."""



from __future__ import annotations

import pytest

from app.voices import OPENAI_VOICE_ALIASES, VOICE_IDS, VOICES, VoiceError, catalogue, resolve





def test_voice_count():

    assert len(VOICES) == 14

    assert len(VOICE_IDS) == 14





def test_unique_ids():

    assert len(set(VOICE_IDS)) == len(VOICE_IDS)





def test_default_voice_present():

    assert "diem_trinh" in VOICE_IDS





def test_resolve_alias():

    assert resolve("alloy", "diem_trinh") == "diem_trinh"

    assert resolve("nova", "diem_trinh") == "mai_loan"

    assert resolve("onyx", "diem_trinh") == "manh_dung"

    assert resolve("cedar", "diem_trinh") == "thanh_dat"





def test_resolve_case_insensitive():

    assert resolve("Diễm_Trinh", "diem_trinh") == "diem_trinh"

    assert resolve("DIEM_TRINH", "diem_trinh") == "diem_trinh"





def test_resolve_label_alias():

    assert resolve("Diễm Trinh", "diem_trinh") == "diem_trinh"

    assert resolve("Mạnh Dũng", "diem_trinh") == "manh_dung"





def test_resolve_language_prefix():

    assert resolve("vi_diem_trinh", "diem_trinh") == "diem_trinh"

    assert resolve("vi_alloy", "diem_trinh") == OPENAI_VOICE_ALIASES["alloy"]





def test_resolve_defaults_none():

    assert resolve(None, "diem_trinh") == "diem_trinh"

    assert resolve("", "diem_trinh") == "diem_trinh"

    assert resolve("   ", "diem_trinh") == "diem_trinh"





def test_resolve_unknown_raises():

    with pytest.raises(VoiceError):

        resolve("not_a_voice", "diem_trinh")





def test_catalogue_shape():

    for item in catalogue():

        assert "id" in item

        assert "name" in item

        assert "language" in item

        assert "gender" in item

        assert "style" in item

