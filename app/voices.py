"""The Vietnamese voice catalogue and the OpenAI voice aliases.

``kokoro-vietnamese`` ships fourteen voicepacks, one ``.pt`` file each. A
voicepack is loaded when a :class:`~kokoro_vietnamese.KokoroVietnamese`
instance is constructed, so a voice is not a per-request knob: the engine keeps
one instance per voice and this module is what turns whatever name a client
sends into one of those canonical ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    """One voicepack, described well enough to advertise over HTTP."""

    id: str
    label: str
    gender: str
    #: Kokoro-style language tag, kept so the payload resembles the upstream
    #: Kokoro API that clients may already parse.
    language: str = "vi-VN"
    #: Roughly what the voice is good for, surfaced in ``/v1/voices``.
    style: str = "general"

#: Canonical ids accepted by ``kokoro-vietnamese``, in a stable display order.
VOICES: tuple[Voice, ...] = (
    Voice("diem_trinh", "Diễm Trinh", "female", style="narration"),
    Voice("mai_linh", "Mai Linh", "female"),
    Voice("mai_loan", "Mai Loan", "female"),
    Voice("my_yen", "Mỹ Yến", "female"),
    Voice("ngoc_huyen", "Ngọc Huyền", "female"),
    Voice("thuc_trinh", "Thục Trinh", "female"),
    Voice("duc_an", "Đức An", "male"),
    Voice("duc_duy", "Đức Duy", "male"),
    Voice("hung_thinh", "Hùng Thịnh", "male"),
    Voice("manh_dung", "Mạnh Dũng", "male"),
    Voice("phat_tai", "Phát Tài", "male"),
    Voice("thanh_dat", "Thành Đạt", "male"),
    Voice("tuan_ngoc", "Tuấn Ngọc", "male"),
    Voice("storyvert", "Storyvert", "neutral", style="audiobook"),
)

VOICE_IDS: tuple[str, ...] = tuple(v.id for v in VOICES)
VOICE_BY_ID: dict[str, Voice] = {v.id: v for v in VOICES}

#: The twelve names OpenAI's ``tts-1`` accepts. They have no Vietnamese
#: equivalent, so each is pinned to a voicepack of the matching gender; clients
#: that hard-code ``alloy`` or ``nova`` get Vietnamese speech instead of a 400.
OPENAI_VOICE_ALIASES: dict[str, str] = {
    "alloy": "diem_trinh",
    "ash": "duc_an",
    "ballad": "tuan_ngoc",
    "coral": "mai_linh",
    "echo": "duc_duy",
    "fable": "storyvert",
    "marin": "my_yen",
    "nova": "mai_loan",
    "onyx": "manh_dung",
    "sage": "ngoc_huyen",
    "shimmer": "thuc_trinh",
    "verse": "hung_thinh",
    "cedar": "thanh_dat",
}

#: Accepted as a prefix on a voice id and stripped, e.g. ``vi_diem_trinh``.
#: Kokoro's own model card uses ``a``/``b`` prefixes for English, and clients
#: copied from English Kokoro servers often send them on Vietnamese ids too.
_LANGUAGE_PREFIXES: tuple[str, ...] = ("vi", "vn", "v")

class VoiceError(ValueError):
    """Raised for a voice name that cannot be resolved."""

def _normalise(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")

def _clean(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())

def resolve(name: str | None, default: str) -> str:
    """Map a client-supplied voice name onto a canonical voicepack id.

    Accepts a canonical id, an OpenAI alias, language/gender prefixes
    (e.g. ``vi-vn-female-ngochuyen``, ``vi_ngoc_huyen``), or a label
    such as ``"Diễm Trinh"``. Raises :class:`VoiceError` when nothing matches.
    """
    candidate = _normalise(name) if name else ""
    if not candidate:
        return default

    # Direct match on canonical id or OpenAI alias
    if candidate in VOICE_BY_ID:
        return candidate
    if candidate in OPENAI_VOICE_ALIASES:
        return OPENAI_VOICE_ALIASES[candidate]

    # Check direct match with label
    for voice in VOICES:
        if _normalise(voice.label) == candidate:
            return voice.id

    # Strip composite prefixes like vi_vn_, vn_, vi_, female_, male_
    prefixes = ("vi_vn_female_", "vi_vn_male_", "vi_vn_", "vn_female_", "vn_male_", "vn_", "vi_female_", "vi_male_", "vi_", "v_", "female_", "male_")
    stripped = candidate
    for p in prefixes:
        if stripped.startswith(p):
            stripped = stripped[len(p):]
            break

    if stripped in VOICE_BY_ID:
        return stripped
    if stripped in OPENAI_VOICE_ALIASES:
        return OPENAI_VOICE_ALIASES[stripped]
    for voice in VOICES:
        if _normalise(voice.label) == stripped:
            return voice.id

    # Try matching without any separators (e.g. ngochuyen == ngoc_huyen)
    clean_target = _clean(candidate)
    for v_id in VOICE_IDS:
        clean_id = _clean(v_id)
        if clean_target.endswith(clean_id) or clean_target == clean_id:
            return v_id
    for alias, v_id in OPENAI_VOICE_ALIASES.items():
        if clean_target.endswith(alias) or clean_target == alias:
            return v_id

    raise VoiceError(
        f"unknown voice {name!r}; known voices: {', '.join(VOICE_IDS)}"
    )

def describe(voice: Voice) -> dict[str, object]:
    """The JSON shape used by ``/v1/voices``."""
    return {
        "id": voice.id,
        "name": voice.label,
        "voice_id": voice.id,
        "language": voice.language,
        "gender": voice.gender,
        "style": voice.style,
    }

def catalogue() -> list[dict[str, object]]:
    return [describe(v) for v in VOICES]
