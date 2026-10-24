"""Encoding helpers. These never invoke ffmpeg itself."""



from __future__ import annotations

import numpy as np
import pytest

from app.audio import (
    AudioError,
    base64_chunk,
    concat,
    crossfade,
    duration_seconds,
    encode_stream_chunk,
    pcm16le_to_float32,
    stream_prefix,
    streaming_mime,
    to_pcm16le,
    wav_streaming_header,
)


def test_pcm16le_roundtrip():

    samples = np.linspace(-1.0, 1.0, 1000).astype(np.float32)

    raw = to_pcm16le(samples)

    assert isinstance(raw, bytes)

    recovered = pcm16le_to_float32(raw)

    assert recovered.shape == samples.shape

    assert np.allclose(recovered, samples, atol=1e-3)





def test_pcm16le_clips_out_of_range():

    samples = np.array([0.0, 5.0, -5.0, 1.0], dtype=np.float32)

    recovered = pcm16le_to_float32(to_pcm16le(samples))

    assert np.max(recovered) <= 1.0

    assert np.min(recovered) >= -1.0

    # The clipped sample clips to roughly +/-1.

    assert np.isclose(recovered[1], 1.0, atol=1e-3)

    assert np.isclose(recovered[2], -1.0, atol=1e-3)





def test_pcm16le_empty():

    assert to_pcm16le(np.zeros(0)) == b""

    assert len(pcm16le_to_float32(b"")) == 0





def test_base64_comes_back():

    data = b"\x01\x02\xff\x00hello"

    assert base64_chunk(data) != data





def test_crossfade_idempotent_for_zero_fade():

    a = np.arange(100, dtype=np.float32)

    b = np.arange(100, dtype=np.float32) + 100

    merged = crossfade(a, b, 0)

    assert np.array_equal(merged, np.concatenate((a, b)))





def test_crossfade_blends_overlap():
    a = np.ones(100, dtype=np.float32) * 0.0
    b = np.ones(100, dtype=np.float32) * 2.0
    merged = crossfade(a, b, 20)
    assert merged.shape == (len(a) + len(b) - 20,)
    # Overlap region samples transition smoothly from 0 to 2
    overlap = merged[80:100]
    assert np.all(overlap[1:-1] > 0.0) and np.all(overlap[1:-1] < 2.0)
    assert np.all(np.diff(overlap) >= 0)





def test_concat_empty():

    assert concat([]).size == 0

    assert len(concat([np.zeros(0)])) == 0





def test_concat_single_chunk():

    a = np.arange(10, dtype=np.float32)

    out = concat([a])

    assert np.array_equal(out, a)





def test_duration_seconds():

    assert duration_seconds(np.zeros(24000), 24000) == 1.0

    assert duration_seconds(np.zeros(0), 24000) == 0.0

    assert duration_seconds(np.zeros(100), 0) == 0.0





def test_stream_prefix_empty_for_pcm():

    assert stream_prefix("pcm", 24000) == b""

    assert stream_prefix("flac", 24000) == b""





def test_wav_streaming_header_shape():

    header = wav_streaming_header(24000, 1)

    assert len(header) == 44

    assert header[:4] == b"RIFF"

    assert header[8:12] == b"WAVE"





def test_wav_streaming_header_stereo():

    header = wav_streaming_header(24000, 2)

    assert len(header) == 44

    assert b"\x02\x00" in header[22:24]





def test_streaming_mime():

    assert streaming_mime("mp3") == "audio/mpeg"

    assert streaming_mime("wav") == "audio/wav"

    assert streaming_mime("pcm") == "audio/pcm"





def test_encode_pcm_stream():

    samples = np.ones(240, dtype=np.float32)

    raw = encode_stream_chunk(samples, "pcm", 24000)

    assert isinstance(raw, bytes)

    assert len(raw) == 240 * 2





def test_encode_stream_chunk_rejects_mp3():

    samples = np.ones(240, dtype=np.float32)

    with pytest.raises(AudioError):

        encode_stream_chunk(samples, "mp3", 24000)

