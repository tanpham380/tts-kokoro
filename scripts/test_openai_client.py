#!/usr/bin/env python3
"""Example script demonstrating OpenAI SDK integration with Kokoro-Vietnamese TTS.

Requirements:
    pip install openai

Usage:
    python scripts/test_openai_client.py [BASE_URL]
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package not installed. Run: pip install openai")
    sys.exit(1)


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8880/v1"
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    print(f"Connecting to OpenAI-compatible TTS at: {base_url}")
    client = OpenAI(base_url=base_url, api_key=os.environ.get("API_KEY", "not-needed"))

    # 1. Non-streaming synthesis
    print("\n[1] Synthesizing audio (batch mode)...")
    response = client.audio.speech.create(
        model="Kokoro-Vietnamese",
        voice="diem_trinh",
        input="Xin chào quý khách! Hệ thống chuyển đổi văn bản thành giọng nói tiếng Việt đang hoạt động rất tốt.",
        response_format="mp3",
        speed=1.0,
    )
    output_batch = "sample_batch.mp3"
    response.stream_to_file(output_batch)
    print(f" -> Saved batch audio to {output_batch} ({os.path.getsize(output_batch)} bytes)")

    # 2. Streaming synthesis
    print("\n[2] Synthesizing audio with stream=True...")
    stream_response = client.audio.speech.create(
        model="Kokoro-Vietnamese",
        voice="thanh_dat",
        input="Đây là thử nghiệm truyền phát âm thanh trực tiếp theo từng câu văn bản.",
        response_format="wav",
        speed=1.05,
    )
    output_stream = "sample_stream.wav"
    stream_response.stream_to_file(output_stream)
    print(f" -> Saved streamed audio to {output_stream} ({os.path.getsize(output_stream)} bytes)")

    print("\n[OK] OpenAI Python SDK test completed successfully!")


if __name__ == "__main__":
    main()
