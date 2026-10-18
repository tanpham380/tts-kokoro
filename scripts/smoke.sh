#!/usr/bin/env bash
# Smoke test script for Kokoro-Vietnamese TTS server.
#
# Usage:
#   ./scripts/smoke.sh [http://127.0.0.1:8880]

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8880}"
echo "==> Testing Kokoro-Vietnamese TTS Server at: $BASE_URL"

echo ""
echo "1. Checking /health..."
curl -fsS "$BASE_URL/health" | grep -E '"status"|"model"'
echo "   [OK] /health is reachable"

echo ""
echo "2. Checking /ready..."
curl -fsS "$BASE_URL/ready" | grep -E '"ready":true|"status":"ok"'
echo "   [OK] /ready reports ready"

echo ""
echo "3. Querying /v1/models (OpenAI format)..."
curl -fsS "$BASE_URL/v1/models" | grep -E '"object":"list"'
echo "   [OK] /v1/models returned model list"

echo ""
echo "4. Querying /v1/voices (Vietnamese voicepacks)..."
VOICES_COUNT=$(curl -fsS "$BASE_URL/v1/voices" | grep -o '"id":' | wc -l | tr -d ' ')
echo "   [OK] /v1/voices returned $VOICES_COUNT voices"

echo ""
echo "5. Testing /v1/audio/speech (MP3 output)..."
HTTP_CODE=$(curl -s -o /tmp/test_speech.mp3 -w "%{http_code}" -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kokoro-Vietnamese",
    "input": "Xin chào, đây là bài kiểm tra giọng nói tiếng Việt.",
    "voice": "diem_trinh",
    "response_format": "mp3"
  }')

if [ "$HTTP_CODE" = "200" ]; then
  FILE_SIZE=$(wc -c < /tmp/test_speech.mp3 | tr -d ' ')
  echo "   [OK] Synthesized MP3 successfully (${FILE_SIZE} bytes)"
else
  echo "   [FAIL] Expected 200, got $HTTP_CODE"
  exit 1
fi

echo ""
echo "6. Testing /v1/audio/speech with OpenAI alias 'alloy'..."
HTTP_CODE=$(curl -s -o /tmp/test_alias.wav -w "%{http_code}" -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Thử nghiệm giọng đọc ánh xạ OpenAI alloy.",
    "voice": "alloy",
    "response_format": "wav"
  }')

if [ "$HTTP_CODE" = "200" ]; then
  FILE_SIZE=$(wc -c < /tmp/test_alias.wav | tr -d ' ')
  echo "   [OK] Synthesized WAV successfully via alloy alias (${FILE_SIZE} bytes)"
else
  echo "   [FAIL] Expected 200, got $HTTP_CODE"
  exit 1
fi

echo ""
echo "==> ALL SMOKE TESTS PASSED SUCCESSFULLY!"
