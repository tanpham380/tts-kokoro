# Kokoro-Vietnamese TTS Server (FastAPI & GPUStack Compatible)

> 💡 **Lưu ý phân biệt hệ thống**: Dự án này là **Text-to-Speech (TTS Engine)** – chuyển đổi văn bản thành giọng nói tiếng Việt, hoàn toàn độc lập với dự án `quan-asr` (vốn là hệ thống **Speech-to-Text / STT / ASR**).

Máy chủ Text-to-Speech (TTS) hiệu năng cao, chuẩn hóa 100% theo API OpenAI (`POST /v1/audio/speech`), tích hợp mô hình **[Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese)** (`contextboxai/Kokoro-Vietnamese`) cùng biến thể fine-tune **[dinhthuan/kokoro-vi-ngoc-huyen](https://huggingface.co/dinhthuan/kokoro-vi-ngoc-huyen)**, hỗ trợ 14 giọng đọc tiếng Việt mượt mà và tương thích trực tiếp với nền tảng quản lý mô hình **GPUStack**.

---

## 🔬 Phân Tích & Giải Pháp Kỹ Thuật Cho Engine Thực Tế (Production-Grade)

Khi đưa một mô hình nghiên cứu (như Kokoro-Vietnamese) vào phục vụ thực tế (serving production) tương tự **SpeakCrate** và **Kokoro-FastAPI**, hệ thống đã giải quyết triệt để các bài toán cốt lõi sau:

### 1. Xử lý triệt để Issue #4: Lỗi phát âm chữ cái đơn (`Y` -> "quai")
- **Nguyên nhân gốc rễ**: Khi văn bản có các ký tự đơn lẻ đứng độc lập (như "Khoa Y", "Bác sĩ Y", "nhóm máu A, B, O", "Vitamin C", "Gen Y"), bộ phonemizer fallback tiếng Anh (Misaki/eSpeak) sẽ nhận diện đây là bảng chữ cái tiếng Anh và phát âm thành tên chữ cái tiếng Anh ("Y" thành "quai/wai", "A" thành "ei", "O" thành "âu").
- **Giải pháp trong engine**: Xây dựng tầng tiền xử lý [`normalize_vietnamese_text`](file:///Users/apple/Desktop/dev/tts-kokoro/app/engine.py#L90) với bộ quy tắc regex ranh giới từ `STANDALONE_LETTER_FIXES`:
  - `\bY\b`, `\by\b` ➔ `i` (đọc chuẩn âm /i/ của tiếng Việt, ví dụ "khoa Y" đọc là "khoa i").
  - `\bA\b` ➔ `a`, `\bB\b` ➔ `bê`, `\bC\b` ➔ `xê`, `\bD\b` ➔ `dê`, `\bO\b` ➔ `o`, `\bU\b` ➔ `u`.
  - Mở rộng chuẩn hóa viết tắt thông dụng: `TP.HCM` ➔ `Thành phố Hồ Chí Minh`, `VN` ➔ `Việt Nam`, `15km` ➔ `15 ki lô mét`, `100k` ➔ `100 nghìn`, `100%` ➔ `100 phần trăm`.

### 2. Tích hợp sâu G2P Tiếng Việt (`vig2p`)
- Tích hợp trực tiếp thư viện **`vig2p`** để đảm bảo phân tách âm tiết tiếng Việt, giữ vững độ tương phản âm đầu (`t/th`, `tr/ch`, `s/x`, `d/gi`) và hệ thống thanh điệu 6 thanh chuẩn của tiếng Việt.

### 3. Xử lý triệt để Issue #3: Tiếng rè và click giật xen kẽ giữa các câu
- **Nguyên nhân**: Khi đọc văn bản dài, mô hình tách thành nhiều câu nhỏ để giải mã. Nếu chỉ ghép mảng đơn thuần (`np.concatenate`), bước nhảy pha (phase jump) và độ lệch biên độ tại điểm nối tạo ra các tiếng "click/pop" và tiếng rè rẹt khó chịu.
- **Giải pháp**: Áp dụng thuật toán **Dynamic Linear Crossfade (50ms)** tại [`app/audio.py:crossfade`](file:///Users/apple/Desktop/dev/tts-kokoro/app/audio.py#L213). Hai câu kế tiếp được làm mờ chéo mượt mà, triệt tiêu hoàn toàn tiếng rè giật.

### 4. Tương thích mô hình fine-tune `dinhthuan/kokoro-vi-ngoc-huyen`
- Tác giả Đinh Thuận đã huấn luyện lại riêng một checkpoint sạch cho giọng Ngọc Huyền tại `dinhthuan/kokoro-vi-ngoc-huyen` để khắc phục lỗi rè của voicepack gốc.
- Engine hỗ trợ cơ chế tự động tìm kiếm thông minh: hỗ trợ cả cấu trúc của repo gốc `contextboxai/Kokoro-Vietnamese` (`kokoro_vi.pth`, `voicepacks/*.pt`) lẫn cấu trúc của `dinhthuan/kokoro-vi-ngoc-huyen` (`model/kokoro_vi_ngoc_huyen.pth`, `voices/*.pt`).

### 5. Tối ưu hóa bộ nhớ RAM/VRAM (Shared Model Architecture)
- Học hỏi từ `Kokoro-FastAPI` và `speakcrate`: Mô hình neural network Kokoro 82M (`KModel`) chỉ cần nạp **duy nhất 1 bản** vào RAM/VRAM. Cả 14 giọng nói tiếng Việt chỉ là các vector đặc trưng phong cách (`voicepack` ~200KB).
- Thay vì nhân bản 14 model gây cạn kiệt VRAM, kiến trúc [`VoiceRuntime`](file:///Users/apple/Desktop/dev/tts-kokoro/app/engine.py#L125) tái sử dụng model dùng chung và hoán đổi voicepack tức thì trong vài micro-giây.

### 6. Streaming kép chuẩn OpenAI & Raw Audio
- **SSE Stream (`stream_format=sse`)**: Bắn sự kiện Server-Sent Events chuẩn OpenAI `speech.audio.delta` dùng cho các Web Chatbot (Open WebUI, LibreChat, Dify).
- **Chunked Byte Stream (`stream_format=audio` hoặc `stream=true`)**: Truyền phát luồng byte âm thanh trực tiếp cho trình duyệt và các ứng dụng phát audio Native.

---

## 🧠 Bản chất kiến trúc: Kokoro là Model hay Engine? Lấy từ đâu và ta viết gì?

1. **Kokoro là gì?**
   - **Kokoro-82M** là một mô hình Neural Text-to-Speech mã nguồn mở siêu nhẹ (82 triệu tham số) do tác giả **hexgrad** phát triển, xây dựng trên kiến trúc **StyleTTS 2 + iSTFTNet vocoder**. Điểm mạnh của nó là chất lượng phát âm tự nhiên vượt trội nhưng kích thước cực nhỏ, có thể chạy real-time trên cả CPU (< 0.1 RTF).
2. **Kokoro-Vietnamese lấy từ đâu?**
   - Do cộng đồng AI Việt Nam (**`contextboxai`** và tác giả **`iamdinhthuan` - Đinh Thuận**) tinh chỉnh (fine-tune) trên tập dữ liệu giọng đọc tiếng Việt, kết hợp từ điển âm vị tiếng Việt + công cụ G2P (`vig2p`) và trích xuất 14 vector đặc trưng phong cách (`voicepacks/*.pt`).
3. **Engine `tts-kokoro` này lấy gì và viết những gì?**
   - **Ta KHÔNG viết lại từ đầu mô hình Neural Network**: Ta kế thừa toàn bộ trọng số (weights) đã được huấn luyện sẵn từ `contextboxai/Kokoro-Vietnamese` và `dinhthuan/kokoro-vi-ngoc-huyen` (tiết kiệm hàng ngàn giờ huấn luyện và chi phí GPU cluster).
   - **Ta XÂY DỰNG TOÀN BỘ Serving Engine & Enterprise Gateway**:
     - **Học hỏi kiến trúc serving** từ `remsky/Kokoro-FastAPI` và `hwdsl2/speakcrate`.
     - **Viết bộ xử lý Singleton VoiceRuntime**: Thay vì tạo 14 model riêng biệt gây tràn VRAM, ta chia sẻ duy nhất 1 transformer backbone và hoán đổi style vector 200KB tức thì.
     - **Viết bộ Text Normalizer sửa triệt để Issue #4**: Khắc phục hiện tượng chữ cái độc lập bị tiếng Anh hoá (`Y` đọc là "quai"), số tiền, đơn vị đo, viết tắt.
     - **Viết thuật toán Dynamic Crossfade sửa triệt để Issue #3**: Khắc phục tiếng click/pop và rè giữa các câu khi ghép mảng audio.
     - **Tương thích 100% GPUStack**: Xử lý tham số CLI, probe `/health`, `/ready`, `/live`, nạp local volume mount không cần Internet và hỗ trợ toàn bộ tham số trong GPUStack Audio Playground.

---

## 🎛️ Bảng tham số tương thích GPUStack TTS Playground

Khi người dùng kiểm thử model TTS trong **GPUStack Web UI Playground** hoặc gọi API, GPUStack sẽ gửi một payload chứa các tham số sau tới `POST /v1/audio/speech`. Engine `tts-kokoro` hỗ trợ đầy đủ 100%:

| Tham số trên GPUStack UI | Field trong Request JSON | Kiểu dữ liệu | Vai trò & Cách engine `tts-kokoro` xử lý |
|---|---|---|---|
| **Model** ﹡ | `model` | `string` | Tên model cần tổng hợp (ví dụ `Kokoro-Vietnamese`, `tts-1`, hoặc `{{model_name}}`). |
| **Voice** | `voice` | `string` | Giọng đọc tiếng Việt (`diem_trinh`, `ngoc_huyen`, `duc_duy`...) hoặc OpenAI alias (`alloy`, `nova`...). |
| **Format** | `format` / `response_format` | `string` | Định dạng xuất: `mp3`, `wav`, `opus`, `aac`, `flac`, `pcm`. Engine tự động map alias `format` ➔ `response_format`. |
| **Enable Streaming** | `enable_streaming` / `stream` | `boolean` | Bật truyền phát audio theo câu. Engine map alias `enable_streaming` ➔ `stream`. |
| **Task Type** (Nâng cao) | `task_type` | `string` | Chế độ tác vụ của GPUStack (`CustomVoice`, `ZeroShot`, `TTS`). Engine ghi nhận và log định tuyến. |
| **Language** (Nâng cao) | `language` | `string` | Ngôn ngữ phát âm: `vi` (mặc định), `en`, `auto`. Nếu là `vi`/`auto`, kích hoạt bộ normalizer tiếng Việt. |
| **Instructions** (Nâng cao) | `instructions` | `string` | Chỉ dẫn phong cách / cảm xúc (e.g. "Vui vẻ", "Hào hứng", "Thì thầm"). |
| **Max New Tokens** (Nâng cao) | `max_new_tokens` | `integer` | Giới hạn số token sinh tối đa (e.g. 2048). Engine xác thực `> 0`. |
| **Reference Audio** (Nâng cao) | `reference_audio` / `ref_audio` | `string` | Đường dẫn file hoặc Base64 audio mẫu. Nếu trỏ tới voicepack `.pt`, engine hỗ trợ tải giọng riêng. |
| **Transcript of Reference Audio** | `transcript_of_reference_audio` / `ref_text` | `string` | Lời thoại tương ứng của audio mẫu (chế độ In-Context Learning). **Đã fix GPUStack Issue #4565** (xử lý chuỗi rỗng `""` thành `None` an toàn). |
| **Use Speaker Embedding Only** | `use_speaker_embedding_only` | `boolean` | Cờ chỉ định dùng speaker embedding (style vector) thay vì prompt in-context learning. |

---

## 🌟 Điểm nổi bật & Tính năng chính

1. **Chuẩn OpenAI 100% (Drop-in Replacement)**:
   - Endpoint chính: `POST /v1/audio/speech`
   - Tra cứu danh sách model: `GET /v1/models`
   - Tra cứu danh sách voice: `GET /v1/voices` và `GET /v1/audio/voices`
   - Tương thích tức thì với các thư viện chính thức: `openai-python`, `openai-node`, Open WebUI, Dify, LangChain, v.v.

2. **14 Voicepacks tiếng Việt & Ánh xạ OpenAI Aliases**:
   - **Giọng Nữ**: `diem_trinh` (mặc định), `mai_linh`, `mai_loan`, `my_yen`, `ngoc_huyen`, `thuc_trinh`.
   - **Giọng Nam**: `duc_an`, `duc_duy`, `hung_thinh`, `manh_dung`, `phat_tai`, `thanh_dat`, `tuan_ngoc`.
   - **Giọng Kể chuyện / Audio book**: `storyvert`.
   - **OpenAI Voice Aliases**: Tự động ánh xạ `alloy`, `nova`, `onyx`, `echo`, `shimmer`, `fable`, `coral`, v.v. sang giọng Việt tương ứng theo giới tính.

3. **Tương thích hoàn hảo với GPUStack**:
   - Cung cấp sẵn manifest `deploy/gpustack-backend.yaml` và `deploy/gpustack-backend-source.yaml`.
   - Khả năng tự động nhận diện cả **Hugging Face Repo ID** và **Local Directory Mount** (`{{model_path}}`). Khi GPUStack tải sẵn weights về máy chủ, server nhận diện và nạp trực tiếp không cần mạng.
   - Hỗ trợ 2 framework profile: `cuda` (NVIDIA GPU) và `cpu` (CPU đa luồng).
   - Đầy đủ probe kiểm tra trạng thái: `/health`, `/ready` (503 khi đang tải, 200 khi sẵn sàng), `/live`.
   - Hỗ trợ toàn diện các trường từ GPUStack Playground: `task_type`, `language`, `instructions`, `max_new_tokens`, `reference_audio`, `transcript_of_reference_audio`, `use_speaker_embedding_only`.

4. **Web Studio Playground Hiện đại (`/`)**:
   - Giao diện Dark-mode glassmorphic hiện đại, hiển thị trạng thái model, thời gian tạo âm thanh (latency ms), mẫu câu thử nghiệm, trình phát âm thanh trực tiếp.
   - Tích hợp sẵn mục **Tham số Nâng cao (GPUStack Playground Compatible)** để người dùng thử nghiệm trực tiếp ngay trên trình duyệt.

---

## 📂 Cấu trúc mã nguồn

```
tts-kokoro/
├── app/
│   ├── __init__.py         # Phiên bản máy chủ
│   ├── audio.py            # Xử lý PCM, container encoding (ffmpeg / soundfile), crossfade, streaming headers
│   ├── cli.py              # CLI entrypoint hỗ trợ cờ GPUStack (--model-path, --host, --port, v.v.)
│   ├── config.py           # Quản lý cấu hình qua biến môi trường Settings
│   ├── engine.py           # Quản lý vòng đời KokoroVietnamese, nạp voicepack, sentence decode & concurrency gate
│   ├── schemas.py          # Pydantic schemas chuẩn OpenAI Request / Error Envelope
│   ├── server.py           # FastAPI routes (/v1/audio/speech, /v1/models, /v1/voices, /health, Studio UI)
│   └── voices.py           # Danh mục 14 giọng Việt, ánh xạ OpenAI alias, chuẩn hóa tên
├── deploy/
│   ├── gpustack-backend.yaml        # GPUStack Custom Inference Backend manifest
│   └── gpustack-backend-source.yaml # GPUStack InferenceBackendSource definition
├── docker/
│   └── docker-compose.yaml # Docker compose hỗ trợ profile CPU và GPU
├── scripts/
│   ├── smoke.sh            # Script kiểm thử nhanh toàn bộ endpoints
│   └── test_openai_client.py # Mẫu gọi qua thư viện openai chính thức
├── tests/                  # Test suite với 55 unit tests (100% pass)
├── Dockerfile              # Production multi-stage Dockerfile cho CPU (sử dụng uv)
├── Dockerfile.gpu          # Production multi-stage Dockerfile cho CUDA GPU (sử dụng uv)
├── pyproject.toml          # Định nghĩa gói Python và dependencies
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Khởi chạy cục bộ với Python

```bash
# Cài đặt dependencies (yêu cầu Python >= 3.10)
pip install -e .

# Đảm bảo hệ thống có ffmpeg và espeak-ng:
# Trên macOS: brew install ffmpeg espeak-ng
# Trên Ubuntu/Debian: apt-get install ffmpeg libsndfile1 espeak-ng espeak-ng-data

# Khởi chạy server:
kokoro-vi-server --port 8880
```

Truy cập Web Studio tại: `http://localhost:8880/`

---

### 2. Khởi chạy bằng Docker & Docker Compose

Dự án cung cấp 2 Dockerfile tối ưu hóa kích thước bằng **Astral uv**:

#### Cách 1: Docker Compose

```bash
# Khởi chạy bản CPU (Port 8881)
docker compose --profile cpu up -d

# Khởi chạy bản NVIDIA GPU (Port 8882)
docker compose --profile gpu up -d
```

#### Cách 2: Docker Build thủ công

```bash
# Build bản GPU CUDA
docker build -f Dockerfile.gpu -t kokoro-vi-server:gpu .

# Chạy bản GPU
docker run -d --gpus all -p 8880:8880 \
  -v ~/.cache/huggingface:/cache/huggingface \
  -e DEVICE=cuda \
  kokoro-vi-server:gpu
```

---

## 🧩 Tích hợp GPUStack

Dự án được thiết kế chuẩn xác để kết nối vào cụm GPUStack:

1. **Import Backend vào GPUStack**:
   - Truy cập giao diện quản trị GPUStack: **Inference Backends** -> Nhấn **Add Backend**.
   - Chọn chế độ **YAML** và dán nội dung từ tệp [`deploy/gpustack-backend.yaml`](file:///Users/apple/Desktop/dev/tts-kokoro/deploy/gpustack-backend.yaml).
   - Tên backend đăng ký là `Kokoro-TTS-Vietnamese-custom`.

2. **Triển khai Model trong GPUStack**:
   - Vào mục **Models** -> Nhấn **Deploy Model**.
   - Chọn **Category**: `Audio` / `Text to Speech` (hoặc Custom).
   - **Backend**: Chọn `Kokoro-TTS-Vietnamese-custom`.
   - **Hugging Face Model**: `contextboxai/Kokoro-Vietnamese`.
   - GPUStack sẽ tự động tải model về worker, gắn mount thư mục vào `{{model_path}}` và chạy lệnh:
     ```bash
     kokoro-vi-server --model-path {{model_path}} --served-model-name {{model_name}} --host {{worker_ip}} --port {{port}}
     ```
   - Server lập tức chuyển trạng thái `ready` (qua probe `/health` và `/ready`).

---

## 💻 Mẫu gọi API chuẩn OpenAI

### 1. Gọi bằng cURL (REST API)

```bash
# Tạo file MP3 tiếng Việt
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kokoro-Vietnamese",
    "input": "Xin chào! Đây là bài kiểm tra giọng nói tiếng Việt mượt mà với Kokoro.",
    "voice": "diem_trinh",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output tieng_viet.mp3
```

### 2. Sử dụng OpenAI Python SDK

```python
from openai import OpenAI

# Trỏ base_url tới server Kokoro
client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="Kokoro-Vietnamese",
    voice="mai_linh",  # Hoặc alias: "alloy", "nova", "onyx", v.v.
    input="Hôm nay thời tiết Hà Nội và Sài Gòn đều rất đẹp, chúc bạn một ngày làm việc hiệu quả!",
    response_format="mp3",
    speed=1.05
)

response.stream_to_file("output.mp3")
```

### 3. Gọi Streaming trực tiếp (Giảm độ trễ phát âm)

```python
response = client.audio.speech.create(
    model="Kokoro-Vietnamese",
    voice="thanh_dat",
    input="Từng câu văn sẽ được chuyển thành âm thanh ngay khi mô hình xử lý xong.",
    response_format="wav",
    stream=True
)

response.stream_to_file("stream_output.wav")
```

---

## 🎙️ Bảng đối chiếu Giọng đọc & OpenAI Aliases

| Giọng Tiếng Việt | Tên hiển thị | Giới tính | Phong cách | OpenAI Alias |
|---|---|---|---|---|
| `diem_trinh` | Diễm Trinh | Nữ (Female) | Dẫn truyện (Narration) | `alloy` |
| `mai_linh` | Mai Linh | Nữ (Female) | Tự nhiên, nhẹ nhàng | `coral` |
| `mai_loan` | Mai Loan | Nữ (Female) | Truyền cảm | `nova` |
| `my_yen` | Mỹ Yến | Nữ (Female) | Tươi sáng | `marin` |
| `ngoc_huyen` | Ngọc Huyền | Nữ (Female) | Ấm áp | `sage` |
| `thuc_trinh` | Thục Trinh | Nữ (Female) | Điềm tĩnh | `shimmer` |
| `duc_an` | Đức An | Nam (Male) | Trầm ấm | `ash` |
| `duc_duy` | Đức Duy | Nam (Male) | Tự nhiên, rõ ràng | `echo` |
| `hung_thinh` | Hùng Thịnh | Nam (Male) | Dứt khoát | `verse` |
| `manh_dung` | Mạnh Dũng | Nam (Male) | Đĩnh đạc | `onyx` |
| `phat_tai` | Phát Tài | Nam (Male) | Năng động | - |
| `thanh_dat` | Thành Đạt | Nam (Male) | Trẻ trung | `cedar` |
| `tuan_ngoc` | Tuấn Ngọc | Nam (Male) | Trầm ấm | `ballad` |
| `storyvert` | Storyvert | Trung tính (Neutral) | Sách nói (Audiobook) | `fable` |

---

## ⚙️ Các biến môi trường tùy chỉnh (Environment Variables)

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MODEL_PATH` | `contextboxai/Kokoro-Vietnamese` | HF repo id hoặc đường dẫn thư mục model cục bộ |
| `SERVED_MODEL_NAME` | `Kokoro-Vietnamese` | Tên model công bố trên `/v1/models` |
| `DEFAULT_VOICE` | `diem_trinh` | Giọng đọc mặc định khi client không chỉ định |
| `DEVICE` | `auto` (hoặc `cuda`, `cpu`, `mps`) | Thiết bị phần cứng giải mã âm thanh |
| `PRELOAD` | `true` | Nạp trước model vào RAM/VRAM khi khởi động container |
| `WARMUP` | `true` | Chạy câu khởi động làm ấm kernel CUDA |
| `MAX_CONCURRENCY` | `1` | Số lượt tổng hợp âm thanh thực thi song song |
| `MAX_QUEUE` | `256` | Giới hạn hàng đợi trước khi trả về HTTP 503 |
| `QUEUE_TIMEOUT` | `60` | Thời gian chờ tối đa trong hàng đợi (giây) |
| `CROSSFADE_MS` | `50` | Thời gian nối âm giữa 2 câu (mili-giây) |
| `PORT` | `8880` | Cổng HTTP lắng nghe |
| `API_KEY` | `None` | Khóa xác thực Bearer token (nếu cần bảo mật) |

---

## 🧪 Kiểm thử (Testing)

Chạy bộ kiểm thử tự động với 55 bài test bao phủ các thành phần engine, âm thanh, cấu hình, voice mapping và FastAPI routes:

```bash
python3 -m pytest tests/
```
