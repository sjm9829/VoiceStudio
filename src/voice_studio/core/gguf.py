"""P17-C GGUF 고정 상수. 임의 latest 금지 - 아래 revision/tag만 사용한다.

모델: ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF @ ca27d74bc954b73dadab5b71ca265d87fc861a7c
엔진: ggml-org/llama.cpp release b11540 (Windows cuda-12.4-x64, CC 7.5 Turing 지원)
해시는 HF LFS sha256(git revision 고정)와 GitHub release asset digest 기준이다.
"""

GGUF_MODEL_REPO = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
GGUF_MODEL_REVISION = "ca27d74bc954b73dadab5b71ca265d87fc861a7c"
GGUF_MAIN_FILE = "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
GGUF_MMPROJ_FILE = "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
GGUF_MAIN_SHA256 = "ac7931aeb2e7aad1a6ed6602d353a5679c9d096b18ce8204ac730a8408d572e1"
GGUF_MMPROJ_SHA256 = "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2"
GGUF_MAIN_SIZE = 1847874400
GGUF_MMPROJ_SIZE = 446422912

LLAMA_RELEASE_TAG = "b11540"
LLAMA_WIN_CUDA_ASSET = "llama-b11540-bin-win-cuda-12.4-x64.zip"
LLAMA_CUDART_ASSET = "cudart-llama-bin-win-cuda-12.4-x64.zip"
LLAMA_TTS_BINARY = "llama-tts"  # Windows 실행확장자 .exe는 런타임에 결정

LLAMA_WIN_CUDA_SHA256 = "5103995b75db5538f17a51823e997561a2036b05ef7545c8aa424e916ff22485"
LLAMA_CUDART_SHA256 = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"

SUPPORTED_TTS_LANGS = ("ko", "en", "zh", "ja", "de", "it", "pt", "es", "fr", "ru")
