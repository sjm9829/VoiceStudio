"""사용자와 내부 계층이 공유하는 오류 계층."""

class VoiceStudioError(Exception):
    """모든 애플리케이션 오류의 기반."""
    code = "E_VOICE_STUDIO"
    user_message = "문제가 발생했습니다."

    def __init__(self, detail: str = "", *, user_message: str | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail or self.user_message
        if user_message:
            self.user_message = user_message

class FfmpegNotFoundError(VoiceStudioError):
    code = "E_FFMPEG_NOT_FOUND"
    user_message = "오디오 처리 프로그램(FFmpeg)을 찾을 수 없습니다. 설치 후 다시 시도해 주세요."

class UnsupportedAudioError(VoiceStudioError):
    code = "E_UNSUPPORTED_AUDIO"
    user_message = "지원하지 않는 오디오 파일입니다. MP3/M4A/WAV/FLAC 파일을 선택해 주세요."

class GpuUnavailableError(VoiceStudioError):
    code = "E_GPU_UNAVAILABLE"
    user_message = "NVIDIA 그래픽 카드를 사용할 수 없습니다. 설정에서 진단 정보를 확인해 주세요."

class ModelNotDownloadedError(VoiceStudioError):
    code = "E_MODEL_NOT_DOWNLOADED"
    user_message = "음성 모델이 아직 받아지지 않았습니다. 설정에서 모델을 받아 주세요."

class OfflineError(VoiceStudioError):
    code = "E_OFFLINE"
    user_message = "인터넷에 연결할 수 없고 모델도 저장되어 있지 않습니다. 연결 후 다시 시도해 주세요."

class ProfileError(VoiceStudioError):
    code = "E_PROFILE"
    user_message = "목소리 정보를 처리하는 중 문제가 발생했습니다."

class DuplicateNameError(ProfileError):
    code = "E_PROFILE_DUPLICATE_NAME"
    user_message = "같은 이름의 목소리가 이미 있습니다. 다른 이름을 사용해 주세요."

class ConsentRequiredError(ProfileError):
    code = "E_PROFILE_CONSENT_REQUIRED"
    user_message = "사용 권한 확인에 체크한 뒤 등록할 수 있습니다."

class TranscriptRequiredError(ProfileError):
    code = "E_PROFILE_TRANSCRIPT_REQUIRED"
    user_message = "참조 음성의 대사를 입력해야 등록할 수 있습니다."

class WorkerError(VoiceStudioError):
    code = "E_WORKER"
    user_message = "음성 생성 작업 중 오류가 발생했습니다."

class WorkerCrashedError(WorkerError):
    code = "E_WORKER_CRASHED"
    user_message = "음성 생성 작업이 비정상 종료되었습니다. 다시 시도해 주세요."

class CancelledError(WorkerError):
    code = "E_CANCELLED"
    user_message = "작업을 취소했습니다."

class DiskSpaceError(WorkerError):
    code = "E_DISK_SPACE"
    user_message = "저장 공간이 부족합니다. 공간을 확보한 뒤 다시 시도해 주세요."
