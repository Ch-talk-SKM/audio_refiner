import torch
import torchaudio

def remove_background_music(input_path: str, output_path: str):
    """
    단일 WAV 오디오 파일에서 Hybrid Demucs 모델을 사용하여 배경 음악을 제거하고 음성만 추출합니다.
    """
    # 디바이스 설정 (CUDA 사용 가능 시 GPU, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hybrid Demucs 모델 로드 (사전 학습된 파이프라인 사용)
    try:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as HDemucsBundle
    except ImportError as e:
        raise RuntimeError("Torchaudio 최신 버전이 필요합니다. Hybrid Demucs 모델을 불러올 수 없습니다.") from e

    # 모델 인스턴스 생성 및 가중치 로드
    try:
        bundle = HDemucsBundle  # MUSDB-HQ 기반 사전학습 모델 (고음질 버전)
        model = bundle.get_model().to(device)
        model.eval()  # 추론 모드
    except Exception as e:
        raise RuntimeError("Hybrid Demucs 모델 로드에 실패하였습니다: {}".format(str(e))) from e

    # 입력 WAV 오디오 로드
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        # waveform shape: (채널 수, 시간 샘플)
    except Exception as e:
        raise RuntimeError(f"입력 오디오 파일을 불러올 수 없습니다: {e}")

    # 모델이 기대하는 샘플링 레이트로 리샘플링 (필요한 경우)
    target_sr = getattr(bundle, "sample_rate", None)
    if target_sr is None:
        target_sr = 16000  # bundle에 sample_rate 속성이 없으면 16000Hz로 가정
    if sample_rate != target_sr:
        try:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr
        except Exception as e:
            raise RuntimeError(f"리샘플링 실패: {e}")

    # 채널 수 맞추기: 모델은 스테레오(2채널) 입력을 기대. 모노인 경우 2채널로 복제
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # (1, N) -> (2, N)
    elif waveform.shape[0] > 2:
        # 2채널 초과일 경우 앞의 2채널만 사용 (필요에 따라 믹싱하거나 에러 처리 가능)
        waveform = waveform[:2, :]
    
    # 배치 차원 추가: 모델 입력 형상 (batch, channels, time)
    waveform = waveform.unsqueeze(0)  # (채널, 시간) -> (1, 채널, 시간)
    
    # GPU로 데이터 이동
    waveform = waveform.to(device)
    
    # 추론 (Gradients 비활성화)
    try:
        with torch.no_grad():
            sources = model(waveform)  # 분리된 음원들 출력
    except Exception as e:
        raise RuntimeError(f"모델 추론 중 오류 발생: {e}")
    finally:
        # 메모리 해제 (필요한 경우)
        waveform = waveform.cpu()
    
    # 출력 처리: 모델의 출력 소스 목록에서 "vocals" (음성) 트랙 추출
    # 모델 출력 `sources`가 Tensor인 경우 (batch, source_index, channel, time)
    if isinstance(sources, torch.Tensor):
        sources = sources.cpu()           # CPU로 이동
        if sources.dim() == 4:            # 형상: (batch, sources, channels, time)
            sources = sources.squeeze(0)  # batch 차원 제거 -> (sources, channels, time)
        # sources 목록에 해당하는 소스 이름 얻기
        source_names = getattr(model, "sources", None)
        if source_names is None:
            # model.sources 속성이 없을 경우, Demucs 기본 순서로 가정
            source_names = ["drums", "bass", "other", "vocals"]
        try:
            # 소스 이름과 텐서를 매핑하여 딕셔너리 생성
            source_dict = {name: sources[idx] for idx, name in enumerate(source_names)}
            vocals_waveform = source_dict.get("vocals")
        except Exception as e:
            raise RuntimeError(f"모델 출력 처리 실패: {e}")
    else:
        # 예상치 못한 출력 타입 (Tensor가 아닌 경우)
        raise RuntimeError(f"알 수 없는 모델 출력 타입: {type(sources)}")
    
    if vocals_waveform is None:
        raise RuntimeError("분리된 음성 트랙을 찾을 수 없습니다 (\"vocals\" 키 없음).")
    
    # 음성 트랙을 WAV 파일로 저장
    try:
        # torchaudio.save는 (channels, time) 형상의 텐서를 요구
        vocals_waveform = vocals_waveform.clone().detach()  # Tensor 보존
        if vocals_waveform.dim() == 1:
            vocals_waveform = vocals_waveform.unsqueeze(0)  # (time) -> (1, time)
        torchaudio.save(output_path, vocals_waveform, sample_rate)
    except Exception as e:
        raise RuntimeError(f"출력 파일 저장 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"사용법: python {sys.argv[0]} <input WAV path> <output WAV path>")
    else:
        inp, out = sys.argv[1], sys.argv[2]
        try:
            remove_background_music(inp, out)
            print(f"배경 음악 제거 완료: '{out}' 저장되었습니다.")
        except Exception as err:
            print(f"오류 발생: {err}")
