# AudioRefinery: Audio Data Cleaning Toolkit for Conversational AI

**AudioRefinery**는 유튜브 및 팟캐스트에서 수집한 오디오 데이터를 Conversational AI 시스템을 위한 고품질 학습 DB로 정제하기 위한 **올인원 파이프라인**입니다.

현재 **배경 음악 제거** 기능이 구현되어 있으며, 추가로 **화자 분할**, **음성 구간 검출(VAD)**, **오디오 업스케일링** 기능이 구현될 예정입니다.

---

## ⚙️ 기능 구성 (Roadmap)

- ✅ **(1) 배경 음악 분리** (*구현 완료*)
  - Hybrid Demucs 모델로 음성+배경음악 혼합 오디오에서 배경 음악 제거
  - [main_separation.py](./main_separation.py)

- ⬜ **(2) Speaker Diarization** (*To-Do*)
  - 다중 화자 대화에서 화자별 구간 자동 태깅 (Duplex 환경 최적화)
  - 화자별 학습 데이터 구축 지원

- ⬜ **(3) Voice Activity Detection (VAD)** (*To-Do*)
  - 오디오에서 음성 구간만 검출하여 무음·배경 음악만 있는 구간 제외
  - 데이터 정제 효율성 극대화

- ⬜ **(4) 오디오 품질 업스케일링** (*To-Do*)
  - 저음질 오디오 데이터의 품질 향상 (명료도 향상, 고주파 성분 복원 등)
  - 최적의 오디오 품질로 데이터 활용성 강화

---

## 🚀 현재 구현된 기능: 배경 음악 제거

Hybrid Demucs 모델을 활용하여 WAV 오디오 파일에서 **배경 음악을 제거**하고 음성만 추출합니다.

### 📌 사용 방법

```bash
python main_separation.py <input.wav> <output.wav>
```

예시:
```bash
python main_separation.py mixed_audio.wav clean_voice.wav
```

### 📚 Dependencies

- Python 3.8+
- PyTorch 1.12+
- torchaudio 0.12+ (Hybrid Demucs 지원)


### ✅ 지원 환경

- GPU (CUDA) 환경 권장
- CPU에서도 동작하지만 처리 시간이 길어질 수 있습니다.

---

## 🌟 기여 방법

기능 추가 또는 버그 수정은 언제든 환영입니다!

1. 이 레포를 Fork 후 기능별로 새로운 브랜치를 만들어주세요.
2. PR 전 로컬 환경에서 충분히 테스트해 주세요.
3. PR을 보내시면 코드 리뷰 후 반영하겠습니다.

---
