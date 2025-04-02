# AudioRefinery: Audio Data Cleaning Toolkit for Conversational AI

**AudioRefinery**는 유튜브 및 팟캐스트에서 수집한 오디오 데이터를 Conversational AI 시스템을 위한 고품질 학습 DB로 정제하기 위한 **올인원 파이프라인**입니다.

---

## ⚙️ 기능 구성 (Roadmap)

- ✅ **(1) 배경 음악 분리** (*구현 완료*)
  - Hybrid Demucs 모델로 음성+배경음악 혼합 오디오에서 배경 음악 제거
  - [main_separation.py](./main_separation.py)

- ✅ **(2) Speaker Diarization** (*구현 완료*)
  - 다중 화자 대화에서 화자별 구간 자동 태깅
  - 화자별 Whisper ASR까지 연결해 최종 자막/스크립트까지 생성
  - [main_custom_diarization.py](./main_custom_diarization.py)

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

---

## 🚀 현재 구현된 기능: 화자 분할(Speaker Diarization)

**`main_custom_diarization.py`** 스크립트를 통해 **화자 구간 분리**와 **자동 음성 인식(ASR)**을 한 번에 수행할 수 있습니다.

### 📌 주요 특징

- **다중 임베딩(Multi-model Embedding) + Fusion**  
  - Pyannote, SpeechBrain 등 **여러 스피커 임베딩 모델**을 동시에 활용  
  - 스코어(유사도) 기반 평균화, 또는 피처(벡터) 레벨 합치기 등 **유연한 퓨전** 가능  

- **다양한 클러스터링(AHC, HDBSCAN, Spectral) & 앙상블(DOVER-lap 등)**  
  - AHC, HDBSCAN, Spectral Clustering 등 여러 알고리즘 결과를 **동시에** 얻고  
  - 스코어 기반(score-level) 또는 **DOVER-lap** 방식으로 **최종 앙상블** 수행  

- **추가 지표(Purity, NMI) 평가**  
  - DER, JER(전통적 지표) 외에도 **Purity**, **NMI**와 같은 군집 품질 지표 제공  
  - **화자 분할 성능**을 여러 각도에서 평가 가능  

### 📚 Dependencies

- Python 3.8+
- PyTorch 1.12+
- `pyannote.audio`, `speechbrain`, `openai-whisper`, `dover-lap`, `hdbscan`, `scikit-learn`, `librosa`, `soundfile`
  
  ```bash
  pip install pyannote.audio speechbrain openai-whisper dover-lap hdbscan scikit-learn librosa soundfile
  ```

### ⚙️ 사용 방법

1. **Config 파일 준비(옵션)**  
   - `conf.yaml` 등의 설정 파일에서 VAD 모델, 임베딩 모델, 클러스터링 방법, Whisper ASR 사용 여부 등 **세부 파라미터**를 지정 가능합니다.

2. **명령어 실행**  

   ```bash
   python main_custom_diarization.py --audio_path=audio.wav --config_path=conf.yaml
   ```
   - `audio.wav`에 대해 화자 분할 및 ASR 수행 후,  
   - 결과 `asr_results.csv` 파일에 화자별 구간 및 인식된 텍스트가 저장됩니다.

3. **RTTM 파일 및 메트릭 계산(옵션)**  
   - `--config_path` 설정에 **reference_rttm** 경로가 지정되어 있다면, DER, JER, Purity, NMI 등 **평가 지표**가 로그로 출력됩니다.

---

## 🌟 기여 방법

기능 추가 또는 버그 수정은 언제든 환영입니다!

1. 이 레포를 Fork 후 기능별로 새로운 브랜치를 만들어주세요.
2. PR 전 로컬 환경에서 충분히 테스트해 주세요.
3. PR을 보내시면 코드 리뷰 후 반영하겠습니다.
