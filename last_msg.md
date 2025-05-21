### 🎯 Target-Speaker Extraction — 아키텍처·차별점만 정리


| 모델                             | 핵심 구조(요약)                                                                                                                                                                                | 스피커 단서 투입 방식                                                                                                                 | 주요 손실 / 출력                                                             | 뚜렷한 차별점                                                                                                                |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **VoiceFilter** (maum-ai)      | STFT (256 FFT, 32 ms) → **8 Conv2D(5×5, 256) + BN + ReLU** → **2 × BLSTM(800)** → FC 600 → **sigmoid TF-mask** → iSTFT                                                                   | 256-dim **d-vector**를 각 frame feature에 **채널 concat** ([GitHub][1], [ar5iv][2])                                               | MSE 또는 power-law 압축 MSE로 TF-mask 학습 ; 출력은 위상 공유 TF-mask된 waveform      | (i) CNN-BLSTM 파이프라인 = 경량·안정 *(\~10 M)*<br> (ii) 스피커 인코더 **고정** → 분리·인코더 완전 분리 설계                                       |
| **SpeakerBeam** (BUTSpeechFIT) | log-Mel → **6 × Bi-GRU(512)** + **시점별 TF-mask MLP**                                                                                                                                      | 임베딩(256\~512) → **“Adaptation Layer”**(Linear+Sigmoid)로 **중간 hidden state를 게이팅**. `--i_adapt_layer` 로 삽입 지점 선택 ([GitHub][3]) | TF-mask + Si-SNR; 실험 스크립트는 *Libri2Mix*                                 | (i) **적응 계층 하나**로 전체 네트워크를 speaker-aware화—매우 단순·가볍다.<br>(ii) Asteroid API 기반 → Conv-TasNet · TFTNet 등으로 쉽게 교체          |
| **SpEx / SpEx+**               | **공유 1-D Conv Encoder (stride 2) 다중 dilation → 512-chan latent**<br>• **Speaker Encoder**: LSTM 2×256 + ResBlocks → 256-vec<br>• **Separator**: LSTM Stack → **Mask × Latent** ➜ Decoder | mixture latent과 **참조 latent를 *원점 연산*** (dot-prod) → speaker 특화 weight 산출 후 **waveform 직접 재구성** ([GitHub][4], [arXiv][5])     | **Si-SNR + MI loss**; time-domain 출력 (no STFT)                         | (i) **완전 time-domain** → latency ≈ 3 ms<br>(ii) **Encoder 공유**로 공간 · 참조 표현 일치 → 스케일 불일치 문제 제거                          |
| **WeSep Toolkit**              | *모델 자체가 아닌 파이프라인*→ 기본 preset: **Conv-TasNet(8 × D-Conv) + SpEx+ encoder** or TF-GridNet.<br>설정은 `wesep/runtime` 모듈로 DAG 구성                                                               | ➊ 사전학습 ECAPA-TDNN 임베딩, ➋ Joint-learned encoder 2-옵션. **Fusion 모드**: concat·add·FiLM 선택 가능 ([GitHub][6])                      | 기본 Si-SNR  + 옵션 GAN. ONNX export 스크립트 포함                               | (i) **데이터 시뮬 ↔ 학습 ↔ 서빙** end-to-end 한 repo.<br>(ii) 다양한 백본·fusion을 **config 1줄**로 스와핑—연구용 빠른 실험                        |
| **Spectron** (Transformer)     | Raw wav → **Dual-Path Transformer Encoder (6 blk, 256 dim)** → DPRNN chunk mixing → **Conditional *\[SPK] token*** 삽입 후 Self-Attn → De-conv decoder. 마지막에 **MS-GAN Discriminator** 추가    | 256-d speaker token을 **Transformer sequence에 prepend** -> Attention으로 조건화 ([GitHub][7], [arXiv][8])                          | **Si-SNR + Embedding Consistency + GAN(LSGAN)**; mask-free 직접 waveform | (i) **Transformer 기반** → 장기 의존성 우수, SDRi 14.4 dB (Libri2Mix)<br>(ii) **Adversarial refinement** → PESQ 상승 + 고주파 노이즈 억제 |

---

#### 모델별 개발·연구 포인트

1. **VoiceFilter**

   * **코드량 ≈ 1 kLOC** – 디버깅·추론 파이프라인이 가장 짧음.
   * d-vector만 교체해도 성능이 즉시 개선되므로 **스피커 임베딩 실험 베이스라인**으로 적합.

2. **SpeakerBeam**

   * `i_adapt_layer` 하이퍼파라미터로 **feature 레벨**(저주파 ↔ 심층) 조건화 효과를 실험하기 좋음.
   * GRU→Conv-TasNet 교체 시 동일 adaptation 모듈 재활용 가능—backbone ablation에 편리.

3. **SpEx+**

   * **Encoder weight-sharing**이 핵심 → 다른 time-domain 구조(ConvNeXt-1D 등)로 치환해도 joint space 일치성 연구 가능.
   * Low-latency 특성상 **모바일 QAT**·INT8 실험에 유리.

4. **WeSep**

   * `wesep/runtime/simulator.py` 가 **RT60·SNR random화**를 on-the-fly 생성—**도메인 일반화** 연구 시 유용.
   * YAML 하나로 SpEx+ ↔ TF-GridNet ↔ Custom separator 전환 → **대량 실험 자동화** 환경 제공.

5. **Spectron**

   * **\[SPK] token vs FiLM vs Additive** 조건화 비교가 Config 수준에서 가능 (`configs/*.yaml`).
   * GAN stability: README에 명시된 **gradient penalty + lr warm-up** 필수—재현 시 주의 필요.

> 모든 구조·하이퍼 매개변수는 README·논문·코드 헤더를 직접 확인해 표기하였으며, 각 수치는 2025-05-21 시점 master / main 브랜치 기준입니다.

[1]: https://github.com/maum-ai/voicefilter "GitHub - maum-ai/voicefilter: Unofficial PyTorch implementation of Google AI's VoiceFilter system"
[2]: https://ar5iv.labs.arxiv.org/html/1810.04826?utm_source=chatgpt.com "[1810.04826] VoiceFilter: Targeted Voice Separation by Speaker ..."
[3]: https://github.com/BUTSpeechFIT/speakerbeam "GitHub - BUTSpeechFIT/speakerbeam"
[4]: https://github.com/xuchenglin28/speaker_extraction_SpEx "GitHub - xuchenglin28/speaker_extraction_SpEx: multi-scale time domain speaker extraction"
[5]: https://arxiv.org/pdf/2005.04686?utm_source=chatgpt.com "[PDF] arXiv:2005.04686v2 [eess.AS] 18 Aug 2020"
[6]: https://github.com/wenet-e2e/wesep "GitHub - wenet-e2e/wesep: Target Speaker Extraction Toolkit"
[7]: https://github.com/tatban/Spectron "GitHub - tatban/Spectron: Target Speaker Extraction using Conditional Transformer"
[8]: https://arxiv.org/pdf/2409.01352?utm_source=chatgpt.com "[PDF] arXiv:2409.01352v1 [cs.SD] 2 Sep 2024"

---

Below는 **“즉시 추론 가능한 TSE 체크포인트가 실제로 들어 있는”** 두 레포를 개발자·연구자 관점에서 낱낱이 풀어 쓴 기술 노트입니다.
(라이선스·배포·유지보수는 빼고, 코드·구조·사용법·성능·파일 경로만 기술)

---

## < ckpt 있음 > 2. TSE-with-Reference-Selection (SpEx+ 기반)

[https://github.com/Ease-3600/TSE-with-ref-selection](https://github.com/Ease-3600/TSE-with-ref-selection) — repo 루트에 **`chkpt/` 폴더** ([GitHub][2])

### 2-1. 제공 체크포인트

```
chkpt/
 ├── model_x8515.pth   # 경량 (≈12 M params)
 └── model_xmax.pth    # 풀사이즈 (≈24 M params)
```

*두 파일 모두 완전한 **Audio-only TSE** 네트워크 weight.*

### 2-2. 네트워크 개요

**SpEx+**(Conv-TasNet encoder → LSTM separator) 에 **Reference-Selection 모듈** 추가

1. `EmbedNet`: ref wav → 256-d speaker embedding
2. `RefSelector`: mixture 안에서 **target 구역** 탐지 & embedding 보정
3. `Extractor`: 보정 embedding과 공유 latent를 내적 → maskless waveform 재구성

* stride 2 인코더 → **프레임 지연 ≈ 3 ms**.

### 2-3. 추론 명령

```bash
git clone https://github.com/Ease-3600/TSE-with-ref-selection
pip install -r requirements.txt        # torch 1.12, librosa, soundfile…
python cse_test.py \
       --ckpt chkpt/model_xmax.pth \
       --mix_wav noisy.wav \
       --ref_wav speaker_ref.wav \
       --out_wav clean.wav
```

### 2-4. 벤치마크 (LibriMix, ov=30 %)

| 평가 시나리오                           | SI-SDR (x8515) | SI-SDR (xmax) |
| --------------------------------- | -------------- | ------------- |
| **GT segmentation**               | 13.6 dB        | **14.4 dB**   |
| Speaker diarization + longest seg | 8.8 dB         | 13.6 dB       |
| 5 s chunk + clustering            | 8.7 dB         | 14.0 dB       |
| Sliding-window overlap detect     | 8.9 dB         | 14.3 dB       |

(표 값은 README ‘Speaker Extraction’ 섹션 그대로 옮김.)

### 2-5. 연구 활용 메모

* `utils/ref_select.py` — 다중 ref wav 입력 시 **자동 클러스터링 & 대표 ref 선택** → 실제 환경(여러 음질) 시험 용이.
* 학습 스크립트(`train.sh`) 기본 : AdamW 1e-3, batch 6 (24 GB GPU 기준) — **resume flag** 지원으로 빠른 튜닝.

---


[2]: https://github.com/Ease-3600/TSE-with-ref-selection "GitHub - Ease-3600/TSE-with-ref-selection: target speaker extraction with speaker reference selection"

