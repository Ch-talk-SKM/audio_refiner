## 1. 다중 임베딩(Multi-model Embedding) + Fusion

### 1-1. 다중 임베딩 로딩 (`embedding_models` 파라미터)
```python
# __init__() 내부
self.embedding_models = []
for emb in config["embedding_models"]:
    if emb.lower() == "pyannote":
        emb_id = "pyannote/embedding"
    elif emb.lower() == "speechbrain":
        emb_id = "speechbrain/spkrec-ecapa-voxceleb"
    else:
        emb_id = emb
    emb_obj = PretrainedSpeakerEmbedding(emb_id, device=("cuda" if torch.cuda.is_available() else "cpu"))
    self.embedding_models.append(emb_obj)
```
- `config["embedding_models"]`에 리스트로 모델 이름(예: `"pyannote"`, `"speechbrain"`)을 지정하면, 위와 같이 **여러 임베딩 모델**을 **동시에** 로드합니다.  
- 사용자 임의의 huggingface 모델 경로를 문자열로 직접 넣으면, 그대로 `PretrainedSpeakerEmbedding`에 할당해 쓸 수도 있습니다.

### 1-2. 임베딩 추출 로직 (`_extract_embeddings` 메소드)
```python
def _extract_embeddings(self, audio_path:str, segments):
    fuse_mode = self.config.get("embedding_fusion","feature")
    ...
    separate_storage = [[] for _ in range(len(self.embedding_models))] if (fuse_mode=="score" and len(self.embedding_models)>1) else None

    for i,(st,ed) in enumerate(segments):
        ...
        for m_idx, emb_obj in enumerate(self.embedding_models):
            with torch.no_grad():
                e = emb_obj(wave_th.to(emb_obj.device))
            e_np = e.squeeze()  # numpy 변환 전
            # L2 정규화 적용(옵션)
            if self.config.get("embedding_norm", True):
                l2 = np.linalg.norm(e_np)
                if l2 > 1e-9:
                    e_np = e_np / l2

            # "score" 모드면 separate_storage에 따로 저장
            if separate_storage is not None:
                separate_storage[m_idx].append(e_np)
            else:
                emb_concat.append(e_np)
    ...
```
- `fuse_mode`가 **`"score"`** 일 경우:  
  - 모델별 임베딩을 **따로** (`separate_storage[m_idx]`) 저장해 둡니다.  
  - 이후 스코어(유사도 행렬) 단에서 평균 내거나, 여러 모델의 HDBSCAN 등을 앙상블할 수 있게 됨.
  
- `fuse_mode`가 **`"feature"`** 일 경우:  
  - 모델별 임베딩을 **연결(Concatenate)** 해서 하나의 벡터로 만든 뒤(`emb_concat`), 각 세그먼트에 담아둡니다.  
  - 이후 클러스터링 시에는 **단일 피처 공간**(예: [dim=192] 형태)에서 직접 군집을 수행하게 됩니다.

### 1-3. PCA 및 최종 L2 정규화 (`embedding_pca_dim`, `final_l2_norm` 파라미터)

```python
pca_dim = self.config.get("embedding_pca_dim", None)
if pca_dim:
    # 전체 벡터에 PCA 적용
    p = PCA(n_components=pca_dim, random_state=0)
    arr2 = p.fit_transform(arr)
    # 필요시 최종 L2 정규화
    if self.config.get("final_l2_norm", False):
        arr2 = l2_normalize(arr2)
    ...
```
- `embedding_pca_dim`을 지정하면, 연결된(혹은 단일) 임베딩 벡터를 **PCA로 차원 축소**하여 클러스터링 성능을 높이거나 오버피팅 등을 방지합니다.
- `final_l2_norm`이 `True`면, 최종 축소된 벡터에 대해 다시 **L2 정규화**를 수행해, **코사인 거리 기반** 클러스터링이 좀 더 안정적이 되도록 도와줍니다.

### 정리

- **`embedding_models`**: 어떤 임베딩 모델을 몇 개나 쓸지 결정. (복수 지정 가능)  
- **`embedding_norm`**: 각 임베딩 벡터 추출 후 L2 정규화 수행 여부.  
- **`embedding_fusion`**: `"score"`면 **스코어 레벨 앙상블**, `"feature"`면 **벡터 레벨 연결** 후 클러스터링.  
- **`embedding_pca_dim`**, **`final_l2_norm`**: PCA 차원 축소와 추가 L2 정규화 기능.

---

## 2. 다양한 클러스터링(AHC, HDBSCAN, Spectral) & 앙상블(DOVER-lap 등)

### 2-1. 클러스터링 방식 설정 (`clustering_methods`)
```python
"clustering_methods": ["ahc", "hdbscan", "spectral"],
```
- AHC (Agglomerative Clustering), HDBSCAN, Spectral Clustering 등 **복수** 지정할 수 있습니다.
- 코드 내부에서는 각각 `_run_clustering_on_distance`(스코어 방식) 혹은 `_run_clustering_on_feature`(피처 방식)을 호출하여 분기 처리합니다.

#### 2-1-1. AHC (Agglomerative Clustering)
```python
if method=="ahc":
    if num_speakers:
        # 스피커 수가 고정된 경우
        clust= AgglomerativeClustering(n_clusters=num_speakers, ...)
        labs= clust.fit_predict(dist_mat or X)
    else:
        # 스피커 수 미지정 → 계층적 트리에서 distance 기준으로 best cut 찾기
        Y= sch.linkage(tri, method='average')
        labs= sch.fcluster(Y, t=cut_dist, criterion='distance')
```
- **`num_speakers`** 옵션에 따라  
  - 스피커 수를 고정(AHC의 `n_clusters`)  
  - 혹은 linkage 결과를 보고 최적의 `cut_dist`를 찾는 방식(가장 큰 거리차 분기를 찾아 자동 결정)  
- 피처 기반(`X`) 또는 거리행렬(`dist_mat`) 기반 둘 다 지원.

#### 2-1-2. HDBSCAN
```python
if method=="hdbscan":
    c= hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2)
    labs= c.fit_predict(dist_mat)
    labs= handle_hdbscan_outliers(labs)
```
- **`hdbscan`** 라이브러리가 설치되어 있을 때만 사용 가능(미설치 시 자동 스킵).  
- 클러스터링 결과에서 `-1`(outlier) 라벨을 발견하면, 내부적으로 `handle_hdbscan_outliers()`를 통해 **각 outlier마다 새 라벨**을 부여.

#### 2-1-3. Spectral Clustering
```python
elif method=="spectral":
    sc= SpectralClustering(n_clusters=num_speakers, affinity='precomputed', random_state=0)
    labs= sc.fit_predict(sim_mat)
```
- **코사인 유사도 행렬**(혹은 score matrix)을 만들어 `affinity='precomputed'`로 스펙트럴 클러스터링 수행.  
- Spectral 클러스터링은 일반적으로 **스피커 수(`num_speakers`)**가 고정되어 있어야 유의미하게 동작합니다.

---

### 2-2. 다수 클러스터링 결과의 앙상블 (`cluster_ensemble`)

클러스터링 방식이 여러 개인 경우, 코드 내부에서 다음 과정을 거칩니다.
1. **각 방법으로 클러스터 라벨**을 먼저 얻음 (`cluster_labels_dict`에 저장).
2. **앙상블 방법**(`cluster_ensemble`)에 따라 최종 라벨 결정.

```python
"cluster_ensemble": "dover",  # or "score", "voting"
```

#### 2-2-1. `score` 앙상블
```python
def _ensemble_score(self, label_dict, N):
    # 각 알고리즘의 라벨이 동일할 때 1, 다르면 0
    # 이를 평균하여 adjacency matrix 만들고, 다시 AHC 방식으로 최종 레이블 산출
```
- 여러 클러스터링 결과를 **유사도**(동일 스피커인지 여부) 형태로 변환하여,  
- 그 유사도를 평균낸 뒤(Averaging), 다시 AHC로 합치는 방식입니다.

#### 2-2-2. `voting` 앙상블
```python
def _ensemble_voting(self, label_dict, N):
    # 각 클러스터링에서 동일한지(=라벨이 같은지) 다수결로 판단
    # 동일하다고 나온 세그먼트는 하나의 connected component로 묶어 최종 라벨 부여
```
- 단순히 **여러 알고리즘 중 과반수(≥ M/2)로 동일하면** 같은 클러스터로 묶습니다.  
- 세그먼트를 그래프로 보고, **DFS**(혹은 Union-Find)로 연결 컴포넌트를 찾는 식으로 최종 라벨 할당.

#### 2-2-3. `dover` 앙상블 (DOVER-lap)
```python
if ensemble_method=="dover":
    if not DOVER_AVAILABLE:
        # dover-lap 미설치 시 점수 앙상블로 fallback
    else:
        # 클러스터 라벨별로 임시 RTTM 파일 생성 → DOVER-lap 합성 → 최종 RTTM 파싱
```
- **`dover-lap`** 라이브러리를 사용해, **각 클러스터링 결과를 RTTM 형태**로 저장 후, `combine_rttms(rttm_files, method="lap")` 함수를 통해 **중첩 가능(soft)한** 앙상블 결과를 산출합니다.
- RTTM을 기반으로 하기 때문에, 결과 라벨은 `"SPEAKER_XX"` 식으로 배정된 후, 다시 세그먼트 단위로 매핑해 최종 라벨을 얻습니다.  
- **딥러닝 모델 결과(RTTM) 다수를 합쳐 쓰는 Pyannote 생태계**에서 자주 쓰이는 방법입니다.

---

## 3. 추가 지표(Purity, NMI) 평가

### 3-1. 기존 Pyannote 메트릭(DER, JER)

코드에서는 `compute_der_jer()` 함수를 통해 **DiarizationErrorRate**, **JaccardErrorRate**를 계산:
```python
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

def compute_der_jer(ref: Annotation, hyp: Annotation, collar=0.25, skip_overlap=False):
    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer_metric = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    der_val = der_metric(ref, hyp)
    jer_val = jer_metric(ref, hyp)
    return der_val, jer_val
```
- `collar`(보정 구간)과 `skip_overlap`(오버랩 부분의 평가 제외 여부)을 옵션으로 설정하여,  
- **DER**, **JER**를 Pyannote 표준 방식대로 산출합니다.

### 3-2. Purity, NMI 계산 (`compute_purity_nmi()`)

```python
def compute_purity_nmi(ref: Annotation, hyp: Annotation):
    # 1) 일정 time step마다 레퍼런스/하이퍼로부터 화자 레이블 추출
    # 2) 해당 time step별 ref_labels, hyp_labels를 비교
    # 3) Purity 계산: 각 (예측클러스터 → 실제 스피커) 매핑에서 최대빈도를 합산 / 전체 샘플 수
    # 4) NMI 계산: normalized_mutual_info_score(ref_labels, hyp_labels)
```
- **Purity**: “시스템이 예측한 한 클러스터가 실제로 동일 화자로 얼마나 잘 모였는가”를 비율로 보는 지표.  
- **NMI**: 군집 결과와 실제 라벨 분포의 **정보 이론적 유사도**. 0~1 범위.

### 3-3. 사용 방식

- `config["compute_metrics"]`가 `True`이고, `config["reference_rttm"]`에 참조 RTTM 경로가 있으면,  
- `_evaluate_annotation()` 메소드 내부에서 `compute_der_jer()` 및 `compute_purity_nmi()`를 함께 호출해 로그로 출력.  

```python
def _evaluate_annotation(self, ann: Annotation):
    ref_path = self.config["reference_rttm"]
    ref_ann = read_rttm_to_annotation(ref_path)

    der_val, jer_val = compute_der_jer(ref_ann, ann, collar=0.25, skip_overlap=False)
    pu_val, nmi_val  = compute_purity_nmi(ref_ann, ann)
    logging.info(f"[EVAL] DER={der_val*100:.2f}%, JER={jer_val*100:.2f}%, Purity={pu_val*100:.2f}%, NMI={nmi_val:.3f}")
```
- **DER, JER, Purity, NMI** 모두 한 번에 계산해 출력하는 구조.

---

## 정리 요약

1. **다중 임베딩 + Fusion**  
   - `embedding_models`로 다양한 모델(Pytorch Lightning 기반, Hugging Face 등) 동시 사용.  
   - `embedding_fusion` 모드에 따라 **스코어 레벨**(각 임베딩별 유사도 행렬 후 평균) vs. **피처 레벨**(벡터 연결)로 융합.  
   - 필요 시 **PCA**(`embedding_pca_dim`)와 **최종 L2 정규화**(`final_l2_norm`)를 통해 군집 최적화.

2. **여러 클러스터링 + 앙상블**  
   - `clustering_methods` 파라미터(예: ["ahc","hdbscan","spectral"])로 **복수 알고리즘** 결과를 모두 수행.  
   - `cluster_ensemble`(`"score"`, `"voting"`, `"dover"`) 설정으로 **최종 라벨**을 하나로 합침.  
   - `dover-lap` 라이브러리가 설치된 경우 **RTTM 기반** 합성 가능.

3. **추가 지표(Purity, NMI) 평가**  
   - `DER`, `JER`(pyannote) 외에 **Purity**, **NMI** 같은 군집 품질 지표까지 자동 계산.  
   - `reference_rttm` 파일과 비교하여 보다 **종합적인 성능 평가**가 가능.

이처럼, 해당 코드에서는 **Pyannote**의 음성 분할/VAD/임베딩 기능을 적극 활용하면서도,  
- 임베딩을 복수로 활용하고,  
- 여러 클러스터링 알고리즘 & 앙상블 기법을 조합해,  
- 추가 지표들을 통해 상세 평가가 이뤄지도록  
**코드 레벨**에서 세심한 옵션들이 구성되어 있습니다.
