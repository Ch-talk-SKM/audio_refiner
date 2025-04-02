#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_custom_diarization.py

스피커 다이어라이제이션 + ASR 통합 파이프라인 :
 - Chunking + Overlap(옵션)
 - Pyannote VAD/Segmentation
 - Resegmentation(옵션)
 - 다중 임베딩(Pyannote, SpeechBrain, etc.) + Feature/Score-level Fusion + PCA
 - 다양한 클러스터링(AHC, HDBSCAN, Spectral) & 앙상블(Score-level, DOVER-lap)
 - Diarization 평가(DER, JER, Purity, NMI)
 - Whisper ASR(옵션)

 의존 라이브러리:
   pip install pyannote.audio speechbrain openai-whisper dover-lap hdbscan scikit-learn librosa soundfile
 실행 예:
   python main_custom_diarization.py --audio_path=audio.wav --config_path=conf.yaml
   CSV 파일 생성 (예: asr_results.csv)
"""

import os
import sys
import math
import logging
import argparse
import yaml
import warnings
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import csv

from pyannote.audio import Pipeline, Model, Audio
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

try:
    import hdbscan
    HDBSCAN_AVAILABLE=True
except ImportError:
    HDBSCAN_AVAILABLE=False
    hdbscan=None

try:
    from dover_lap import combine_rttms
    DOVER_AVAILABLE=True
except ImportError:
    DOVER_AVAILABLE=False

try:
    import whisper
    WHISPER_AVAILABLE=True
except ImportError:
    WHISPER_AVAILABLE=False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def l2_normalize(data: np.ndarray, eps=1e-9):
    norm = np.linalg.norm(data, axis=1, keepdims=True) + eps
    return data/norm

def handle_hdbscan_outliers(labels: np.ndarray):
    out=labels.copy()
    if (out==-1).any():
        noise_idx=np.where(out==-1)[0]
        if (out!=-1).any():
            max_label = out[out!=-1].max()
        else:
            max_label=-1
        next_label = max_label+1
        for idx in noise_idx:
            out[idx]= next_label
            next_label +=1
    return out

def read_rttm_to_annotation(path: str) -> Annotation:
    ann=Annotation()
    if not os.path.isfile(path):
        return ann
    with open(path,"r") as f:
        for line in f:
            if line.strip()=="" or line.startswith("#"):
                continue
            parts=line.strip().split()
            if len(parts)<8:
                continue
            if parts[0]!="SPEAKER":
                continue
            start= float(parts[3])
            dur  = float(parts[4])
            spk  = parts[7]
            ann[Segment(start, start+dur)] = spk
    return ann

def annotation_to_rttm(ann: Annotation, out_path:str):
    with open(out_path,"w") as f:
        for seg,track_id,spk in ann.itertracks(yield_label=True):
            start=seg.start
            dur= seg.end - seg.start
            line=f"SPEAKER {os.path.basename(out_path)} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            f.write(line)
    logging.info(f"Saved RTTM => {out_path}")

def compute_der_jer(ref: Annotation, hyp: Annotation, collar=0.25, skip_overlap=False):
    der_metric= DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer_metric= JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    der_val= der_metric(ref, hyp)
    jer_val= jer_metric(ref, hyp)
    return der_val, jer_val

def compute_purity_nmi(ref: Annotation, hyp: Annotation):
    step=0.01
    t_min= min(ref.get_timeline().extent().start, hyp.get_timeline().extent().start)
    t_max= max(ref.get_timeline().extent().end, hyp.get_timeline().extent().end)
    times= np.arange(t_min, t_max, step)
    ref_labels=[]
    hyp_labels=[]
    for t in times:
        seg= Segment(t, t+step/2)
        r_spk = list(ref.label(seg))
        h_spk = list(hyp.label(seg))
        r_str = "_".join(sorted(r_spk)) if r_spk else "None"
        h_str = "_".join(sorted(h_spk)) if h_spk else "None"
        ref_labels.append(r_str)
        hyp_labels.append(h_str)
    # Purity
    from collections import defaultdict, Counter
    data = list(zip(hyp_labels, ref_labels))
    cluster_map= defaultdict(list)
    for syslb, reflb in data:
        cluster_map[syslb].append(reflb)
    total_pts=len(data)
    correct=0
    for syslb, ref_list in cluster_map.items():
        c = Counter(ref_list)
        correct += c.most_common(1)[0][1]
    purity= correct/total_pts if total_pts>0 else 0.0
    # NMI
    nmi_val= normalized_mutual_info_score(ref_labels, hyp_labels)
    return purity, nmi_val

def do_doverlap(rttm_files):
    if not DOVER_AVAILABLE:
        logging.warning("DOVER-lap not installed.")
        return None
    combined_text= combine_rttms(rttm_files, method="lap")
    return combined_text

###############################################################################
# Pipeline Class
###############################################################################

class IntegratedDiarASRPipeline:
    """
    - Chunk(겹침) + VAD => segments
    - Resegmentation(옵션)
    - 임베딩( Pyannote/SpeechBrain 등 ) => Score/Feature Fusion + PCA
    - 여러 클러스터링 => 앙상블
    - Pyannote.metrics로 Diar Eval
    - Whisper ASR
    """
    def __init__(self, config: dict):
        self.config=config
        set_seed(config.get("seed", 0))

        # VAD pipeline
        self.vad_pipeline=None
        if self.config.get("vad",{}).get("enabled",True):
            vad_model_id= self.config["vad"].get("model_id","pyannote/voice-activity-detection")
            logging.info(f"Loading VAD pipeline: {vad_model_id}")
            self.vad_pipeline= Pipeline.from_pretrained(vad_model_id, use_auth_token=config.get("hf_token",None))
            if hasattr(self.vad_pipeline,"to"):
                dev= torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    self.vad_pipeline.to(dev)
                except:
                    pass

        # Resegmentation? (Pyannote)
        self.do_reseg= self.config.get("resegmentation",{}).get("enabled",False)
        self.reseg_model_id= self.config.get("resegmentation",{}).get("model_id","pyannote/segmentation")

        # Embedding models
        self.embedding_models=[]
        for emb in config["embedding_models"]:
            if emb.lower()=="pyannote":
                emb_id="pyannote/embedding"
            elif emb.lower()=="speechbrain":
                emb_id="speechbrain/spkrec-ecapa-voxceleb"
            else:
                emb_id= emb
            logging.info(f"Loading embedding: {emb_id}")
            emb_obj = PretrainedSpeakerEmbedding(emb_id, device=("cuda" if torch.cuda.is_available() else "cpu"))
            self.embedding_models.append(emb_obj)

        # Whisper ASR
        self.whisper_asr=None
        if config.get("whisper",{}).get("enabled",True) and WHISPER_AVAILABLE:
            wmodel= config["whisper"].get("model_name","medium")
            logging.info(f"Loading Whisper model: {wmodel}")
            self.whisper_asr= whisper.load_model(wmodel, device=("cuda" if torch.cuda.is_available() else "cpu"))

    def run_pipeline(self):
        audio_path= self.config["audio_path"]
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # 1) chunk the audio (if configured) => each chunk do VAD => accumulate segments
        total_segments= self._chunk_and_vad(audio_path)
        if not total_segments:
            logging.warning("No speech segments from chunk+VAD.")
            return None

        # 2) optional reseg
        if self.do_reseg and self.reseg_model_id:
            logging.info("Applying resegmentation chunk by chunk.")
            total_segments= self._apply_reseg(audio_path, total_segments)

        # 3) Embedding extraction: score-level or feature-level
        seg_info_list= self._extract_embeddings(audio_path, total_segments)

        # 4) clustering & ensemble
        final_labels= self._do_clustering(seg_info_list)

        # 5) build final annotation
        final_annotation= self._build_annotation(seg_info_list, final_labels)

        # 6) evaluation
        if self.config.get("compute_metrics",False) and self.config.get("reference_rttm",None):
            self._evaluate_annotation(final_annotation)

        # 7) ASR
        asr_list=[]
        if self.whisper_asr:
            asr_list= self._run_whisper(audio_path, final_annotation)

        return {
            "annotation": final_annotation,
            "asr": asr_list
        }

    def _chunk_and_vad(self, audio_path:str):
        """
        Chunk the audio if config has chunk_duration>0,
        then run VAD in each chunk => unify segments in global timeline
        """
        chunk_dur= self.config.get("chunk",{}).get("duration", 0)
        chunk_overlap= self.config.get("chunk",{}).get("overlap", 0)
        import librosa
        total_dur= librosa.get_duration(filename=audio_path)
        # build chunk list
        chunks=[]
        if chunk_dur>0 and total_dur>chunk_dur:
            step= chunk_dur - chunk_overlap
            num_ch= math.ceil((total_dur - chunk_overlap)/step)
            for i in range(num_ch):
                st= i*step
                ed= min(st+chunk_dur, total_dur)
                if ed<=st:
                    break
                chunks.append((st,ed))
        else:
            # single chunk
            chunks=[(0.0,total_dur)]
        # run VAD in each chunk
        all_segments=[]
        for st_t, ed_t in chunks:
            if not self.vad_pipeline:
                # no vad => entire chunk is speech
                all_segments.append((st_t, ed_t))
            else:
                # pyannote pipeline on chunk
                temp_wav="temp_chunk.wav"
                sr=16000
                # extract chunk audio => save => pipeline
                y, sr2= librosa.load(audio_path, sr=None, mono=True, offset=st_t, duration=(ed_t-st_t))
                if sr2!=16000:
                    y= librosa.resample(y, orig_sr=sr2, target_sr=16000)
                sf.write(temp_wav, y, sr)
                vad_res= self.vad_pipeline(temp_wav)
                # parse result
                timeline= vad_res.get_timeline().support()
                for seg in timeline:
                    seg_start= seg.start + st_t
                    seg_end  = seg.end + st_t
                    if seg_end>seg_start:
                        all_segments.append((seg_start, seg_end))
                os.remove(temp_wav)

        # Merge short segments or filter
        min_seg= self.config.get("vad",{}).get("min_duration_on", 0.0)
        all_segments= sorted(all_segments, key=lambda x:x[0])
        merged=[]
        prev_st=None; prev_ed=None
        for (st,ed) in all_segments:
            if prev_st is None:
                prev_st, prev_ed= st, ed
            else:
                if st - prev_ed < 0.01:  # contiguous
                    prev_ed= max(prev_ed, ed)
                else:
                    if (prev_ed - prev_st)>= min_seg:
                        merged.append((prev_st, prev_ed))
                    prev_st, prev_ed= st, ed
        if prev_st is not None and (prev_ed - prev_st)>= min_seg:
            merged.append((prev_st, prev_ed))
        return merged

    def _apply_reseg(self, audio_path:str, segments):
        """
        Apply pyannote Resegmentation on each chunk's segments
        For simplicity, we do it segment by segment => annotation => reseg => collect
        """
        from pyannote.audio.pipelines import Resegmentation
        reseg_pipe= Resegmentation(segmentation=self.reseg_model_id)
        # Optionally set param
        reseg_conf= self.config.get("resegmentation",{})
        inst_params={}
        if "min_duration_off" in reseg_conf:
            inst_params["min_duration_off"]= reseg_conf["min_duration_off"]
        try:
            reseg_pipe.instantiate(inst_params)
        except:
            logging.warning("Cannot instantiate reseg param. skipping.")
        new_all=[]
        audio_reader= Audio()
        # chunk approach not repeated, we just do each segment individually => might be slow for large # segments
        # or do bigger merges
        for (st,ed) in segments:
            ann= Annotation()
            ann[Segment(st,ed)] = "SPEAKER_00"
            wave, sr= audio_reader.crop(audio_path, Segment(st,ed))
            if wave.ndim==2 and wave.shape[0]>1:
                wave= wave.mean(axis=0, keepdims=True)

            #wave_th= torch.from_numpy(wave).float()
            wave_th= wave.float()
            
            # reseg
            try:
                refined= reseg_pipe({"waveform": wave_th, "sample_rate":sr}, diarization=ann)
                for seg, track_id, spk in refined.itertracks(yield_label=True):
                    new_all.append((seg.start+st, seg.end+st))
            except Exception as e:
                logging.warning(f"Reseg error for segment {st}-{ed}: {e}")
                new_all.append((st,ed))
        new_all= sorted(new_all, key=lambda x:x[0])
        # merge again
        merged=[]
        prev_s=None; prev_e=None
        min_seg= self.config["vad"].get("min_duration_on", 0.3)
        for (s,e) in new_all:
            if prev_s is None:
                prev_s,prev_e= s,e
            else:
                if s - prev_e<0.01:
                    prev_e= max(prev_e,e)
                else:
                    if (prev_e-prev_s)>= min_seg:
                        merged.append((prev_s, prev_e))
                    prev_s, prev_e= s,e
        if prev_s is not None and (prev_e-prev_s)>=min_seg:
            merged.append((prev_s, prev_e))
        return merged

    def _extract_embeddings(self, audio_path:str, segments):
        """
        Return a list of dict:
          [{"start":..., "end":..., "embedding":..., "embedding_sep":[...], ...}, ...]
        Depending on 'embedding_fusion' (score vs feature).
        """
        fuse_mode= self.config.get("embedding_fusion","feature")
        audio_reader= Audio()
        seg_list=[]
        separate_storage= [[] for _ in range(len(self.embedding_models))] if (fuse_mode=="score" and len(self.embedding_models)>1) else None
        for i,(st,ed) in enumerate(segments):
            wave, sr= audio_reader.crop(audio_path, Segment(st,ed))
            if wave.ndim==2 and wave.shape[0]>1:
                wave= wave.mean(axis=0, keepdims=True)

            #wave_th= torch.from_numpy(wave).float()
            wave_th= wave.float()
            wave_th = wave_th.unsqueeze(0)

            # embedding
            emb_concat=[]
            for m_idx, emb_obj in enumerate(self.embedding_models):
                with torch.no_grad():
                    e= emb_obj(wave_th.to(emb_obj.device))
                
                #e_np= e.detach().cpu().numpy().squeeze()
                e_np= e.squeeze()

                e_np= np.nan_to_num(e_np)
                if self.config.get("embedding_norm",True):
                    l2= np.linalg.norm(e_np)
                    if l2>1e-9:
                        e_np= e_np/l2
                if separate_storage is not None:
                    separate_storage[m_idx].append(e_np)
                else:
                    emb_concat.append(e_np)
            if separate_storage is None:
                if len(emb_concat)==1:
                    fused= emb_concat[0]
                else:
                    fused= np.concatenate(emb_concat, axis=0)
            else:
                fused=None
            seg_info={
                "start": st,
                "end": ed,
                "embedding": fused,
                "index": i
            }
            seg_list.append(seg_info)
        # PCA etc. => if feature fusion
        if fuse_mode=="score" and separate_storage is not None:
            # store in seg_info["embedding_sep"]
            # Possibly apply PCA model by model
            pca_dim= self.config.get("embedding_pca_dim", None)
            for m_idx in range(len(separate_storage)):
                arr= np.vstack(separate_storage[m_idx])
                if pca_dim and arr.shape[1]>pca_dim:
                    p= PCA(n_components=pca_dim, random_state=0)
                    arr= p.fit_transform(arr)
                separate_storage[m_idx]=arr
            for seg_i, info in enumerate(seg_list):
                emb_list=[]
                for m_idx in range(len(separate_storage)):
                    emb_list.append(separate_storage[m_idx][seg_i])
                info["embedding_sep"]= emb_list
        else:
            # feature-level => we can apply PCA globally
            pca_dim= self.config.get("embedding_pca_dim", None)
            if pca_dim:
                arr= np.array([s["embedding"] for s in seg_list])
                if arr.shape[1]> pca_dim:
                    logging.info(f"Applying global PCA => {pca_dim}")
                    p= PCA(n_components=pca_dim, random_state=0)
                    arr2= p.fit_transform(arr)
                    # L2 norm again if needed
                    if self.config.get("final_l2_norm",False):
                        arr2= l2_normalize(arr2)
                    for j in range(len(seg_list)):
                        seg_list[j]["embedding"]= arr2[j]
                else:
                    # final L2 norm
                    if self.config.get("final_l2_norm",False):
                        for j in range(len(seg_list)):
                            v= seg_list[j]["embedding"]
                            v= np.nan_to_num(v)
                            nrm= np.linalg.norm(v)
                            if nrm>1e-9:
                                seg_list[j]["embedding"]= v/nrm
                    pass
            else:
                # final l2 norm if config
                if self.config.get("final_l2_norm",False):
                    for j in range(len(seg_list)):
                        v= seg_list[j]["embedding"]
                        v= np.nan_to_num(v)
                        nrm= np.linalg.norm(v)
                        if nrm>1e-9:
                            seg_list[j]["embedding"]= v/nrm
        return seg_list

    def _do_clustering(self, seg_list):
        if not seg_list:
            return np.array([], dtype=int)
        fuse_mode= self.config.get("embedding_fusion","feature")
        N= len(seg_list)
        methods= self.config.get("clustering_methods", ["ahc"])
        # build cluster_labels
        cluster_labels_dict={}
        if fuse_mode=="score" and len(self.embedding_models)>1:
            # separate embeddings => average sim => dist => cluster
            # gather them
            num_models= len(self.embedding_models)
            # build big arrays
            # shape: (num_models, N, emb_dim?)
            # we have seg["embedding_sep"][m_idx]
            all_sep= []
            for m_idx in range(num_models):
                arr= np.array([s["embedding_sep"][m_idx] for s in seg_list])
                all_sep.append(arr)
            # compute average sim
            sims= []
            for arr in all_sep:
                # l2 norm check
                arr= np.nan_to_num(arr)
                # cos sim
                csim= np.dot(arr, arr.T)
                csim= np.clip(csim, -1.0,1.0)
                sims.append(csim)
            combined_sim= sum(sims)/len(sims)
            combined_dist= 1.0 - combined_sim
            for m in methods:
                labs= self._run_clustering_on_distance(m, combined_sim, combined_dist)
                if labs is not None:
                    cluster_labels_dict[m]= labs
        else:
            # feature-level => direct
            X= np.array([s["embedding"] for s in seg_list])
            for m in methods:
                labs= self._run_clustering_on_feature(m, X)
                if labs is not None:
                    cluster_labels_dict[m]= labs
        if not cluster_labels_dict:
            return np.zeros(N,dtype=int)
        if len(cluster_labels_dict)==1 or not self.config.get("cluster_ensemble",None):
            return list(cluster_labels_dict.values())[0]
        # ensemble
        ensemble_method= self.config["cluster_ensemble"].lower()
        if ensemble_method=="score":
            return self._ensemble_score(cluster_labels_dict, N)
        elif ensemble_method=="voting":
            return self._ensemble_voting(cluster_labels_dict, N)
        elif ensemble_method=="dover":
            return self._ensemble_dover(cluster_labels_dict, seg_list)
        else:
            logging.warning(f"Unknown ensemble {ensemble_method}. fallback score ensemble.")
            return self._ensemble_score(cluster_labels_dict, N)

    def _run_clustering_on_distance(self, method, sim_mat, dist_mat):
        method= method.lower()
        N= dist_mat.shape[0]
        if N<=1:
            return np.zeros(N,int)
        num_speakers= self.config.get("num_speakers", None)
        if method=="ahc":
            if num_speakers:
                try:
                    clust= AgglomerativeClustering(n_clusters=num_speakers, affinity='precomputed', linkage='average')
                except TypeError:
                    clust= AgglomerativeClustering(n_clusters=num_speakers, metric='precomputed', linkage='average')
                labs= clust.fit_predict(dist_mat)
                return labs
            else:
                # hierarchical cut
                import scipy.cluster.hierarchy as sch
                tri= dist_mat[np.triu_indices(N,1)]
                Y= sch.linkage(tri, method='average')
                d= Y[:,2]
                if len(d)==0:
                    return np.zeros(N,int)
                diff= np.diff(np.sort(d))
                idx= np.argmax(diff) if len(diff)>0 else 0
                cut_dist= (np.sort(d)[idx]+np.sort(d)[idx+1])/2.0 if idx+1<len(d) else d[-1]
                labs= sch.fcluster(Y, t=cut_dist, criterion='distance')
                labs= labs-1
                return labs
        elif method=="hdbscan":
            if not HDBSCAN_AVAILABLE:
                logging.warning("HDBSCAN not installed.")
                return None
            c= hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2)
            labs= c.fit_predict(dist_mat)
            labs= handle_hdbscan_outliers(labs)
            return labs
        elif method=="spectral":
            if not num_speakers:
                logging.warning("Spectral needs num_speakers => skip.")
                return None
            sc= SpectralClustering(n_clusters=num_speakers, affinity='precomputed', random_state=0)
            labs= sc.fit_predict(sim_mat)
            return labs
        else:
            logging.warning(f"Unknown method {method}")
            return None

    def _run_clustering_on_feature(self, method, X: np.ndarray):
        method= method.lower()
        N= X.shape[0]
        if N<=1:
            return np.zeros(N,int)
        num_spk= self.config.get("num_speakers", None)
        if method=="ahc":
            if num_spk:
                c= AgglomerativeClustering(n_clusters=num_spk, affinity='euclidean', linkage='average')
                labs= c.fit_predict(X)
                return labs
            else:
                import scipy.cluster.hierarchy as sch
                from sklearn.metrics import pairwise_distances
                dist_mat= pairwise_distances(X, metric='euclidean')
                tri= dist_mat[np.triu_indices(N,1)]
                Y= sch.linkage(tri, method='average')
                d= Y[:,2]
                if len(d)==0:
                    return np.zeros(N,int)
                diff= np.diff(np.sort(d))
                idx= np.argmax(diff) if len(diff)>0 else 0
                cut_d= (np.sort(d)[idx]+np.sort(d)[idx+1])/2.0 if idx+1<len(d) else d[-1]
                labs= sch.fcluster(Y, t=cut_d, criterion='distance')
                labs= labs-1
                return labs
        elif method=="hdbscan":
            if not HDBSCAN_AVAILABLE:
                logging.warning("HDBSCAN not installed.")
                return None
            c= hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=2)
            labs= c.fit_predict(X)
            labs= handle_hdbscan_outliers(labs)
            return labs
        elif method=="spectral":
            if not num_spk:
                logging.warning("Spectral needs num_speakers => skip.")
                return None
            # build cos sim => [0,1]
            Xn= l2_normalize(X)
            sim= np.dot(Xn, Xn.T)
            aff= (sim+1)/2
            sc= SpectralClustering(n_clusters=num_spk, affinity='precomputed', random_state=0)
            labs= sc.fit_predict(aff)
            return labs
        else:
            logging.warning(f"Unknown method {method}")
            return None

    def _ensemble_score(self, label_dict, N):
        # adjacency average => AHC
        M= len(label_dict)
        adj_sum= np.zeros((N,N), float)
        for labs in label_dict.values():
            for i in range(N):
                for j in range(N):
                    if labs[i]==labs[j]:
                        adj_sum[i,j]+=1
        adj= adj_sum/M
        dist= 1.0- adj
        num_spk= self.config.get("num_speakers", None)
        import scipy.cluster.hierarchy as sch
        tri= dist[np.triu_indices(N,1)]
        Y= sch.linkage(tri, method='average')
        d= Y[:,2]
        if len(d)==0:
            return np.zeros(N,int)
        diff= np.diff(np.sort(d))
        idx= np.argmax(diff) if len(diff)>0 else 0
        cut_d= (np.sort(d)[idx]+np.sort(d)[idx+1])/2.0 if idx+1<len(d) else d[-1]
        labs= sch.fcluster(Y, t=cut_d, criterion='distance')
        return labs-1

    def _ensemble_voting(self, label_dict, N):
        # majority => same cluster
        M= len(label_dict)
        same_count= np.zeros((N,N), float)
        for labs in label_dict.values():
            for i in range(N):
                for j in range(N):
                    if labs[i]==labs[j]:
                        same_count[i,j]+=1
        same_bool= (same_count>=(M/2.0))
        final_labs= -np.ones(N,int)
        label_id=0
        for i in range(N):
            if final_labs[i]!=-1:
                continue
            stack=[i]
            final_labs[i]= label_id
            while stack:
                k= stack.pop()
                for j in range(N):
                    if final_labs[j]==-1 and same_bool[k,j]:
                        final_labs[j]= label_id
                        stack.append(j)
            label_id+=1
        return final_labs

    def _ensemble_dover(self, label_dict, seg_list):
        if not DOVER_AVAILABLE:
            logging.warning("DOVER-lap not installed => fallback score ensemble.")
            return self._ensemble_score(label_dict, len(seg_list))
        tmp_files=[]
        for algo_name,labs in label_dict.items():
            ann= self._build_annotation(seg_list, labs)
            path= f"{algo_name}_temp.rttm"
            annotation_to_rttm(ann, path)
            tmp_files.append(path)
        combined= do_doverlap(tmp_files)
        if not combined:
            logging.warning("DOVER-lap combine failed => fallback score ensemble.")
            return self._ensemble_score(label_dict, len(seg_list))
        # parse
        dover_ann= Annotation()
        lines= combined.strip().split("\n")
        for line in lines:
            if line.strip()=="" or line.startswith("#"):
                continue
            parts=line.split()
            if len(parts)<8:
                continue
            if parts[0]!="SPEAKER":
                continue
            st= float(parts[3])
            dur= float(parts[4])
            spk= parts[7]
            dover_ann[Segment(st, st+dur)]=spk
        # map each segment
        final_labs=[]
        for info in seg_list:
            st, ed= info["start"], info["end"]
            seg= Segment(st, ed)
            spkset= list(dover_ann.label(seg))
            if spkset:
                final_labs.append(spkset[0])
            else:
                final_labs.append(-1)
        return np.array(final_labs)

    def _build_annotation(self, seg_list, labels):
        ann= Annotation()
        for info, lb in zip(seg_list, labels):
            st, ed= info["start"], info["end"]
            if isinstance(lb,int):
                spk=f"SPEAKER_{lb:02d}"
            else:
                spk=str(lb)
            ann[Segment(st, ed)] = spk
        return ann

    def _evaluate_annotation(self, ann: Annotation):
        ref_path= self.config["reference_rttm"]
        ref_ann= read_rttm_to_annotation(ref_path)
        if ref_ann.get_timeline().duration()<=0:
            logging.warning("Empty reference RTTM or parse error.")
            return
        collar= self.config.get("evaluation",{}).get("collar",0.25)
        skip_ov= self.config.get("evaluation",{}).get("skip_overlap",False)
        der_val, jer_val= compute_der_jer(ref_ann, ann, collar=collar, skip_overlap=skip_ov)
        pu_val, nmi_val= compute_purity_nmi(ref_ann, ann)
        logging.info(f"[EVAL] DER={der_val*100:.2f}%, JER={jer_val*100:.2f}%, Purity={pu_val*100:.2f}%, NMI={nmi_val:.3f}")

    def _run_whisper(self, audio_path:str, ann: Annotation):
        results=[]
        from pyannote.audio import Audio
        audio_reader= Audio()
        wconf= self.config.get("whisper",{})
        language= wconf.get("language",None)
        for seg, track_id, spk in ann.itertracks(yield_label=True):
            st, ed= seg.start, seg.end
            wave, sr= audio_reader.crop(audio_path, Segment(st,ed))
            if wave.ndim==2 and wave.shape[0]>1:
                wave= wave.mean(axis=0, keepdims=True)

            #wave_np= wave.squeeze(0).astype(np.float32)
            wave_np = wave.squeeze(0).cpu().numpy().astype(np.float32)

            dur= ed-st
            if dur>30:
                tmpf="temp_seg.wav"
                sf.write(tmpf, wave_np, sr)
                res= self.whisper_asr.transcribe(tmpf, language=language)
                text= res.get("text","").strip()
                os.remove(tmpf)
            else:
                from whisper.audio import log_mel_spectrogram, pad_or_trim
                if sr!=16000:
                    wave_np= librosa.resample(wave_np, orig_sr=sr, target_sr=16000)
                    sr=16000
                wave_np= pad_or_trim(wave_np)

                #mel= log_mel_spectrogram(torch.from_numpy(wave_np), pad=False).to(self.whisper_asr.device)
                mel = log_mel_spectrogram(torch.from_numpy(wave_np)).to(self.whisper_asr.device)

                import whisper
                opts= whisper.DecodingOptions(language=language, fp16=False)
                dec= whisper.decode(self.whisper_asr, mel, opts)
                text= dec.text.strip()
            results.append({
                "start": st,
                "end": ed,
                "speaker": spk,
                "text": text
            })
        return results


###############################################################################
# MAIN
###############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default= "/home/skmoon/PycharmProjects/chat_tmp/CH_T_audio/Sample/01_data/YUDE220001.wav")
    parser.add_argument("--config_path", type=str, default=None)
    args= parser.parse_args()

    default_config= {
       "audio_path": args.audio_path,
       "hf_token": "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",
       "seed":0,
       "chunk": {
         "duration": 300.0,
         "overlap": 30.0
       },
       "vad": {
         "enabled": True,
         "model_id": "pyannote/voice-activity-detection",
         "min_duration_on": 0.3
       },
       "resegmentation":{
         "enabled": True,
         "model_id":"pyannote/segmentation",
         "min_duration_off":0.2
       },
       "embedding_models": ["pyannote","speechbrain"],
       "embedding_norm": True,
       "embedding_pca_dim": None,
       "final_l2_norm":False,
       "embedding_fusion": "feature",   # or "score"
       "clustering_methods": ["ahc","hdbscan","spectral"],
       "cluster_ensemble": "dover",     # "score", "voting","dover"
       "num_speakers": None,
       "compute_metrics": False,
       "reference_rttm": None,
       "evaluation":{
         "collar": 0.25,
         "skip_overlap": False
       },
       "whisper":{
         "enabled": True,
         "model_name":"small",
         "language": "ko"
       }
    }

    # 설정파일(conf.yaml 등) 추가 로딩/머지
    if args.config_path and os.path.isfile(args.config_path):
        with open(args.config_path,"r") as f:
            user_conf= yaml.safe_load(f)
        def recursive_update(d,u):
            for k,v in u.items():
                if isinstance(v,dict) and k in d and isinstance(d[k],dict):
                    recursive_update(d[k],v)
                else:
                    d[k]=v
        recursive_update(default_config, user_conf)

    config= default_config

    # 파이프라인 생성 및 실행
    pipeline= IntegratedDiarASRPipeline(config)
    result= pipeline.run_pipeline()
    if not result or not result["annotation"]:
        logging.info("No final annotation produced.")
        sys.exit(0)

    ann= result["annotation"]
    logging.info(f"Final annotation => #speakers: {len(ann.labels())}")

    # ASR 결과 로그 및 CSV 저장
    if result["asr"]:
        logging.info("ASR Results:")
        # CSV 파일 생성 (예: asr_results.csv)
        with open("asr_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start", "end", "speaker", "text"])

            for seg_info in result["asr"]:
                st = seg_info["start"]
                ed = seg_info["end"]
                spk = seg_info["speaker"]
                txt = seg_info["text"]

                # 로그에 출력
                logging.info(f"[{st:.2f}-{ed:.2f}] {spk}: {txt}")
                # CSV에 한 줄 추가
                writer.writerow([f"{st:.2f}", f"{ed:.2f}", spk, txt])

if __name__=="__main__":
    main()
            
