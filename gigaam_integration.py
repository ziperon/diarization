# gigaam_integration.py
import torch
import os
import torchaudio
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan

from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import os
import gigaam
import settings
import time
import threading
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class GigaAMRecognizer:
    """Wrapper for GigaAM speech recognition models with RNNT-v3 support."""
    
    def __init__(self, model_type: str = "e2e_rnnt", device: str = None):
        """
        Initialize GigaAM model with RNNT-v3 support.
        
        Args:
            model_type: Type of model to use ("e2e_rnnt" for RNNT-v3)
            device: Device to run the model on (cuda or cpu)
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sampling_rate = 16000  # GigaAM uses 16kHz audio
        
       
        
    def load_model(self):
        """Load the GigaAM model with timeout and fallback to 'ctc' if needed."""
        try:
            selected = self.model_type
            timeout_s = getattr(settings, "GIGAAM_LOAD_TIMEOUT", 30)
            logging.info(f"🔄 Loading GigaAM {selected} on {self.device} (timeout={timeout_s}s)")
            t0 = time.time()

            result = {"model": None, "err": None}

            def _load(kind: str):
                try:
                    logging.info(f"📥 Starting gigaam.load_model({kind})")
                    #os.environ["HTTP_PROXY"] = "http://dmsk2054:8080"
                    #os.environ["HTTPS_PROXY"] = "http://dmsk2054:8080"

                    result["model"] = gigaam.load_model(kind)
                    os.environ["HTTP_PROXY"] = ""
                    os.environ["HTTPS_PROXY"] = ""

                except Exception as e:
                    result["err"] = e

            used = selected
            th = threading.Thread(target=_load, args=(selected,), daemon=True)
            th.start()
            th.join(timeout_s)

            if result["model"] is None:
                logging.warning(f"⚠️ gigaam.load_model({selected}) did not complete within {timeout_s}s or errored: {result['err']}")
                if selected != "ctc":
                    used = "ctc"
                    result = {"model": None, "err": None}
                    logging.info("↪️ Falling back to 'ctc'")
                    th = threading.Thread(target=_load, args=(used,), daemon=True)
                    th.start()
                    th.join(timeout_s)

            if result["model"] is None:
                raise RuntimeError(f"Failed to load GigaAM model '{selected}' and fallback 'ctc'")

            self.model = result["model"]
            logging.info(f"📥 gigaam.load_model({used}) finished in {time.time()-t0:.1f}s")

            # Move to device if CUDA is available
            if "cuda" in self.device and torch.cuda.is_available():
                t1 = time.time()
                self.model = self.model.cuda()
                logging.info(f"✅ Model moved to GPU in {time.time()-t1:.1f}s")
            else:
                logging.info("ℹ️ Using CPU for inference")

            logging.info(f"✅ GigaAM {used} loaded successfully (total {time.time()-t0:.1f}s)")

        except Exception as e:
            logging.error(f"❌ Error loading GigaAM model: {str(e)}")
            raise
    
    def transcribe(
        self, 
        audio_path: str,
        language: str = "ru",
        word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe audio file using GigaAM RNNT-v3.
        
        Args:
            audio_path: Path to audio file
            language: Language code (currently only 'ru' is supported)
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary containing transcription results
        """
        if not self.model:
            self.load_model()
            
        try:
            logging.info(f"🧪 Preparing to transcribe: {audio_path}")
            # Check if we need to handle long-form audio
            t0 = time.time()
            duration = self._get_audio_duration(audio_path)
            is_long_audio = duration > 25.0  # GigaAM's short-form limit is 25 seconds
            
            logging.info(f"🎧 Transcribing {'long' if is_long_audio else 'short'} audio: {os.path.basename(audio_path)} (dur={duration:.2f}s)")
            
            if is_long_audio:
                res = self._transcribe_long(audio_path, language)
                logging.info(f"✅ Long-form transcription finished in {time.time()-t0:.1f}s")
                return res
            else:
                res = self._transcribe_short(audio_path, language, duration)
                logging.info(f"✅ Short-form transcription finished in {time.time()-t0:.1f}s")
                return res
            
        except Exception as e:
            logging.error(f"❌ Error during transcription: {str(e)}")
            raise
    
    def _transcribe_short(self, audio_path: str, language: str, duration: float) -> Dict:
        """Transcribe short audio (up to 25 seconds)."""
        try:
            logging.info("▶️ GigaAM _transcribe_short: calling model.transcribe(...) ")
            t0 = time.time()
            text = self.model.transcribe(audio_path)
            logging.info(f"⏱ model.transcribe finished in {time.time()-t0:.1f}s")
            return {
                "text": text,
                "language": language,
                "segments": [{
                    "start": 0,
                    "end": duration,
                    "text": text
                }]
            }
        except Exception as e:
            logging.error(f"❌ Error in short-form transcription: {str(e)}")
            raise
    
    def _transcribe_long(self, audio_path: str, language: str) -> Dict:
        """Transcribe long audio (more than 25 seconds)."""
        try:
            # Check if HF_TOKEN is set for pyannote
            if not os.environ.get("HF_TOKEN"):
                logging.warning("HF_TOKEN not set. Long-form transcription might fail. Set HF_TOKEN with read access to 'pyannote/segmentation-3.0'")
            
            # Perform long-form transcription
            utterances = self.model.transcribe_longform(audio_path,min_duration=0.01, max_duration=3, new_chunk_threshold=0.2,strict_limit_duration=2)
            logging.info(utterances)
            # Format the result
            segments = []
            for utt in utterances:
                text, (start, end) = utt["transcription"], utt["boundaries"]
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text.strip()
                })
            
            return {
                "text": " ".join([s["text"] for s in segments]),
                "language": language,
                "segments": segments
            }
            
        except Exception as e:
            logging.error(f"❌ Error in long-form transcription: {str(e)}")
            raise
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get the duration of an audio file in seconds."""
        try:
            logging.info(f"ℹ️ Reading audio info via torchaudio: {audio_path}")
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            logging.warning(f"Could not determine audio duration: {str(e)}")
            return 0.0


def transcribe_with_gigaam(
    audio_path: str,
    model_type: str = "e2e_rnnt",
    device: str = None,
    **kwargs
) -> Dict:
    """
    Convenience function to transcribe audio using GigaAM RNNT-v3.
    
    Args:
        audio_path: Path to audio file
        model_type: Type of model to use (default: "e2e_rnnt" for RNNT-v3)
        device: Device to run the model on (cuda or cpu)
        **kwargs: Additional arguments to pass to the recognizer
        
    Returns:
        Dictionary containing transcription results
    """
    recognizer = GigaAMRecognizer(model_type=model_type, device=device)
    return recognizer.transcribe(audio_path, **kwargs)


class GigaAM3Diarizer:
    """Diarize with pyannote and transcribe each segment with provided GigaAM model."""
    def __init__(
        self,
        gigamodel,
        hf_token: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.gigamodel = gigamodel
        self.device = device
        _dev = torch.device(device) if isinstance(device, str) else device
        try:
            self.diarizer = Pipeline.from_pretrained(
                getattr(settings, "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
                cache_dir=getattr(settings, "MODELS_DIR", "./models"),
                local_files_only=True if not hf_token else False,
                use_auth_token=hf_token if hf_token else None
            ).to(_dev)
        except TypeError:
            # Some versions use 'auth_token' or no token param when local
            self.diarizer = Pipeline.from_pretrained(
                getattr(settings, "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
                cache_dir=getattr(settings, "MODELS_DIR", "./models")
            ).to(_dev)

    def diarize(self, wav_file: str):
        diarization = self.diarizer(wav_file)
        segments: List[Dict] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": float(segment.start),
                "end": float(segment.end)
            })
        return segments

    def _save_segment_to_wav(self, wav_file: str, start: float, end: float) -> str:
        waveform, sr = torchaudio.load(wav_file)
        start_i = max(0, int(start * sr))
        end_i = min(waveform.shape[-1], int(end * sr))
        if end_i <= start_i:
            end_i = min(waveform.shape[-1], start_i + 1)
        seg = waveform[:, start_i:end_i]

        # Resample to 16k mono if needed
        if sr != 16000:
            seg = torchaudio.functional.resample(seg, orig_freq=sr, new_freq=16000)
            sr = 16000
        if seg.dim() == 2 and seg.size(0) > 1:
            seg = seg[:1, :]

        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        torchaudio.save(path, seg, sample_rate=sr)
        return path

    def transcribe_segment(self, wav_path: str) -> str:
        # Expecting GigaAM model to accept file path as in existing integration
        return self.gigamodel.transcribe(wav_path)

    def diarize_and_transcribe(self, wav_file: str) -> List[Dict]:
        segments = self.diarize(wav_file)
        try:
            info = torchaudio.info(wav_file)
            audio_dur = info.num_frames / info.sample_rate
        except Exception:
            audio_dur = None
        pad = float(getattr(settings, "ASR_SEGMENT_PAD_S", 0.1))
        min_len = float(getattr(settings, "MIN_ASR_SEGMENT_S", 0.5))
        results: List[Dict] = []
        for seg in segments:
            s = float(seg["start"]) if isinstance(seg["start"], (int, float)) else seg["start"]
            e = float(seg["end"]) if isinstance(seg["end"], (int, float)) else seg["end"]
            qs = s - pad
            qe = e + pad
            if audio_dur is not None:
                qs = max(0.0, qs)
                qe = min(audio_dur, qe)
            if (qe - qs) < min_len:
                mid = (qs + qe) / 2.0
                qs = max(0.0, mid - min_len / 2.0)
                qe = qs + min_len
                if audio_dur is not None and qe > audio_dur:
                    qe = audio_dur
                    qs = max(0.0, qe - min_len)
            path = self._save_segment_to_wav(wav_file, qs, qe)
            try:
                text = self.transcribe_segment(path)
            finally:
                try:
                    import os
                    os.remove(path)
                except Exception:
                    pass
            results.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": text if isinstance(text, str) else str(text)
            })
        return results 