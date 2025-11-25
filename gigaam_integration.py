# gigaam_integration.py
import torch
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
                    result["model"] = gigaam.load_model(kind)
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
            utterances = self.model.transcribe_longform(audio_path)
            
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