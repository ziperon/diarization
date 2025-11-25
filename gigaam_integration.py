# gigaam_integration.py
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import os
import gigaam
from gigaam.utils import format_time  # Import format_time for timestamp formatting

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
        
        # RNNT-v3 specific configuration
        self.supported_models = {
            "e2e_rnnt": f"v3_e2e_rnnt",  # RNNT-v3 model
            "ctc": "v3_ctc",             # CTC model (alternative)
            "ssl": "v3_ssl"              # Self-supervised model
        }
        
        if model_type not in self.supported_models:
            logging.warning(f"Model type '{model_type}' not recognized. Defaulting to 'e2e_rnnt'")
            self.model_type = "e2e_rnnt"
        
    def load_model(self):
        """Load the GigaAM RNNT-v3 model."""
        try:
            model_name = self.supported_models[self.model_type]
            logging.info(f"🔄 Loading GigaAM {model_name} on {self.device}")
            
            # Load the model
            self.model = gigaam.load_model(model_name)
            
            # Move to device if CUDA is available
            if "cuda" in self.device and torch.cuda.is_available():
                self.model = self.model.cuda()
                logging.info("✅ Model moved to GPU")
            else:
                logging.info("ℹ️ Using CPU for inference")
                
            logging.info(f"✅ GigaAM {model_name} loaded successfully")
            
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
            # Check if we need to handle long-form audio
            duration = self._get_audio_duration(audio_path)
            is_long_audio = duration > 25.0  # GigaAM's short-form limit is 25 seconds
            
            logging.info(f"🎧 Transcribing {'long' if is_long_audio else 'short'} audio: {os.path.basename(audio_path)}")
            
            if is_long_audio:
                return self._transcribe_long(audio_path, language)
            else:
                return self._transcribe_short(audio_path, language, duration)
            
        except Exception as e:
            logging.error(f"❌ Error during transcription: {str(e)}")
            raise
    
    def _transcribe_short(self, audio_path: str, language: str, duration: float) -> Dict:
        """Transcribe short audio (up to 25 seconds)."""
        try:
            text = self.model.transcribe(audio_path)
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