#!/usr/bin/env python3
"""
Advanced Speech Recognition System using Whisper v3 Large
Comparable to Lingvanex quality with full control
"""

import argparse
import asyncio
import json
import warnings
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any
import logging

import torch
import torchaudio
import numpy as np
try:
    setattr(np, "NaN", np.nan)
    setattr(np, "NAN", np.nan)
except Exception:
    pass
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing similar to Lingvanex"""
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        enable_denoise: bool = True,
        enable_normalize: bool = True
    ):
        self.target_sample_rate = target_sample_rate
        self.enable_denoise = enable_denoise
        self.enable_normalize = enable_normalize
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        logger.info(f"Loading audio from {file_path}")
        
        # Load with pydub (supports many formats like Lingvanex)
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Normalize volume
        if self.enable_normalize:
            audio = normalize(audio)
            logger.debug("Audio normalized")
        
        # Resample to target rate
        if audio.frame_rate != self.target_sample_rate:
            audio = audio.set_frame_rate(self.target_sample_rate)
            logger.debug(f"Resampled to {self.target_sample_rate} Hz")
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2**15)  # Normalize to [-1, 1]
        
        # Denoise
        if self.enable_denoise:
            samples = nr.reduce_noise(
                y=samples,
                sr=self.target_sample_rate,
                stationary=True,
                prop_decrease=0.8
            )
            logger.debug("Noise reduced")
        
        return samples
    
    def preprocess_for_whisper(self, audio: np.ndarray) -> np.ndarray:
        """Final preprocessing for Whisper model"""
        # Ensure audio is in correct range
        audio = np.clip(audio, -1.0, 1.0)
        return audio


class WhisperRecognizer:
    """
    Advanced Whisper v3 Large recognizer with Lingvanex-like features
    """
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter: bool = True,
        vad_threshold: float = 0.5,
        enable_diarization: bool = False,
        hf_token: Optional[str] = None
    ):
        """
        Initialize Whisper recognizer
        
        Args:
            model_size: Model size (large-v3, large-v2, medium, etc.)
            device: cuda or cpu
            compute_type: float16, int8, int8_float16 (for GPU)
            beam_size: Beam search size (like Lingvanex)
            best_of: Number of candidates when sampling
            temperature: Temperature fallback sequence
            vad_filter: Enable Voice Activity Detection
            vad_threshold: VAD sensitivity threshold
            enable_diarization: Enable speaker diarization
            hf_token: HuggingFace token for diarization model
        """
        logger.info(f"Initializing Whisper {model_size} on {device}")
        
        # Initialize Whisper model
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type if device == "cuda" else "int8",
            download_root="./models"
        )
        
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.enable_diarization = enable_diarization
        
        # Initialize diarization pipeline if needed
        self.diarization_pipeline = None
        if enable_diarization and hf_token:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                if device == "cuda":
                    self.diarization_pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline loaded")
            except Exception as e:
                logger.warning(f"Failed to load diarization: {e}")
        
        logger.info("Whisper model loaded successfully")
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        word_timestamps: bool = False,
        output_format: Literal["text", "srt", "vtt", "json"] = "text"
    ) -> Dict[str, Any]:
        """
        Transcribe audio with Lingvanex-like quality
        
        Args:
            audio: Audio array (preprocessed)
            language: Source language code (None for auto-detect)
            task: transcribe or translate (to English)
            word_timestamps: Generate word-level timestamps
            output_format: Output format
        
        Returns:
            Dictionary with transcription results
        """
        logger.info("Starting transcription...")
        
        # Transcribe with all optimizations
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            vad_parameters={
                "threshold": self.vad_threshold,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100
            },
            word_timestamps=word_timestamps or (output_format in ["srt", "vtt"]),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            initial_prompt=None,
            suppress_blank=True,
            suppress_tokens=[-1]
        )
        
        # Convert segments to list
        segments_list = list(segments)
        
        # Get detected language
        detected_language = info.language
        language_probability = info.language_probability
        
        logger.info(f"Detected language: {detected_language} (confidence: {language_probability:.2%})")
        logger.info(f"Transcribed {len(segments_list)} segments")
        
        # Apply diarization if enabled
        speaker_segments = None
        if self.enable_diarization and self.diarization_pipeline:
            speaker_segments = self._apply_diarization(audio, segments_list)
        
        # Format output
        result = {
            "text": " ".join([segment.text.strip() for segment in segments_list]),
            "language": detected_language,
            "language_probability": language_probability,
            "segments": self._format_segments(segments_list, speaker_segments),
            "word_count": sum(len(segment.text.split()) for segment in segments_list)
        }
        
        # Format based on output type
        if output_format == "srt":
            result["formatted"] = self._to_srt(segments_list, speaker_segments)
        elif output_format == "vtt":
            result["formatted"] = self._to_vtt(segments_list, speaker_segments)
        elif output_format == "json":
            result["formatted"] = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            result["formatted"] = result["text"]
        
        return result
    
    def _apply_diarization(
        self,
        audio: np.ndarray,
        segments: List[Any]
    ) -> Optional[Dict[float, str]]:
        """Apply speaker diarization"""
        if not self.diarization_pipeline:
            return None
        
        try:
            logger.info("Applying speaker diarization...")
            
            # Convert audio for diarization
            waveform = torch.from_numpy(audio).unsqueeze(0)
            
            # Run diarization
            diarization = self.diarization_pipeline(
                {"waveform": waveform, "sample_rate": 16000}
            )
            
            # Map speakers to timestamps
            speaker_map = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_map[turn.start] = speaker
            
            logger.info(f"Found {len(set(speaker_map.values()))} speakers")
            return speaker_map
            
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
            return None
    
    def _format_segments(
        self,
        segments: List[Any],
        speaker_map: Optional[Dict[float, str]] = None
    ) -> List[Dict[str, Any]]:
        """Format segments with optional speaker info"""
        formatted = []
        
        for segment in segments:
            seg_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": segment.avg_logprob
            }
            
            # Add speaker if available
            if speaker_map:
                # Find closest speaker
                closest_speaker = None
                min_diff = float('inf')
                for time, speaker in speaker_map.items():
                    diff = abs(time - segment.start)
                    if diff < min_diff:
                        min_diff = diff
                        closest_speaker = speaker
                
                if closest_speaker and min_diff < 1.0:  # Within 1 second
                    seg_dict["speaker"] = closest_speaker
            
            formatted.append(seg_dict)
        
        return formatted
    
    def _to_srt(
        self,
        segments: List[Any],
        speaker_map: Optional[Dict[float, str]] = None
    ) -> str:
        """Convert to SRT format"""
        srt_output = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self._format_timestamp(segment.start, srt=True)
            end_time = self._format_timestamp(segment.end, srt=True)
            text = segment.text.strip()
            
            # Add speaker prefix if available
            if speaker_map:
                for time, speaker in speaker_map.items():
                    if abs(time - segment.start) < 1.0:
                        text = f"[{speaker}] {text}"
                        break
            
            srt_output.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
        
        return "\n".join(srt_output)
    
    def _to_vtt(
        self,
        segments: List[Any],
        speaker_map: Optional[Dict[float, str]] = None
    ) -> str:
        """Convert to WebVTT format"""
        vtt_output = ["WEBVTT\n"]
        
        for segment in segments:
            start_time = self._format_timestamp(segment.start)
            end_time = self._format_timestamp(segment.end)
            text = segment.text.strip()
            
            if speaker_map:
                for time, speaker in speaker_map.items():
                    if abs(time - segment.start) < 1.0:
                        text = f"<v {speaker}>{text}"
                        break
            
            vtt_output.append(f"{start_time} --> {end_time}\n{text}\n")
        
        return "\n".join(vtt_output)
    
    @staticmethod
    def _format_timestamp(seconds: float, srt: bool = False) -> str:
        """Format timestamp for subtitles"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        if srt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class SpeechRecognitionSystem:
    """Complete speech recognition system"""
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        enable_denoise: bool = True,
        enable_normalize: bool = True,
        enable_diarization: bool = False,
        hf_token: Optional[str] = None,
        **whisper_kwargs
    ):
        """Initialize complete system"""
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            enable_denoise=enable_denoise,
            enable_normalize=enable_normalize
        )
        
        # Initialize recognizer
        self.recognizer = WhisperRecognizer(
            model_size=model_size,
            device=device,
            enable_diarization=enable_diarization,
            hf_token=hf_token,
            **whisper_kwargs
        )
    
    def process_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        output_format: str = "text",
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """Process audio file end-to-end"""
        
        # Load and preprocess
        audio = self.preprocessor.load_audio(file_path)
        audio = self.preprocessor.preprocess_for_whisper(audio)
        
        # Transcribe
        result = self.recognizer.transcribe(
            audio=audio,
            language=language,
            task=task,
            output_format=output_format
        )
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Speech Recognition with Whisper v3 Large"
    )
    
    # Input/Output
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument("--output", "-o", type=str, help="Output file (optional)")
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "srt", "vtt", "json"],
        help="Output format"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        choices=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
        help="Whisper model size"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--language", "-l", type=str, help="Source language code")
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task type"
    )
    
    # Quality settings (Lingvanex-like)
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--best-of", type=int, default=5, help="Best of N")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD threshold")
    
    # Advanced features
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable speaker diarization"
    )
    parser.add_argument("--hf-token", type=str, help="HuggingFace token for diarization")
    
    args = parser.parse_args()
    
    # Validate file
    if not Path(args.audio_file).exists():
        logger.error(f"File not found: {args.audio_file}")
        return
    
    # Initialize system
    system = SpeechRecognitionSystem(
        model_size=args.model,
        device=args.device,
        enable_denoise=not args.no_denoise,
        enable_normalize=not args.no_normalize,
        enable_diarization=args.diarization,
        hf_token=args.hf_token,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_filter=not args.no_vad,
        vad_threshold=args.vad_threshold
    )
    
    # Process
    logger.info(f"Processing: {args.audio_file}")
    result = system.process_file(
        file_path=args.audio_file,
        language=args.language,
        output_format=args.format,
        task=args.task
    )
    
    # Output results
    print("\n" + "="*80)
    print(f"Language: {result['language']} ({result['language_probability']:.2%} confidence)")
    print(f"Word count: {result['word_count']}")
    print("="*80 + "\n")
    
    print(result["formatted"])
    
    # Save to file if specified
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["formatted"])
        logger.info(f"Saved to: {args.output}")
    
    # Print segment info for debugging
    if args.format == "json":
        logger.info(f"Processed {len(result['segments'])} segments")


if __name__ == "__main__":
    main()