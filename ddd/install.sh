#!/bin/bash

# ============================================================================
# Installation script for Whisper v3 Large Speech Recognition System
# ============================================================================

echo "=========================================="
echo "Whisper v3 Large Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "ERROR: Python 3.8+ required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "Installing dependencies..."

# ============================================================================
# Core dependencies
# ============================================================================

# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU only:
# pip install torch torchvision torchaudio

# Whisper and audio processing
pip install faster-whisper==1.0.3
pip install openai-whisper
pip install pydub==0.25.1
pip install noisereduce==3.0.2
pip install torchaudio

# Speaker diarization (optional but recommended)
pip install pyannote.audio==3.1.1

# Utilities
pip install numpy
pip install scipy

# FFmpeg (required for audio processing)
echo "Installing FFmpeg..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg portaudio19-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg portaudio
else
    echo "Please install FFmpeg manually"
fi

# ============================================================================
# Create requirements.txt
# ============================================================================

cat > requirements.txt << 'EOF'
# Core ML frameworks
torch>=2.1.0
torchaudio>=2.1.0

# Whisper engines
faster-whisper==1.0.3
openai-whisper>=20231117

# Audio processing
pydub==0.25.1
noisereduce==3.0.2
numpy>=1.24.0
scipy>=1.11.0

# Speaker diarization (optional)
pyannote.audio==3.1.1

# Utilities
tqdm>=4.66.0
EOF

echo "requirements.txt created"

# ============================================================================
# Download models (optional - will auto-download on first run)
# ============================================================================

echo ""
echo "Do you want to pre-download Whisper large-v3 model? (y/n)"
read -r download_model

if [[ "$download_model" == "y" ]]; then
    python3 << 'PYTHON'
from faster_whisper import WhisperModel
print("Downloading Whisper large-v3...")
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
print("Model downloaded successfully!")
PYTHON
fi

# ============================================================================
# GPU Check
# ============================================================================

echo ""
echo "Checking CUDA availability..."
python3 << 'PYTHON'
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA is available!")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("✗ CUDA not available - will use CPU (slower)")
PYTHON

# ============================================================================
# Create example config
# ============================================================================

cat > config_example.json << 'EOF'
{
  "model_size": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "beam_size": 5,
  "best_of": 5,
  "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "vad_filter": true,
  "vad_threshold": 0.5,
  "enable_denoise": true,
  "enable_normalize": true
  "enable_diarization": false
}
EOF

echo "config_example.json created"

# ============================================================================
# Create test script
# ============================================================================

cat > test_recognition.sh << 'EOF'
#!/bin/bash

# Test with various configurations

echo "Testing Whisper v3 Large..."

# Test 1: Basic transcription
echo "Test 1: Basic transcription"
python3 whisper_v3_recognizer.py test_audio.mp3

# Test 2: With SRT subtitles
echo ""
echo "Test 2: SRT subtitles"
python3 whisper_v3_recognizer.py test_audio.mp3 --format srt -o output.srt

# Test 3: With diarization
echo ""
echo "Test 3: Speaker diarization"
python3 whisper_v3_recognizer.py test_audio.mp3 --diarization --hf-token YOUR_TOKEN

# Test 4: Translation to English
echo ""
echo "Test 4: Translation"
python3 whisper_v3_recognizer.py test_audio.mp3 --task translate --language ru

# Test 5: High quality settings
echo ""
echo "Test 5: Maximum quality"
python3 whisper_v3_recognizer.py test_audio.mp3 \
    --model large-v3 \
    --beam-size 10 \
    --best-of 10 \
    --format json \
    -o result.json

EOF

chmod +x test_recognition.sh
echo "test_recognition.sh created"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Usage examples:"
echo ""
echo "1. Basic transcription:"
echo "   python3 whisper_v3_recognizer.py audio.mp3"
echo ""
echo "2. With SRT subtitles:"
echo "   python3 whisper_v3_recognizer.py audio.mp3 --format srt -o output.srt"
echo ""
echo "3. Specific language:"
echo "   python3 whisper_v3_recognizer.py audio.mp3 --language en"
echo ""
echo "4. Maximum quality:"
echo "   python3 whisper_v3_recognizer.py audio.mp3 --beam-size 10 --best-of 10"
echo ""
echo "5. With speaker diarization:"
echo "   python3 whisper_v3_recognizer.py audio.mp3 --diarization --hf-token YOUR_TOKEN"
echo ""
echo "For help:"
echo "   python3 whisper_v3_recognizer.py --help"
echo ""
echo "=========================================="