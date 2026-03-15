import torch
import gc
from qwen_asr import Qwen3ASRModel

ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"

# A singleton class that handles loading the Qwen3ASRModel and Qwen3ForcedAligner once 
# and reusing them for all requests.
class ASREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ASREngine, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def load_model(self):
        if self.model is not None:
            return

        print("Loading Qwen3-ASR Model...")
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map=device,
            forced_aligner=FORCED_ALIGNER_PATH,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device,
            ),
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
        print(f"Model loaded on {device}")

    def transcribe(self, audio_data, language=None):
        if self.model is None:
            self.load_model()
        
        try:
            # audio_data can be a file path, bytes, or (wav, sr)
            # We explicitly pass context="" to ensure no history is kept
            # Qwen3-ASR is a multi-modal model, so it *could* have context,
            # but we want to stay lean.
            with torch.inference_mode():
                results = self.model.transcribe(
                    audio=audio_data,
                    context="",
                    language=language,
                    return_time_stamps=False,
                )
            return results[0] if results else None
        finally:
            # Memory Management: Force cleanup after each transcription
            # This is crucial for MPS (Metal) on Mac which tends to hold onto memory
            self.clear_memory()

    def clear_memory(self):
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            # Crucial for preventing the 7GB -> 12GB growth on Mac
            try:
                torch.mps.empty_cache()
            except:
                pass
        print("DEBUG: Memory cleared.")

asr_engine = ASREngine()
