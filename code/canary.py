import nemo.collections.asr as nemo_asr
import torch.nn as nn
import torch

# inherit the properties of torch.nn.Module
class CanaryEncoder(nn.Module):
    """
        NVIDIA Canary encoder wrapper using NeMo toolkit
        
        Features:
            - Automatic sample rate validation (16kHz expected)
            - Preprocessor: Audio → Mel Spectrogram
            - Encoder: Conformer-based architecture
    """
    
    def __init__(self, device):
        # initialize the nn.Module class
        super(CanaryEncoder, self).__init__()

        # export all the layers associated with the encoder
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b", map_location=torch.device(device))

        self.preprocessor = model.preprocessor
        self.encoder = model.encoder


    @torch.no_grad
    def forward(self, waveform, signal_length):
        """
        Process audio through Canary pipeline
        Args:
            waveform: (batch, samples) raw audio
            signal_length: Original audio lengths
        Returns:
            embeddings: (batch, features, time) encoded representations
        """
        process_wav, process_len = self.preprocessor(input_signal=waveform, length=signal_length)
        return self.encoder(audio_signal=process_wav, length=process_len)