from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import torch.nn as nn
import torchaudio
import torch

class Whisper(nn.Module):
    def __init__(self):
        """
        Whisper encoder wrapper for feature extraction
        
        Attributes:
            processor: Whisper audio processor (mel spectrogram conversion)
            model: Encoder-only portion of Whisper-large-v3
            torch_dtype: Automatic precision selection (FP16/FP32)
            
        Methods:
            forward: Process audio waveforms into embeddings
        """

        super(Whisper, self).__init__()
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        # Load Hugging Face implementation for encoder access
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model = model.model.get_encoder()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def forward(self, x, sampling_rate):
        """
        Process audio through Whisper pipeline
        Args:
            x: (batch, samples) audio tensor
            sampling_rate: input sample rate (16kHz expected)
        Returns:
            encoder_outputs: Last hidden state embeddings
        """
        # Convert raw audio to mel spectrogram features
        inputs = self.processor(x.cpu().numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding="max_length")

        inputs = {key: value.to(x.device, dtype=self.torch_dtype) for key, value in inputs.items()}

        outputs = self.model(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask", None),  # Optional if available
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True  # Ensures outputs are returned as a dictionary
        )

        return outputs