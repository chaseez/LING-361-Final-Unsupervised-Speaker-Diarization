import nemo.collections.asr as nemo_asr
import torch.nn as nn
import torch

# inherit the properties of torch.nn.Module
class CanaryEncoder(nn.Module):
    def __init__(self):
        # initialize the nn.Module class
        super(CanaryEncoder, self).__init__()

        # export all the layers associated with the encoder
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b", map_location=torch.device('cuda'))

        self.preprocessor = model.preprocessor
        self.encoder = model.encoder


    @torch.no_grad
    def forward(self, waveform, signal_length):
        process_wav, process_len = self.preprocessor(input_signal=waveform, length=signal_length)
        return self.encoder(audio_signal=process_wav, length=process_len)