from canary import CanaryEncoder

import nemo.collections.asr as nemo_asr
import torch.nn as nn
import torchaudio
import torch

if __name__ == "__main__":
    device =  'cuda:0' if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    waveform, sample_rate = torchaudio.load('/home/chaseez/nobackup/archive/LING361-Final/data/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav')
    waveform = waveform.to(device)
    waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    signal_length = torch.tensor([waveform.shape[-1]], dtype=torch.int64, device=device)
    asr_model = CanaryEncoder().to(device)

    # Ensure sample rate matches model's expected rate
    if sample_rate != asr_model.preprocessor._cfg.sample_rate:
        raise ValueError(f"Sample rate {sample_rate} does not match model's expected rate {asr_model.preprocessor._cfg.sample_rate}")
    
    clipped_wav = waveform.squeeze(1)#[:300000]
    clipped_signal_length = torch.tensor([clipped_wav.shape[-1]], dtype=torch.int64, device=device)

    embd, length = asr_model(clipped_wav, clipped_signal_length)
    print(f'{(signal_length / length).item():.3f}Hz sampling')
    print(embd.shape, length) 
