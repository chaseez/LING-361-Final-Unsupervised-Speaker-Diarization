from canary import CanaryEncoder

import torchaudio
import torch
import math

if __name__ == "__main__":
    device =  'cuda:0' if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    waveform, sample_rate = torchaudio.load('../data/amicorpus/TS3012d/audio/TS3012d.Mix-Headset.wav')
    waveform = waveform.to(device)
    
    tot_len = waveform.shape[-1]
    segments = math.floor(tot_len / (sample_rate * 6))

    # put files in 3 second windows
    clipped_wav = waveform[:,:segments * sample_rate]
    clipped_wav = clipped_wav.reshape(-1, sample_rate * 6)

    clipped_signal_length = torch.full((clipped_wav.shape[0],), clipped_wav.shape[1], device=device).contiguous()
    
    asr_model = CanaryEncoder().to(device)

    # Ensure sample rate matches model's expected rate
    if sample_rate != asr_model.preprocessor._cfg.sample_rate:
        raise ValueError(f"Sample rate {sample_rate} does not match model's expected rate {asr_model.preprocessor._cfg.sample_rate}")

    embd, length = asr_model(clipped_wav, clipped_signal_length)
    print((clipped_signal_length / length))
    print(embd.shape, length) 
