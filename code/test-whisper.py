from whisper import Whisper

import torchaudio
import torch
import math

if __name__ == "__main__":
    device =  'cuda:0' if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print('loading the audio...')
    waveform, sample_rate = torchaudio.load('/home/chaseez/nobackup/archive/LING361-Final/data/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav')
    waveform = waveform.to(device)
    waveform = waveform  # Add batch dimension
    
    tot_len = waveform.shape[-1]
    segments = math.floor(tot_len / sample_rate)

    # Truncate the remainder of the last second
    clipped_wav = waveform[:,:segments * sample_rate]

    # Resize for batches of 3 seconds of audio
    clipped_wav = clipped_wav.reshape(-1, sample_rate * 3)
    clipped_wav.shape, clipped_wav.dtype

    print('initializing Whisper model...')
    asr_model = Whisper().to(device)

    print('running inference on Whisper...')
    batch_size = 256
    start = 0

    for i in range((clipped_wav.shape[0] // batch_size)+1):
        end = start + batch_size
        # print('starting raw shape:', clipped_wav[start:end,:].shape)
        embd_matrix = asr_model(clipped_wav[start:end,:], sample_rate).last_hidden_state
        
        start += end

    