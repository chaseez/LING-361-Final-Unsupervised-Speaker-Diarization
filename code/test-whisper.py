from torch.utils.data import DataLoader
from kneed import KneeLocator
from whisper import Whisper

import numpy as np
import torchaudio
import torch
import math
import json


if __name__ == "__main__":
    """
    Process audio using Whisper encoder and perform singular value decomposition analysis.

    Steps:
    1. Load and segment audio into 3-second windows
    2. Initialize Whisper encoder model
    3. Extract embeddings in batches (optimized for GPU)
    4. Perform SVD on flattened embeddings
    5. Calculate optimal component count using KneeLocator
    6. Save singular values and knee point to JSON
    """

    # Device configuration (CUDA/CPU)
    device =  'cuda:0' if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print('loading the audio...', flush=True)
    waveform, sample_rate = torchaudio.load('../data/amicorpus/TS3012d/audio/TS3012d.Mix-Headset.wav')
    waveform = waveform.to(device)
    waveform = waveform  # Add batch dimension
    
    tot_len = waveform.shape[-1]
    segments = math.floor(tot_len / (sample_rate * 3))

    # Truncate the remainder of the last second
    clipped_wav = waveform[:,:segments * sample_rate * 3]

    # Resize for batches of 3 seconds of audio
    clipped_wav = clipped_wav.reshape(-1, sample_rate * 3)

    print('initializing Whisper model...', flush=True)
    asr_model = Whisper().to(device)

    print('running inference on Whisper...', flush=True)
    batch_size = 1024

    loader = DataLoader(clipped_wav, batch_size)

    embd = []
    for wav in loader:
        embd.append(asr_model(wav, sample_rate).last_hidden_state)

    embd = torch.cat(embd)

    embd = embd.flatten(0,1)[::100,:].contiguous()

    print('starting SVD', flush=True)
    u,s,v = np.linalg.svd(embd.cpu().to(torch.float32))

    print('Generating Knee', flush=True)
    knee = KneeLocator(np.arange(s.shape[0]), s, S=1.0, curve='concave', direction='decreasing')

    print(knee.knee, knee.elbow, s.shape[0])
    info = {
        'singular values': s.tolist(),
        'knee': int(knee.knee)
    }

    with open('foo.json', 'w') as f:
        json.dump(info, f)