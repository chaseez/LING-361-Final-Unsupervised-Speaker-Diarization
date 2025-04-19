from torch.utils.data import DataLoader
from argparse import ArgumentParser
from canary import CanaryEncoder
from kneed import KneeLocator
from torch.optim import AdamW
from whisper import Whisper
from pathlib import Path
from optim import FISTA
from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchaudio
import torch
import json
import math

@torch.no_grad()
def shrinkage_operator(u, tresh):
        return torch.sign(u) * torch.maximum(torch.abs(u) - tresh, torch.tensor(0.0, device=u.device))

@torch.no_grad()
def project(u):
    u = u / torch.norm(u, p=2)
    # Test it between -1 and 1. Maybe 0 and 1 are better for mathematical properties??
    return torch.clamp(u, -1, 1)

def jitter_loss(A):
    return torch.mean(torch.abs(A[:-1] - A[1:]))

parser = ArgumentParser()
parser.add_argument('-e', '--encoder', type=str, default='whisper', help='Specify which encoder: Whisper or Canary')
parser.add_argument('-n', '--epochs', type=int, default=10, help='Specify how many epochs.')
argv = parser.parse_args()

if __name__ == '__main__':
    encoder = argv.encoder.lower()
    steps = argv.epochs


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = None
    dim_embd = None
    if 'can' in encoder:
        model = CanaryEncoder(device).to(device)
        dim_embd = 1024
    else:
        model = Whisper().to(device)
        dim_embd = 1280

    
    # Found on page 3 at the end of the page
    activation_weight = 0.2424
    embd_basis_weight = 0.3366
    jitter_loss_weight = 0.06

    embd_lagrange_multiplier = 0.001
    activation_lagrange_multiplier = 0.001

    files = list(Path('../data/amicorpus').rglob('*Mix-Headset.wav'))

    print(len(files))
    
    for file in files:
        print('loading audio...', flush=True)
        print(file.name)
        waveform, sample_rate = torchaudio.load(file)
        waveform = waveform.to(device)

        embeddings = None

        tot_len = waveform.shape[-1]
        segments = math.floor(tot_len / (sample_rate * 3))

        # put files in 3 second windows
        clipped_wav = waveform[:,:segments * sample_rate * 3]

        print('Getting embeddings...', flush=True)
        if isinstance(model, Whisper):
            clipped_wav = clipped_wav.reshape(-1, sample_rate * 3)

            # biggest batch size an A100 can handle (3s * 1024 / 60 = 51.2 minutes of audio)
            loader = DataLoader(clipped_wav, batch_size=1024)
            embeddings = []
            for wav in loader:
                embeddings.append(model(wav, sample_rate).last_hidden_state)
            embeddings = torch.cat(embeddings, dim=0)
            
            # 3 second window
            T = 1 * 1500

        else:
            clipped_wav = clipped_wav.reshape(-1, sample_rate * 3)
            loader = DataLoader(clipped_wav, batch_size=32)
            embeddings = []
            for wav in loader:
                clipped_signal_length = torch.full((wav.shape[0],), wav.shape[1], device=device).contiguous()
                embeddings.append(model(wav, clipped_signal_length)[0])
            try:
                embeddings = torch.cat(embeddings, dim=0)
            except RuntimeError as e:
                print(e)
                continue
            # Rearrange shape from (batch, 1024, T) => (batch, T, 1024)
            embeddings = embeddings.permute(0,2,1).contiguous()
            T = embeddings.shape[1]

        # flatten the shape to be (T, model_dim)
        embeddings = embeddings.flatten(start_dim=0, end_dim=1)

        # SVD is used to calculate the number of speakers
        print('calculating SVD...', flush=True)
        _, s, _ = np.linalg.svd(embeddings.cpu().to(torch.float32))

        # print('Generating Knee', flush=True)
        knee = KneeLocator(np.arange(s.shape[0]), s, S=1.0, curve='concave', direction='decreasing')
        print(knee.knee)
        num_speakers = 30

        print('Num Speakers:', num_speakers, flush=True)

        embd_basis_matrix = torch.randn((dim_embd, num_speakers)).to(device)
        activation_matrix = torch.randn((num_speakers, T)).to(device)

        nn.init.kaiming_normal_(embd_basis_matrix)
        embd_basis_matrix.requires_grad_(True)

        nn.init.kaiming_normal_(activation_matrix)
        activation_matrix.requires_grad_(True)

        embd_optim = AdamW([embd_basis_matrix], lr=0.1)
        activation_optim = AdamW([activation_matrix], lr=0.1)

        # embd_optim = FISTA([embd_basis_matrix], lr=0.01, lambda_=embd_lagrange_multiplier)
        # activation_optim = FISTA([activation_matrix], lr=0.01, lambda_=activation_lagrange_multiplier)

        loader = DataLoader(embeddings, batch_size=T)

        losses = []

        curr_loss = []

        log_interval = 10

        print('Starting Training', flush=True)
        with tqdm(range(steps*len(loader))) as pbar:
            for step in range(steps):
                # CANNOT BE BATCHED YET 
                # Doesn't need to be batched, just chunked into 6 segments
                for i, embd in enumerate(loader):
                    pbar.update(1)
                    # Reshape embd to be MxT instead of TxM
                    y_hat = torch.mean((embd.T - (embd_basis_matrix @ activation_matrix.detach()))**2)

                    loss = y_hat + jitter_loss(activation_matrix.detach())

                    loss.backward()
                    embd_optim.step()
                    embd_optim.zero_grad()

                    curr_loss.append(loss.item())

                    embd_basis_matrix.data = project(shrinkage_operator(embd_basis_matrix, embd_lagrange_multiplier))

                    y_hat = torch.mean((embd.T - (embd_basis_matrix.detach() @ activation_matrix))**2)

                    loss = y_hat + jitter_loss(activation_matrix)

                    loss.backward()
                    activation_optim.step()
                    activation_optim.zero_grad()

                    curr_loss.append(loss.item())

                    activation_matrix.data = project(shrinkage_operator(activation_matrix, activation_lagrange_multiplier))

                    if i % log_interval == 0:
                        losses.append(np.mean(curr_loss))
                        pbar.set_description(f'Epoch {step+1}/{steps} loss: {losses[-1]:.3f}')
                        curr_loss = []
        with open(f"results/{file.name.replace('.wav', '.json')}", 'w') as f:
            json.dump({'losses': losses}, f)

            
