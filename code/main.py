from torch.utils.data import DataLoader
from argparse import ArgumentParser
from canary import CanaryEncoder
from kneed import KneeLocator
from torch.optim import AdamW
from whisper import Whisper
from pathlib import Path
from optim import FISTA
from tqdm import tqdm

import torch.nn as nn
import numpy as np
import torchaudio
import torch
import json
import math

def shrinkage_operator(u, tresh):
        return torch.sign(u) * torch.maximum(torch.abs(u) - tresh, torch.tensor(0.0, device=u.device))

def project(u):
    u = u / torch.norm(u, p=2)
    # Test it between -1 and 1. Maybe 0 and 1 are better for mathematical properties??
    return torch.clamp(u, 0, 1)

def jitter_loss(A):
    return 0

parser = ArgumentParser()
parser.add_argument('-e', '--encoder', type=str, default='whisper', help='Specify which encoder: Whisper or Canary')
parser.add_argument('-n', '--epochs', type=int, default=10000, help='Specify how many epochs.')
argv = parser.parse_args()

if __name__ == '__main__':
    encoder = argv.encoder.lower()
    steps = argv.epochs


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = None
    dim_embd = None
    if 'can' in encoder:
        model = CanaryEncoder().to(device)
        dim_embd = 1024
    else:
        model = Whisper().to(device)
        dim_embd = 1280

    
    # Found on page 3 at the end of the page
    activation_weight = 0.2424
    embd_basis_weight = 0.3366
    jitter_loss_weight = 0.06

    # I DUNNO WHAT THESE DO
    embd_lagrange_multiplier = 2
    activation_lagrange_multiplier = 2

    files = list(Path('../data/amicorpus').rglob('*Mix-Headset.wav'))

    print(len(files))
    
    for file in files:
        print('loading audio...', flush=True)
        waveform, sample_rate = torchaudio.load(file)
        waveform = waveform.to(device)

        embeddings = None

        tot_len = waveform.shape[-1]
        segments = math.floor(tot_len / (sample_rate * 6))

        # put files in 3 second windows
        clipped_wav = waveform[:,:segments * sample_rate * 6]

        print('Getting embeddings...', flush=True)
        if isinstance(model, Whisper):
            clipped_wav = clipped_wav.reshape(-1, sample_rate * 3)

            # biggest batch size an A100 can handle (3s * 1024 / 60 = 51.2 minutes of audio)
            loader = DataLoader(clipped_wav, batch_size=1024)
            embeddings = []
            for wav in loader:
                embeddings.append(model(wav, sample_rate).last_hidden_state)
            embeddings = torch.cat(embeddings, dim=0)
        else:
            clipped_signal_length = torch.full((clipped_wav.shape[0],), clipped_wav.shape[1], device=device).contiguous()
            embeddings = model(clipped_wav, clipped_signal_length)

            # Rearrange shape from (batch, 1024, T) => (batch, T, 1024)
            embeddings = embeddings.permute(0,2,1).contiguous()

        # flatten the shape to be (T, model_dim)
        embeddings = embeddings.flatten(start_dim=0, end_dim=1)
        embd = embeddings[::100,:].contiguous().cpu().to(torch.float32)

        # SVD is used to calculate the number of speakers
        print('calculating SVD...', flush=True)
        _, s, _ = np.linalg.svd(embd)

        num_speakers = s * 2

        embd_basis_matrix = nn.Parameter(torch.nn.init.kaiming_normal(torch.randn((dim_embd, num_speakers)))).to(device)
        activation_matrix = None
        if isinstance(model, Whisper):
            # (k, T)
            activation_matrix = nn.Parameter(torch.nn.init.kaiming_normal(torch.randn((num_speakers, 2 * 1500)))).to(device)
        else:
            activation_matrix = nn.Parameter(torch.nn.init.kaiming_normal(torch.randn((num_speakers, embeddings.shape[1])))).to(device)


        embd_optim = AdamW(embd_basis_matrix.parameters(), lr=0.001)
        activation_optim = AdamW(activation_matrix.parameters(), lr=0.001)

        batch_size = None
        if isinstance(model, Whisper):
            # (batch, 1500, 1280) for 3 second window
            batch_size = 2 * 1500
        else:
            # (batch, 1024, T) => (batch, T, 1024) for 6 second window
            batch_size = embeddings.shape[1]

        loader = DataLoader(embeddings, batch_size=batch_size)

        print('Starting Training', flush=True)
        for step in range(steps):
            # CANNOT BE BATCHED YET 
            for embd in tqdm(loader):
                print(embd.size, flush=True)
                # Reshape embd to be MxT instead of TxM
                y_hat = embd.T - (embd_basis_matrix @ activation_matrix.detach())
                term1 = torch.norm(y_hat)
                term2 = embd_basis_weight * torch.norm(embd_basis_matrix)
                term3 = activation_weight * torch.norm(activation_matrix.detach())
                term4 = jitter_loss_weight * jitter_loss(activation_matrix.detach())

                loss = term1 + term2 + term3 + term4

                loss.backward()
                embd_optim.step()
                embd_optim.zero_grad()

                with torch.no_grad():
                    embd_basis_matrix = project(shrinkage_operator(embd_basis_matrix, embd_lagrange_multiplier))

                y_hat = embd - (embd_basis_matrix.detach() @ activation_matrix)
                term1 = torch.norm(y_hat)
                term2 = embd_basis_weight * torch.norm(embd_basis_matrix.detach())
                term3 = activation_weight * torch.norm(activation_matrix)
                term4 = jitter_loss_weight * jitter_loss(activation_matrix)

                loss = term1 + term2 + term3 + term4

                loss.backward()
                activation_optim.step()
                activation_optim.zero_grad()

                with torch.no_grad():
                    activation_matrix = project(shrinkage_operator(activation_matrix, activation_lagrange_multiplier))

            
