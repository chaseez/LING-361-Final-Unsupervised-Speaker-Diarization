from argparse import ArgumentParser
from canary import CanaryEncoder
from torch.optim import Adam
from whisper import Whisper
from pathlib import Path
from optim import FISTA

import torch.nn as nn
import torchaudio
import torch
import math

def shrinkage_operator(u, tresh):
        return torch.sign(u) * torch.maximum(torch.abs(u) - tresh, torch.tensor(0.0, device=u.device))

def project(u):
    u = u / torch.norm(u, p=2)
    # Test it between -1 and 1. Maybe 0 and 1 are better for mathematical properties??
    return torch.clamp(u, -1, 1)

parser = ArgumentParser()
parser.add_argument('-e', '--encoder', type=str, default='whisper', help='Specify which encoder: Whisper or Canary')
parser.add_argument('-s', '--steps', type=int, default=10000, help='Specify how many optimization steps you want your model to learn.')
argv = parser.parse_args()

if __name__ == '__main__':
    encoder = argv.encoder.lower()
    steps = argv.steps


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = None
    if 'can' in encoder:
        model = CanaryEncoder().to(device)
    else:
        model = Whisper().to(device)

    activation_matrix = nn.Parameter(torch.randn((dim_embd, num_speakers)))
    embd_basis_matrix = nn.Parameter(torch.randn((num_speakers, timesteps)))
    
    # Found on page 3 at the end of the page
    activation_weight = 0.2424
    embd_basis_weight = 0.3366
    jitter_loss_weight = 0.06

    # I DUNNO WHAT THESE DO
    embd_lagrange_multiplier = 2
    activation_lagrange_multiplier = 2

    activation_optim = Adam(activation_matrix.parameters(), lr=0.001)
    embd_basis_matrix_optim = Adam(embd_basis_matrix.parameters(), lr=0.001)

    files = list(Path('../data/amicorpus').rglob('*Mix-Headset.wav'))

    print(len(files))
    
    for file in files:
        waveform, sample_rate = torchaudio.load(file)
        print(waveform.shape)

        tot_len = waveform.shape[-1]
        segments = math.floor(tot_len / sample_rate)

        # put files in 3 second windows
        clipped_wav = waveform[:,:segments * sample_rate]
        clipped_wav = clipped_wav.reshape(-1, sample_rate * 3)

        signal_length = torch.tensor([waveform.shape[-1]], dtype=torch.int64)

        embeddings = None
        if isinstance(model, Whisper):
            embeddings = model(clipped_wav, sample_rate)
        else:
            embeddings = model(clipped_wav, signal_length)

        # CANNOT BE BATCHED YET
        y_hat = embeddings - embd_basis_matrix @ activation_matrix.detach()
        term1 = torch.norm(y_hat)
        term2 = embd_basis_weight*torch.norm(embd_basis_matrix)
        term3 = activation_weight*torch.norm(embd_basis_matrix)
        term4 = jitter_loss_weight*torch.norm(embd_basis_matrix)

        loss = term1 + term2 + term3 + term4

        loss.backward()
        embd_basis_matrix_optim.step()
        embd_basis_matrix_optim.zero_grad()


        with torch.no_grad():
            embd_basis_matrix = project(shrinkage_operator(embd_basis_matrix, embd_lagrange_multiplier))

        y_hat = embeddings - embd_basis_matrix.detach() @ activation_matrix
        term1 = torch.norm(y_hat)
        term2 = embd_basis_weight*torch.norm(embd_basis_matrix)
        term3 = activation_weight*torch.norm(embd_basis_matrix)
        term4 = jitter_loss_weight*torch.norm(embd_basis_matrix)

        loss = term1 + term2 + term3 + term4

        loss.backward()
        activation_optim.step()
        activation_optim.zero_grad()

        with torch.no_grad():
            activation_matrix = project(shrinkage_operator(activation_matrix, activation_lagrange_multiplier))

        
