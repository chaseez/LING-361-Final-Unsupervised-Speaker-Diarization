from torch.utils.data import DataLoader
from argparse import ArgumentParser
from canary import CanaryEncoder
from whisper import Whisper
from pathlib import Path
from tqdm import tqdm

import torch.optim as optim
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
    return torch.clamp(u, 0, 1)

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
        T = 38
    else:
        model = Whisper().to(device)
        dim_embd = 1280
        # 3 second window
        T = 1 * 1500

    num_speakers = 18
    
    # Found on page 3 at the end of the page
    activation_weight = 0.2424
    embd_basis_weight = 0.3366
    jitter_loss_weight = 0.06

    embd_lagrange_multiplier = 0.001
    activation_lagrange_multiplier = 0.001

    files = list(Path('../data/voxconverse-master/audio').rglob('*.wav'))

    log_interval = 10
    

    for file in tqdm(files):
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
            loader = DataLoader(clipped_wav, batch_size=32)
            embeddings = []
            for wav in loader:
                embeddings.append(model(wav, sample_rate).last_hidden_state)
            embeddings = torch.cat(embeddings, dim=0)

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
                print(e, flush=True)
                continue
            # Rearrange shape from (batch, 1024, T) => (batch, T, 1024)
            embeddings = embeddings.permute(0,2,1).contiguous()
            T = embeddings.shape[1]

        # flatten the shape to be (T, model_dim)
        embeddings = embeddings.flatten(start_dim=0, end_dim=1)

        # embd_optim = FISTA([embd_basis_matrix], lr=0.01, lambda_=embd_lagrange_multiplier)
        # activation_optim = FISTA([activation_matrix], lr=0.01, lambda_=activation_lagrange_multiplier)

        loader = DataLoader(embeddings, batch_size=T)

        print('Starting Training', flush=True)
        with tqdm(range(steps*len(loader))) as pbar:
            for i, embd in enumerate(loader):
                embd_basis_matrix = torch.randn((dim_embd, num_speakers)).to(device)
                activation_matrix = torch.randn((num_speakers, T)).to(device)

                nn.init.kaiming_normal_(embd_basis_matrix)
                embd_basis_matrix.requires_grad_(True)

                nn.init.kaiming_normal_(activation_matrix)
                activation_matrix.requires_grad_(True)

                embd_optim = optim.AdamW([embd_basis_matrix], lr=0.01, weight_decay=embd_basis_weight)
                activation_optim = optim.AdamW([activation_matrix], lr=0.01, weight_decay=activation_weight)

                embd_scheduler = optim.lr_scheduler.CosineAnnealingLR(embd_optim, T_max=steps)
                activation_scheduler = optim.lr_scheduler.CosineAnnealingLR(embd_optim, T_max=steps)

                losses = [70.0]
                curr_loss = []

                for step in range(steps):
                    # if step == 0 and i == 0:
                    #     # SVD is used to calculate the number of speakers
                    #     print('calculating SVD...', flush=True)
                    #     _, s, _ = np.linalg.svd(embd.cpu().to(torch.float32))

                    #     # print('Generating Knee', flush=True)
                    #     knee = KneeLocator(np.arange(s.shape[0]), s, S=1.0, curve='concave', direction='decreasing')
                    #     print(knee.knee, flush=True)
                    pbar.update(1)
                    # Reshape embd to be MxT instead of TxM
                    y_hat = torch.norm(embd.T - (embd_basis_matrix @ activation_matrix.detach()))
                    loss = y_hat + jitter_loss_weight * jitter_loss(activation_matrix.detach())

                    loss.backward()
                    embd_optim.step()
                    embd_optim.zero_grad()

                    curr_loss.append(loss.item())

                    embd_basis_matrix.data = project(shrinkage_operator(embd_basis_matrix, embd_lagrange_multiplier))

                    y_hat = torch.norm(embd.T - (embd_basis_matrix.detach() @ activation_matrix))

                    loss = y_hat + jitter_loss_weight * jitter_loss(activation_matrix)

                    loss.backward()
                    activation_optim.step()
                    activation_optim.zero_grad()

                    curr_loss.append(loss.item())

                    activation_matrix.data = project(shrinkage_operator(activation_matrix, activation_lagrange_multiplier))

                    embd_scheduler.step()
                    activation_scheduler.step()

                    if step % log_interval == 0:
                        losses.append(np.mean(curr_loss))
                        pbar.set_description(f'Epoch {step+1}/{steps} loss: {losses[-1]:.3f}')
                        curr_loss = []

                results_path = Path('results')
                results_path.mkdir(exist_ok=True)

                results = results_path.joinpath(f"{file.name.removesuffix('.wav')}-{i}-{encoder}-{steps}-epochs-{losses[-1]:.4f}.json")
                with open(results, 'w') as f:
                    json.dump({'losses': losses}, f)

                model_path = Path('models')
                model_path.mkdir(exist_ok=True)

                activation_path = model_path.joinpath(f"{file.name.removesuffix('.wav')}-{i}-{encoder}-activation-{step}-epochs-{losses[-1]:.4f}.pt")
                with open(activation_path, 'wb') as f:
                    torch.save(activation_matrix, f)

                basis_path = model_path.joinpath(f"{file.name.removesuffix('.wav')}-{i}-{encoder}-basis-{step}-epochs-{losses[-1]:.4f}.pt")
                with open(basis_path, 'wb') as f:
                    torch.save(embd_basis_matrix, f)
                
