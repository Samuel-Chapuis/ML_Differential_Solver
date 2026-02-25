# models.py
import torch
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import glob
import sys
from pathlib import Path
import os
files_dir = Path(__file__).parent.parent / 'ML_Development'
sys.path.insert(0, str(files_dir))


# CNN Controller for NFTM
# class CNNController(nn.Module):
#     def __init__(self, field_size):
#         super().__init__()
#         self.field_size = field_size

#         self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
#         self.bn1   = nn.BatchNorm1d(32)
#         # self.dropout1 = nn.Dropout(0.1)

#         self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
#         self.bn2   = nn.BatchNorm1d(64)
#         # self.dropout2 = nn.Dropout(0.1)

#         self.conv3 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
#         self.bn3   = nn.BatchNorm1d(32)
#         # self.dropout3 = nn.Dropout(0.1)

#         self.conv_out = nn.Conv1d(32, 1, kernel_size=5, padding=2)


#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.xavier_uniform_(m.weight, gain=0.1)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, field):
#         B, N = field.shape
#         x = field.unsqueeze(1)  # (B,1,N)
         
#         x = torch.relu(self.bn1(self.conv1(x)))
#         # x = self.dropout1(x)

#         x = torch.relu(self.bn2(self.conv2(x)))
#         # x = self.dropout2(x)

#         x = torch.relu(self.bn3(self.conv3(x)))
#         # x = self.dropout3(x)

#         delta = self.conv_out(x).squeeze(1)  # (B, N)
#         pred_next_field = field + delta
#         return pred_next_field



class TemporalCNNAttention(nn.Module):
    """
    Computes attention over the history window (time dimension W) for every 
    spatial point (P). This layer uses CNNs to extract features and then 
    determines which past field states are most relevant to the current prediction.
    """
    def __init__(self, window_size, embed_dim, n_points):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Initial 1D CNN to map the single input channel to the embedding dimension (E) 
        # and extract local spatial features (across P).
        self.embedding = nn.Sequential(
            nn.Conv1d(1, embed_dim, 3, padding=1),
            nn.GELU()
        )
        
        # Linear projections (1x1 Convs) for Attention components (Query, Key, Value)
        self.query_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.key_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.val_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        
        # Normalization layer after the attention operation
        self.norm = nn.GroupNorm(4, embed_dim)

    def forward(self, u_history):
        # u_history shape: (B, W, P) -> (Batch, Window_Size, Points)
        B, W, P = u_history.shape
        # Flatten B and W dimensions to apply Conv1D over all W frames simultaneously
        flat_hist = u_history.contiguous().view(B * W, 1, P)
        
        # 1. Feature Extraction (CNN Embedding): Embeds the spatial information of all frames
        # Output shape after embedding: (B*W, E, P)
        features_flat = self.embedding(flat_hist)
        # Output shape: (B, W, E, P)
        features = features_flat.reshape(B, W, self.embed_dim, P)


        # 2. Attention Setup
        # Query (Q): Generated ONLY from the last frame $u_t$ (causality)
        last_frame_feat = features[:, -1, :, :] # (B, E, P)
        Q = self.query_proj(last_frame_feat)

        # Key (K) and Value (V): Generated from all frames in the window ($u_{t-W+1}$ to $u_t$)
        K = self.key_proj(features_flat).reshape(B, W, self.embed_dim, P) 
        V = self.val_proj(features_flat).reshape(B, W, self.embed_dim, P)

        # 3. Compute Attention Scores (Dot product Q * K^T)
        # Calculates relevance of each historical frame (W) for every spatial point (P)
        scores = torch.einsum('bep,bwe p->bwp', Q, K) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=1) # Softmax ensures weights sum to 1 across the W dimension

        # 4. Context Vector: Weighted sum of V using attention weights
        # This is the aggregated history feature map (B, E, P)
        context = torch.einsum('bwp,bwep->bep', attn_weights, V)
        
        out = self.out_proj(context)
        
        # Residual connection: Add the processed context back to the original last frame features
        return self.norm(out + last_frame_feat)


class CNNController(nn.Module):
    """
    The main Recurrent CNN Cell. Takes a history window (W) and predicts the 
    correction ($\Delta$) to the last state ($u_t$) to get the next state ($u_{t+1}$).
    This class combines the Temporal CNN Attention mechanism with a CNN decoder.
    """
    def __init__(self, field_size, window_size, corr_clip=0.15):
        super().__init__()
        self.corr_clip = corr_clip
        embed_dim = 32
        
        # 1. Temporal Attention mechanism to fuse history features extracted by CNNs
        self.attn = TemporalCNNAttention(window_size, embed_dim, field_size)
        
        # 2. CNN Decoder: Stack of 1D Convs to map the high-level context features 
        # (E channels) back down to the single-channel correction ($\Delta$).
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, 32, 5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 1, 5, padding=2) # Final layer output is the correction $\Delta$
        )
        
        # Stability initialization: Initialize the last layer to output near-zero
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, u_history):
        # u_history shape: (B, W, P)
        u_last = u_history[:, -1:, :] # (B, 1, P) - The current state, $u_t$
        
        # Get context-aware features from the attention block (B, E, P)
        context_features = self.attn(u_history)
        
        # Decode features into the raw correction $\Delta$ (B, 1, P)
        raw_correction = self.decoder(context_features) 
        
        # Tanh clipping: Limits the magnitude of the correction $\Delta$ for numerical stability
        correction = torch.tanh(raw_correction) * self.corr_clip
        
        # Final Recurrent Prediction: $u_{t+1} = u_t + \Delta$
        u_next = u_last.squeeze(1) + correction.squeeze(1) # Resulting shape: (B, P)
        return u_next # Returns the predicted next state $u_{t+1}$


class RNNControllerPatch(nn.Module):
    """
    Input: temporal patch (chunk_size, patch_size) + viscosity
    -> scalar prediction.
    patch: (B, seq_len, patch_size)
    nu   : (B, 1)
    """
    def _init_(self, patch_size: int, hidden_size: int = 64, rnn_type: str = "LSTM"):
        super()._init_()
        input_size = patch_size + 1  # patch + nu
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, patch_seq, nu):
        # patch_seq: (B, seq_len, patch_size)
        # nu      : (B, 1)
        nu_expanded = nu.unsqueeze(1).expand(-1, patch_seq.size(1), -1)  # (B, seq_len, 1)
        x = torch.cat([patch_seq, nu_expanded], dim=-1)                  # (B, seq_len, patch_size+1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]                                             # (B, hidden)
        y = self.fc(last)                                                # (B, 1)
        return y.squeeze(1)



class TransformerController(nn.Module):
    """
    Input: temporal patch sequence + viscosity
    patch_seq: (B, seq_len, patch_size)
    nu       : (B, 1)
    -> scalar prediction with learned temporal weights (attention).
    """
    def __init__(self, patch_size: int, hidden_size: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # On encode chaque (patch + nu) en un embedding
        self.embed = nn.Sequential(
            nn.Linear(patch_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Score d'attention pour chaque time step
        self.attn = nn.Linear(hidden_size, 1)

        # Tête finale pour prédire le scalaire à partir du contexte agrégé
        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, patch_seq, nu):
        # patch_seq: (B, seq_len, patch_size)
        # nu      : (B, 1)
        B, L, P = patch_seq.shape
        assert P == self.patch_size

        # On répète nu sur la dimension temporelle
        nu_expanded = nu.unsqueeze(1).expand(-1, L, 1)      # (B, L, 1)
        x = torch.cat([patch_seq, nu_expanded], dim=-1)     # (B, L, P+1)

        # Encodage pas de temps par pas de temps
        h = self.embed(x)                                   # (B, L, hidden)

        # Scores d'attention pour chaque time step
        scores = self.attn(h).squeeze(-1)                   # (B, L)
        weights = torch.softmax(scores, dim=-1)             # (B, L)

        # Combinaison pondérée des embeddings temporels
        context = (h * weights.unsqueeze(-1)).sum(dim=1)    # (B, hidden)

        # Prédiction finale
        y = self.fc_out(context)                            # (B, 1)
        return y.squeeze(1)                                 # (B,)
    
    
class TransformerController2D(nn.Module):
    """
    Input:
      patch_seq: (B, L, P)  sequence temporelle de "patch tokens"
      nu      : (B, 1)
        Output:
            field   : (B, C, H, W)
    """
    def __init__(self, patch_size: int, H: int, W: int,
                 hidden_size: int = 128, base_ch: int = 64, out_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.H, self.W = H, W
        self.out_channels = out_channels

        # encode chaque pas de temps: (patch + nu) -> hidden
        self.embed = nn.Sequential(
            nn.Linear(patch_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # attention temporelle (pooling pondéré)
        self.attn = nn.Linear(hidden_size, 1)

        # projection latent -> petit feature map
        # on choisit une résolution de départ (H0, W0) = (H/8, W/8) arrondie
        H0 = max(2, H // 8)
        W0 = max(2, W // 8)
        self.H0, self.W0 = H0, W0
        C0 = base_ch

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, C0 * H0 * W0),
            nn.ReLU(),
        )

        # décodeur: upsample x2 jusqu'à (H,W)
        # On utilise Upsample+Conv (souvent plus stable que ConvTranspose)
        def up_block(cin, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        self.dec1 = up_block(C0, C0)       # -> ~H0*2
        self.dec2 = up_block(C0, C0 // 2)  # -> ~H0*4
        self.dec3 = up_block(C0 // 2, C0 // 4)  # -> ~H0*8

        self.out = nn.Conv2d(C0 // 4, out_channels, kernel_size=1)

    def forward(self, patch_seq, nu):
        B, L, P = patch_seq.shape
        assert P == self.patch_size

        nu_expanded = nu.unsqueeze(1).expand(-1, L, 1)  # (B,L,1)
        x = torch.cat([patch_seq, nu_expanded], dim=-1) # (B,L,P+1)

        h = self.embed(x)                      # (B,L,Hid)
        scores = self.attn(h).squeeze(-1)      # (B,L)
        weights = torch.softmax(scores, dim=-1)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B,Hid)

        z = self.proj(context)                 # (B, C0*H0*W0)
        z = z.view(B, -1, self.H0, self.W0)    # (B,C0,H0,W0)

        y = self.dec1(z)
        y = self.dec2(y)
        y = self.dec3(y)
        y = self.out(y)                        # (B,C,?,?)

        # ajuste exactement à (H,W) si besoin
        y = F.interpolate(y, size=(self.H, self.W), mode="bilinear", align_corners=False)
        return y
    

## Akash - Alex Version

# ==========================================
# Model with Stability Improvements
# ==========================================
class CausalTemporalAttention(nn.Module):
    def __init__(self, window_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Sequential(
            nn.Conv1d(1, embed_dim, 3, padding=1),
            nn.GELU()
        )
        self.query_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.key_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.val_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        self.norm = nn.GroupNorm(4, embed_dim)

    def forward(self, u_history):
        B, W, P = u_history.shape
        flat_hist = u_history.contiguous().view(B * W, 1, P)
        features = self.embedding(flat_hist).view(B, W, self.embed_dim, P)

        last_frame_feat = features[:, -1, :, :]
        Q = self.query_proj(last_frame_feat)

        flat_feat = features.view(B*W, self.embed_dim, P)
        K = self.key_proj(flat_feat).view(B, W, self.embed_dim, P)
        V = self.val_proj(flat_feat).view(B, W, self.embed_dim, P)

        scores = torch.einsum('bcp,bwcp->bwp', Q, K) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=1)

        context = torch.einsum('bwp,bwcp->bcp', attn_weights, V)
        out = self.out_proj(context)
        return self.norm(out + last_frame_feat)


# class ImprovedBurgersNet(nn.Module):
#     def __init__(self, window_size=4, corr_clip=0.1):
#         super().__init__()
#         self.corr_clip = corr_clip
#         embed_dim = 32
        
#         self.attn = CausalTemporalAttention(window_size, embed_dim)
#         self.decoder = nn.Sequential(
#             nn.Conv1d(embed_dim, 32, 5, padding=2),
#             nn.BatchNorm1d(32),
#             nn.GELU(),
#             nn.Conv1d(32, 1, 5, padding=2)
#         )
        
#         # Initialize last layer to output near-zero initially
#         # This helps with stability at the start of training
#         nn.init.zeros_(self.decoder[-1].weight)
#         nn.init.zeros_(self.decoder[-1].bias)

#     def forward(self, u_history):
#         u_last = u_history[:, -1:, :]
#         context_features = self.attn(u_history)
#         raw_correction = self.decoder(context_features)
        
#         # Tanh clipping for bounded corrections
#         correction = torch.tanh(raw_correction) * self.corr_clip
        
#         u_next = u_last + correction
#         return u_next


class ImprovedBurgersNet(nn.Module):
    """
    TCAN mejorado con condicionamiento en viscosidad.
    Compatible con tu código de entrenamiento actual.
    """
    def __init__(self, window_size=4, corr_clip=0.1, use_viscosity=True):
        super().__init__()
        self.corr_clip = corr_clip
        self.use_viscosity = use_viscosity
        embed_dim = 32
        
        # Attention temporal (sin cambios)
        self.attn = CausalTemporalAttention(window_size, embed_dim)
        
        # NUEVO: Viscosity embedding y FiLM conditioning
        if use_viscosity:
            self.nu_embed = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Linear(32, embed_dim)
            )
            self.film_gamma = nn.Linear(embed_dim, embed_dim)
            self.film_beta = nn.Linear(embed_dim, embed_dim)
        
        # Decoder (sin cambios en estructura)
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, 5, padding=2)
        )
        
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, u_history, nu=None):
        """
        Args:
            u_history: (B, W, P) - History window
            nu: (B, 1) - Viscosity [NUEVO parámetro opcional]
        """
        u_last = u_history[:, -1:, :]  # (B, 1, P)
        
        # Extract features con attention
        context_features = self.attn(u_history)  # (B, embed_dim, P)
        
        # NUEVO: Condicionar en viscosidad si está disponible
        if self.use_viscosity and nu is not None:
            nu_emb = self.nu_embed(nu)  # (B, embed_dim)
            gamma = self.film_gamma(nu_emb).unsqueeze(-1)  # (B, embed_dim, 1)
            beta = self.film_beta(nu_emb).unsqueeze(-1)    # (B, embed_dim, 1)
            # FiLM: features' = gamma * features + beta
            context_features = gamma * context_features + beta
        
        # Decode
        raw_correction = self.decoder(context_features)
        correction = torch.tanh(raw_correction) * self.corr_clip
        
        u_next = u_last + correction
        return u_next