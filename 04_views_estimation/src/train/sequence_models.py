import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GRUModel(nn.Module):
    """Gated-RNN branch of the 3-model sequence ensemble (see TRAIN.md) -- picked over LSTM
    since they're the same gated-RNN family and wouldn't add real diversity having both."""

    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class CNNModel(nn.Module):
    """1D-convolution branch -- local-pattern mechanism, distinct from GRU's recurrence and
    PatchTST's attention. kernel_size=21 (padding=10 keeps the sequence length constant) so the
    2-layer stack's receptive field is 2*(21-1)+1=41 raw days -- originally kernel_size=5 gave
    only a 9-day receptive field, too narrow to see much given the 21-day prediction horizon.
    Widened via a bigger kernel rather than more stacked layers deliberately: on this noisy a
    target, more layers means more chained nonlinearities (conv+ReLU repeated), the same kind of
    capacity increase that made the MLP sweep get worse with depth -- a wider single-layer
    filter adds capacity more like extending a moving-average window, which seemed safer."""

    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=21, padding=10)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels * 2, 1)
        self.act = nn.ReLU()

    def forward(self, x):  # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class PatchTSTModel(nn.Module):
    """Attention branch -- splits the 252-day sequence into 21 patches of 12 days each (252/12
    is exact) instead of attending over every single day, matching PatchTST's patch-based
    approach rather than a plain Transformer."""

    def __init__(self, seq_len=252, patch_len=12, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        assert seq_len % patch_len == 0
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model * 2
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (batch, seq_len)
        b = x.shape[0]
        x = x[:, : self.num_patches * self.patch_len].reshape(b, self.num_patches, self.patch_len)
        x = self.embed(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


class _TorchSequenceWrapper:
    """Wraps a raw nn.Module in the same fit(X, y)/predict(X) interface every FACTOR_MODELS
    entry uses, so sequence_oof.py's walk-forward loop can stay generic. Trains on GPU
    automatically whenever torch.cuda.is_available() -- no per-model device wiring needed."""

    def __init__(self, model_fn, epochs=15, batch_size=2048, lr=1e-3):
        self.model_fn = model_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None

    def fit(self, X, y):
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        self.model = self.model_fn().to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        # Batched like fit() -- cuDNN's RNN kernel allocates workspace scaling with
        # batch_size*seq_len*hidden_size all at once, so pushing a large val set through in a
        # single shot (no_grad or not) can blow past GPU memory even though training on the same
        # total row count was fine split into mini-batches.
        X_t = torch.as_tensor(X, dtype=torch.float32)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i : i + self.batch_size].to(DEVICE)
                preds.append(self.model(batch).cpu().numpy())
        return np.concatenate(preds)


# Registry pattern (see TRAIN.md), mirrors FACTOR_MODELS in models.py.
SEQUENCE_MODELS = {
    "gru": lambda: _TorchSequenceWrapper(lambda: GRUModel()),
    "cnn": lambda: _TorchSequenceWrapper(lambda: CNNModel()),
    "patchtst": lambda: _TorchSequenceWrapper(lambda: PatchTSTModel()),
}

# save/load contract for the final trained models: SEQUENCE_MODELS' factories build a fresh
# _TorchSequenceWrapper each call (its model_fn is a lambda, which pickle/joblib can't
# serialize), so the saved artifact is just the underlying nn.Module's state_dict
# (torch.save(wrapper.model.state_dict(), path)) -- loading needs this name -> architecture
# class mapping to reconstruct the right empty module before calling load_state_dict on it.
ARCHITECTURES = {"gru": GRUModel, "cnn": CNNModel, "patchtst": PatchTSTModel}
