"""Reference-only Dense Autoencoder and PCA reconstruction models."""

import os
import random
from dataclasses import asdict, dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data_loading import SENSORS


class DenseAutoencoder(nn.Module):
    """14 -> 32 -> 16 -> 8 -> 16 -> 32 -> 14, linear output for scaled data."""

    def __init__(self, input_dim: int = 14):
        super().__init__()
        self.input_dim = input_dim
        widths = [input_dim, 32, 16, 8, 16, 32, input_dim]
        layers = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = .001
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    seed: int = 42
    healthy_cycles: int = 20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features or len(set(features)) != len(features) or not set(features) <= set(SENSORS):
        raise ValueError("Features must be unique sensor columns, without identifiers/settings/RUL")
    values = frame[features].to_numpy(dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Model inputs must be nonempty and finite")
    return values


def reference_records(frame: pd.DataFrame, healthy_cycles: int = 20) -> pd.DataFrame:
    if healthy_cycles < 1 or not isinstance(healthy_cycles, int):
        raise ValueError("Reference window must be a positive integer")
    reference = frame.loc[frame.cycle.between(1, healthy_cycles)].copy()
    if reference.empty or set(reference.unit_id) != set(frame.unit_id):
        raise ValueError("Every engine must have early-reference records")
    return reference


def reference_sets(development: pd.DataFrame, validation: pd.DataFrame,
                   healthy_cycles: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(development.unit_id) & set(validation.unit_id):
        raise ValueError("Development and validation engine IDs overlap")
    return reference_records(development, healthy_cycles), reference_records(validation, healthy_cycles)


def reconstruct_autoencoder(model: DenseAutoencoder, values: np.ndarray,
                            device: str = "cpu", batch_size: int = 1024) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != model.input_dim or not np.isfinite(values).all():
        raise ValueError("Invalid Autoencoder input matrix")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model.to(device).eval()
    with torch.inference_mode():
        chunks = [model(torch.from_numpy(values[start:start + batch_size]).to(device)).cpu().numpy()
                  for start in range(0, len(values), batch_size)]
    result = np.concatenate(chunks) if chunks else np.empty_like(values)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite Autoencoder reconstruction")
    return result


def train_autoencoder(development: pd.DataFrame, validation: pd.DataFrame,
                      features: list[str], config: TrainingConfig = TrainingConfig(),
                      device: str = "cpu") -> tuple[DenseAutoencoder, pd.DataFrame, dict]:
    """Only development reference rows receive optimizer updates.

    Epoch losses are evaluated over entire reference sets at the end of each
    epoch. Validation selects the best checkpoint but never receives gradients.
    """
    if config.max_epochs < 1 or config.patience < 1 or config.batch_size < 1 or config.learning_rate <= 0:
        raise ValueError("Training parameters must be positive")
    dev_ref, val_ref = reference_sets(development, validation, config.healthy_cycles)
    train_values = feature_matrix(dev_ref, features).astype(np.float32)
    val_values = feature_matrix(val_ref, features).astype(np.float32)
    seed_everything(config.seed)
    model = DenseAutoencoder(len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_values)),
                        batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
        train_loss = float(np.mean((train_values.astype(np.float64) -
                                   reconstruct_autoencoder(model, train_values, device)) ** 2))
        val_loss = float(np.mean((val_values.astype(np.float64) -
                                 reconstruct_autoencoder(model, val_values, device)) ** 2))
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError("Non-finite epoch losses")
        history.append({"epoch": epoch, "training_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    metadata = {
        "config": asdict(config), "input_features": features,
        "architecture": [len(features), 32, 16, 8, 16, 32, len(features)],
        "activation": "ReLU on hidden layers; linear output", "loss": "mean squared error",
        "optimizer": "Adam", "device": device, "epochs_run": len(history),
        "best_epoch": best_epoch, "best_validation_mse": best_loss,
        "minimum_training_mse": min(row["training_mse"] for row in history),
        "training_mse_at_best_epoch": history[best_epoch - 1]["training_mse"],
        "training_reference_records": len(dev_ref), "validation_reference_records": len(val_ref),
        "training_unit_ids": sorted(int(x) for x in dev_ref.unit_id.unique()),
        "validation_unit_ids": sorted(int(x) for x in val_ref.unit_id.unique()),
        "early_stopped": len(history) < config.max_epochs,
        "loss_history_definition": "End-of-epoch full reference-set MSE, not online batch loss",
    }
    return model, pd.DataFrame(history), metadata


def fit_pca(development: pd.DataFrame, features: list[str], healthy_cycles: int = 20,
            n_components: int = 8) -> PCA:
    """Fit a deterministic full-SVD PCA on development reference rows only."""
    reference = reference_records(development, healthy_cycles)
    values = feature_matrix(reference, features)
    if not 1 <= n_components <= min(values.shape):
        raise ValueError("Invalid PCA component count")
    return PCA(n_components=n_components, svd_solver="full").fit(values)


def reconstruct_pca(pca: PCA, values: np.ndarray) -> np.ndarray:
    return pca.inverse_transform(pca.transform(values))
