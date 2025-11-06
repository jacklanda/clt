__version__ = "1.1.3"

from .config import SaeConfig, SparseCoderConfig, TrainConfig, TranscoderConfig
from .runner import CrossLayerRunner
from .sparse_coder import Sae, SparseCoder
from .trainer import SaeTrainer, Trainer

# Optional evaluation imports (requires nnsight)
try:
    from .evaluation import loss_recovered, compute_frac_recovered
    __all__ = [
        "Sae",
        "SaeConfig",
        "SaeTrainer",
        "SparseCoder",
        "SparseCoderConfig",
        "CrossLayerRunner",
        "Trainer",
        "TrainConfig",
        "TranscoderConfig",
        "loss_recovered",
        "compute_frac_recovered",
    ]
except ImportError:
    # nnsight not available, skip evaluation imports
    __all__ = [
        "Sae",
        "SaeConfig",
        "SaeTrainer",
        "SparseCoder",
        "SparseCoderConfig",
        "CrossLayerRunner",
        "Trainer",
        "TrainConfig",
        "TranscoderConfig",
    ]
