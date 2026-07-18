import torch
import torch.nn as nn
import math


class MultiResHashEncoder(nn.Module):
    """
    Multi-resolution hash encoder implementation using tiny-cuda-nn.
    This is a wrapper to make it compatible with existing codebase.
    """
    def __init__(self, L, T, F, N_min, N_max, device, bounding_box):
        """
        Initialize hash encoder
        Args:
            L (int): Number of hash table levels
            T (int): Hash table size per level
            F (int): Feature dimension per hash table entry
            N_min (int): Resolution of the coarsest level
            N_max (int): Resolution of the finest level
            device (str): Computation device ('cpu' or 'cuda')
            bounding_box (list): Scene bounding box in format [min_bounds, max_bounds]
        """
        super().__init__()

        try:
            import tinycudann as tcnn
        except ImportError:
            raise ImportError("tiny-cuda-nn is not installed. Please install it first.")
            
        # tiny-cuda-nn requires bounding box in aabb format (min, max)
        self.register_buffer('min_bounds', torch.tensor(bounding_box[0], dtype=torch.float32))
        self.register_buffer('max_bounds', torch.tensor(bounding_box[1], dtype=torch.float32))

        # tiny-cuda-nn configuration
        # Note: log2_hashmap_size expects T to be a power of 2. If not, we need to adjust.
        # Here we assume T is already a power of 2, or close to one.
        log2_T = int(round(math.log2(T)))
        
        per_level_scale = math.exp((math.log(N_max) - math.log(N_min)) / (L - 1)) if L > 1 else 1.0

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": L,
            "n_features_per_level": F,
            "log2_hashmap_size": log2_T,
            "base_resolution": N_min,
            "per_level_scale": per_level_scale,
            "interpolation": "Linear"
        }
        
        # Input dimension is 3 (x, y, z)
        input_dim = 3
        # We only use the encoder
        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config=encoding_config,
        ).to(device)
        
        self.output_dim = self.encoder.n_output_dims

        # # print parameters
        # for name, param in self.encoder.named_parameters():
        #     print(f"Parameter name: {name}, Requires grad: {param.requires_grad}")

    def forward(self, x):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input coordinates with shape (batch_size, 3)
        Returns:
            torch.Tensor: Encoded features
        """
        # Normalize coordinates to [0, 1]
        normalized_x = (x - self.min_bounds) / (self.max_bounds - self.min_bounds)
        
        # tiny-cuda-nn expects input as (N, D) and float32
        return self.encoder(normalized_x.contiguous().float()) 