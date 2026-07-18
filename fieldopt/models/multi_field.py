import torch
import torch.nn as nn
from .hash_encoder import MultiResHashEncoder
from .mlp import MLP
import torch.nn.functional as F

class IdentityEncoder(nn.Module):
    """Identity encoder that simply passes through the input without encoding."""
    def __init__(self, input_dim=3):
        super().__init__()
        self.output_dim = input_dim
    
    def forward(self, x):
        return x

class ZOnlyTimeField(nn.Module):
    """
    Neural AM time field that depends only on the z coordinate.

    The module accepts full xyz points for compatibility with Field 1 call sites,
    but the learned value is a function of normalized z only. Like the original
    Field 1 module, it returns a normalized value in [0, 1]; MultiFieldModel is
    responsible for scaling it by max_time.
    """
    def __init__(self, bounding_box, max_time, hidden_dim=32, n_hidden_layers=2,
                 activation='softplus', z_group_decimals=None):
        super().__init__()
        self.max_time = float(max_time)
        bbox = torch.as_tensor(bounding_box, dtype=torch.float32)
        self.register_buffer('z_min', bbox[0, 2].reshape(1))
        self.register_buffer('z_max', bbox[1, 2].reshape(1))
        self.z_group_decimals = z_group_decimals

        activation_layers = {
            'relu': nn.ReLU,
            'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01),
            'softplus': nn.Softplus,
            'tanh': nn.Tanh,
        }
        if activation not in activation_layers:
            raise ValueError(f"Unsupported activation for ZOnlyTimeField: {activation}")

        layers = []
        in_dim = 1
        for _ in range(int(n_hidden_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(activation_layers[activation]())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = x[..., 2:3]
        z_span = torch.clamp(self.z_max - self.z_min, min=1e-8)
        z_norm = torch.clamp((z - self.z_min) / z_span, min=0.0, max=1.0)
        return self.net(z_norm)

class FrequencyEncoder(nn.Module):
    """Positional encoding for temporal information using sine and cosine functions."""
    def __init__(self, num_frequencies=10, include_input=False):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Output: 2 * num_frequencies (sin and cos for each frequency)
        self.output_dim = 2 * num_frequencies
        if include_input:
            self.output_dim += 1  # Include original input
        
        # Frequency bands: [2^0, 2^1, ..., 2^(num_frequencies-1)]
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, t):
        """
        Args:
            t: (batch_size, 1) or (batch_size,) time values
        Returns:
            (batch_size, output_dim) encoded features
        """
        # Ensure t is 2D
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # t: (batch_size, 1)
        # freq_bands: (num_frequencies,)
        # t_freq: (batch_size, num_frequencies)
        t_freq = t * self.freq_bands.unsqueeze(0)
        
        # Compute sin and cos
        sin_features = torch.sin(t_freq)
        cos_features = torch.cos(t_freq)
        
        # Concatenate: (batch_size, 2*num_frequencies)
        encoded = torch.cat([sin_features, cos_features], dim=-1)
        
        # Optionally include original input
        if self.include_input:
            encoded = torch.cat([t, encoded], dim=-1)
        
        return encoded

class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation using temporal features.
    
    Uses identity initialization to prevent training collapse:
    - Gamma (scale) initialized to 1: preserves spatial features initially
    - Beta (shift) initialized to 0: no initial offset
    This ensures the network starts with identity transformation and learns modulation gradually.
    """
    def __init__(self, temporal_dim, spatial_dim, leaky_relu_slope=0.01):
        super().__init__()
        # MLP to generate FiLM parameters from temporal features
        # Generates gamma (scale) and beta (shift) parameters
        self.mlp = nn.Sequential(
            nn.Linear(temporal_dim, spatial_dim * 2),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.Linear(spatial_dim * 2, spatial_dim * 2)
        )
        
        # Initialize the last layer for identity mapping
        self._init_film_params()
    
    def _init_film_params(self):
        """Initialize FiLM parameters for identity transformation.
        
        Gamma (scale) part: weights=0, bias=1 → output=1 (identity)
        Beta (shift) part: weights=0, bias=0 → output=0 (no shift)
        
        This prevents training collapse where gamma≈0 would erase spatial features
        and block gradient flow in early training stages.
        """
        last_layer = self.mlp[-1]  # Get the last linear layer
        
        with torch.no_grad():
            # Split the output dimension: first half for gamma, second half for beta
            output_dim = last_layer.out_features  # spatial_dim * 2
            half_dim = output_dim // 2
            
            # Initialize gamma part (first half): weights=0, bias=1
            # This ensures gamma ≈ 1 initially, preserving spatial features
            last_layer.weight[:half_dim, :] = 0.0
            last_layer.bias[:half_dim] = 1.0
            
            # Initialize beta part (second half): weights=0, bias=0
            # This ensures beta ≈ 0 initially, no offset
            last_layer.weight[half_dim:, :] = 0.0
            last_layer.bias[half_dim:] = 0.0
        
    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features: (batch_size, spatial_dim)
            temporal_features: (batch_size, temporal_dim)
        Returns:
            modulated_features: (batch_size, spatial_dim)
        """
        # Generate FiLM parameters
        film_params = self.mlp(temporal_features)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # Apply affine transformation: gamma * features + beta
        return gamma * spatial_features + beta

class Field(nn.Module):
    def __init__(self, encoder, mlp, scale_factor=1.0):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.scale_factor = scale_factor

    def forward(self, x):
        x_encoded = self.encoder(x)
        return self.mlp(x_encoded.float()) * self.scale_factor

# class twoNormVectorField(nn.Module):
#     def __init__(self, encoder, mlp):
#         super().__init__()
#         self.encoder = encoder
#         self.mlp = mlp
    
#     def forward(self, x):
#         x_encoded = self.encoder(x)
#         raw_vec = self.mlp(x_encoded.float())

#         vec1 = raw_vec[...,:3]
#         unit_vec1 = F.normalize(vec1, p=2, dim=-1, eps=1e-12) # normalized vector 1

#         vec2 = raw_vec[...,3:6]
#         modified_vec2 = torch.cat([
#             vec2[...,:2],
#             torch.abs(vec2[...,2:3])
#         ], dim=-1) 
#         unit_vec2 = F.normalize(modified_vec2, p=2, dim=-1, eps=1e-12)

#         twoVecs = torch.cat([unit_vec1, unit_vec2], dim=-1)
#         return twoVecs

class newtwoNormVectorField(nn.Module):
    def __init__(self, encoder, mlp, alpha_deg = 89.0):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.register_buffer('alpha_rad', torch.tensor(torch.pi / 180.0 * alpha_deg))
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        raw_vec = self.mlp(x_encoded.float())

        vec_axis_raw = raw_vec[..., :3]
        u = torch.cat([vec_axis_raw[..., :2], torch.abs(vec_axis_raw[..., 2:3]) + 1e-8], dim=-1)
        u = F.normalize(u, p=2, dim=-1, eps=1e-12)
        ux, uy, uz = u[..., 0:1], u[..., 1:2], u[..., 2:3]

        p = raw_vec[..., 3:5] 
        r = torch.norm(p, p=2, dim=-1, keepdim=True)
        theta = torch.tanh(r) * self.alpha_rad

        eps = 1e-12
        st = torch.sin(theta)
        ct = torch.cos(theta)

        cx = (p[..., 0:1] / (r + eps)) * st
        cy = (p[..., 1:2] / (r + eps)) * st

        inv_1puz = 1.0 / (1.0 + uz)

        vx = ct * ux + cx * (1.0 - ux * ux * inv_1puz) - cy * (ux * uy * inv_1puz)
        vy = ct * uy - cx * (ux * uy * inv_1puz) + cy * (1.0 - uy * uy * inv_1puz)
        vz = ct * uz - cx * ux - cy * uy
        
        v_raw = torch.cat([vx, vy, vz], dim=-1)
        v = F.normalize(v_raw, p=2, dim=-1, eps=1e-12)

        twoVecs = torch.cat([v, u], dim=-1)
        return twoVecs

class SpatioTemporalDensityField(nn.Module):
    """Spatio-temporal density field with FiLM modulation.
    
    Combines spatial hash encoding with temporal frequency encoding,
    using FiLM to modulate spatial features based on time.
    """
    def __init__(self, L, T, F, N_min, N_max, n_neurons, n_hidden_layers,
                 bounding_box, device, num_time_frequencies=10, dropout_rate=0.0):
        """
        Args:
            L, T, F, N_min, N_max: Hash encoder parameters
            n_neurons, n_hidden_layers: MLP parameters
            bounding_box: Scene bounding box
            device: Computation device
            num_time_frequencies: Number of frequency bands for temporal encoding
            dropout_rate: Dropout rate for MLP
        """
        super().__init__()
        
        # Spatial encoder: multi-resolution hash encoding
        self.spatial_encoder = MultiResHashEncoder(L, T, F, N_min, N_max, device, bounding_box)
        spatial_dim = self.spatial_encoder.output_dim
        
        # Temporal encoder: frequency encoding
        self.temporal_encoder = FrequencyEncoder(num_frequencies=num_time_frequencies)
        temporal_dim = self.temporal_encoder.output_dim
        
        # FiLM modulation: modulate spatial features with temporal information
        self.film_modulation = FiLMModulation(temporal_dim, spatial_dim)
        
        # MLP for final density prediction
        self.mlp = MLP(
            input_dim=spatial_dim,
            output_dim=1,
            n_neurons=n_neurons,
            n_hidden_layers=n_hidden_layers,
            dropout_rate=dropout_rate,
            output_activation='sigmoid'  # Binary density output in [0, 1]
        )
    
    def forward(self, x, t):
        """
        Forward pass for spatio-temporal density prediction.
        
        Args:
            x: (batch_size, 3) spatial coordinates (xyz)
            t: (batch_size, 1) or (batch_size,) time values
        Returns:
            density: (batch_size, 1) density values in [0, 1]
        """
        # Ensure t is 2D
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Encode spatial and temporal information
        spatial_features = self.spatial_encoder(x)
        temporal_features = self.temporal_encoder(t)
        
        # FiLM modulation: time modulates space
        modulated_features = self.film_modulation(spatial_features, temporal_features)
        
        # Predict density
        density = self.mlp(modulated_features)
        return density

class MultiFieldModel(nn.Module):
    """
    Multi field model.
    Contains multiple independent networks: two scalar fields, a vector field, and
    a binary mask field. Each field has its own independent hash encoder parameters,
    learning different feature details.
    """
    def __init__(self, 
                 # Parameters for the first field
                 L1, T1, F1, N_min1, N_max1, n_neurons1, n_hidden_layers1, max_time,
                 # Parameters for the second field  
                 L2, T2, F2, N_min2, N_max2, n_neurons2, n_hidden_layers2,

                 # Parameters for the third field
                 L3, T3, F3, N_min3, N_max3, n_neurons3, n_hidden_layers3,

                 # Parameters for the mask field 1
                 LM1, TM1, FM1, N_minM1, N_maxM1, n_neuronsM1, n_hidden_layersM1,
                 # Parameters for the mask field 2
                 LM2, TM2, FM2, N_minM2, N_maxM2, n_neuronsM2, n_hidden_layersM2,

                 # Parameters for the spatio-temporal density field
                 LDT, TDT, FDT, N_minDT, N_maxDT, n_neuronsDT, n_hidden_layersDT,

                 bounding_box, # Added bounding_box parameter
                 device,
                 dropout_rate_field1=0.5,  # Dropout rate for Field 1
                 dropout_rate_field2=0.5,  # Dropout rate for Field 2
                 dropout_rate_field3=0.5,  # Dropout rate for Field 3
                 dropout_rate_fieldM1=0.0,
                 dropout_rate_fieldM2=0.0,
                 num_time_frequencies_DT=10,  # Temporal encoding frequencies
                 dropout_rate_fieldDT=0.0):
        """
        Initialize the multi field model.
        
        Args:
            # Parameters for the first field
            L1, T1, F1, N_min1, N_max1: Hash encoding parameters for the first field
            n_neurons1, n_hidden_layers1: MLP parameters for the first field

            bounding_box (list): Scene bounding box in format [min_bounds, max_bounds]
            device: Computation device

            dropout_rate_field1 (float): Dropout rate for Field 1 MLP
        """
        super().__init__()
        self.max_time = max_time

        self.bounding_box = torch.tensor(bounding_box, dtype=torch.float32, device=device)
        self.device = device
        
        # First scalar field: plain neural network with softplus activation for high-order continuity
        # encoder_field1 = IdentityEncoder(input_dim=3)  # Direct coordinate input
        # mlp_field1 = MLP(input_dim=3,  # xyz coordinates as input
        #                      output_dim=1,  # Output scalar value
        #                      n_neurons=256, 
        #                      n_hidden_layers=6,
        #                      activation='softplus',  # Use softplus for high-order continuity
        #                      dropout_rate=dropout_rate_field1,
        #                      output_activation='sigmoid')
        # self.field1 = Field(encoder_field1, mlp_field1)

                # First scalar field: independent encoder and MLP
        encoder_field1 = MultiResHashEncoder(L1, T1, F1, N_min1, N_max1, device, bounding_box)
        mlp_field1 = MLP(input_dim=encoder_field1.output_dim, 
                             output_dim=1,  # Output scalar value
                             n_neurons=n_neurons1, 
                             n_hidden_layers=n_hidden_layers1,
                             dropout_rate=dropout_rate_field1,
                            #  output_activation='softsign01')
                             output_activation='sigmoid')
        self.field1 = Field(encoder_field1, mlp_field1)
        
        
        # Second scalar field: completely independent encoder and MLP
        # This encoder has its own embeddings parameters, completely separate from the first field
        encoder_field2 = MultiResHashEncoder(L2, T2, F2, N_min2, N_max2, device, bounding_box)
        mlp_field2 = MLP(input_dim=encoder_field2.output_dim, 
                             output_dim=1,  # Output scalar value
                             n_neurons=n_neurons2, 
                             n_hidden_layers=n_hidden_layers2,
                             dropout_rate=dropout_rate_field2,  # Use Field 2's dropout
                            #  output_activation='softsign01')
                             output_activation='sigmoid')
        self.field2 = Field(encoder_field2, mlp_field2)

        # A Vector field: independent encoder and MLP
        encoder_field3 = MultiResHashEncoder(L3, T3, F3, N_min3, N_max3, device, bounding_box)
        mlp_field3 = MLP(input_dim=encoder_field3.output_dim, 
                             output_dim=5,  # Output vector value
                             n_neurons=n_neurons3, 
                             n_hidden_layers=n_hidden_layers3,
                             dropout_rate=dropout_rate_field3,
                             output_activation='forF3')
                            #  output_activation=None)
        self.field3 = newtwoNormVectorField(encoder_field3, mlp_field3)

        # Mask scalar field M1 (output logits)
        encoder_fieldM1 = MultiResHashEncoder(LM1, TM1, FM1, N_minM1, N_maxM1, device, bounding_box)
        mlp_fieldM1 = MLP(input_dim=encoder_fieldM1.output_dim,
                           output_dim=1,
                           n_neurons=n_neuronsM1,
                           n_hidden_layers=n_hidden_layersM1,
                           dropout_rate=dropout_rate_fieldM1,
                           output_activation=None)
        self.fieldM1 = Field(encoder_fieldM1, mlp_fieldM1)

        # Mask scalar field M2 (output logits)
        encoder_fieldM2 = MultiResHashEncoder(LM2, TM2, FM2, N_minM2, N_maxM2, device, bounding_box)
        mlp_fieldM2 = MLP(input_dim=encoder_fieldM2.output_dim,
                           output_dim=1,
                           n_neurons=n_neuronsM2,
                           n_hidden_layers=n_hidden_layersM2,
                           dropout_rate=dropout_rate_fieldM2,
                           output_activation=None)
        self.fieldM2 = Field(encoder_fieldM2, mlp_fieldM2)

        # Spatio-temporal density field
        # self.fieldDT = SpatioTemporalDensityField(
        #     L=LDT, T=TDT, F=FDT, N_min=N_minDT, N_max=N_maxDT,
        #     n_neurons=n_neuronsDT, n_hidden_layers=n_hidden_layersDT,
        #     bounding_box=bounding_box, device=device,
        #     num_time_frequencies=num_time_frequencies_DT,
        #     dropout_rate=dropout_rate_fieldDT
        # )

        # Print parameter information to confirm independence
        print(f"Field 1 hash encoder parameter count: {sum(p.numel() for p in self.field1.encoder.parameters())}")
        print(f"Field 1 MLP parameter count: {sum(p.numel() for p in self.field1.mlp.parameters())}")
        print(f"Field 1 dropout rate: {dropout_rate_field1}")
        print(f"Field 2 hash encoder parameter count: {sum(p.numel() for p in self.field2.encoder.parameters())}")
        print(f"Field 2 MLP parameter count: {sum(p.numel() for p in self.field2.mlp.parameters())}")
        print(f"Field 2 dropout rate: {dropout_rate_field2}")
        print(f"Field 3 hash encoder parameter count: {sum(p.numel() for p in self.field3.encoder.parameters())}")
        print(f"Field 3 MLP parameter count: {sum(p.numel() for p in self.field3.mlp.parameters())}")
        print(f"Field 3 dropout rate: {dropout_rate_field3}")
        print(f"Field M1 hash encoder parameter count: {sum(p.numel() for p in self.fieldM1.encoder.parameters())}")
        print(f"Field M1 MLP parameter count: {sum(p.numel() for p in self.fieldM1.mlp.parameters())}")
        print(f"Field M1 dropout rate: {dropout_rate_fieldM1}")
        print(f"Field M2 hash encoder parameter count: {sum(p.numel() for p in self.fieldM2.encoder.parameters())}")
        print(f"Field M2 MLP parameter count: {sum(p.numel() for p in self.fieldM2.mlp.parameters())}")
        print(f"Field M2 dropout rate: {dropout_rate_fieldM2}")
        # print(f"Field DT spatial encoder parameter count: {sum(p.numel() for p in self.fieldDT.spatial_encoder.parameters())}")
        # print(f"Field DT temporal encoder parameter count: {sum(p.numel() for p in self.fieldDT.temporal_encoder.parameters())}")
        # print(f"Field DT FiLM modulation parameter count: {sum(p.numel() for p in self.fieldDT.film_modulation.parameters())}")
        # print(f"Field DT MLP parameter count: {sum(p.numel() for p in self.fieldDT.mlp.parameters())}")
        # print(f"Field DT dropout rate: {dropout_rate_fieldDT}")
        # print(f"Field DT num time frequencies: {num_time_frequencies_DT}")

        self.to(device)

    def forward(self, x, field_type='both'):
        if field_type == 'field1':
            return self.field1(x) * self.max_time

        elif field_type == 'field2':
            return self.field2(x)

        elif field_type == 'field3':
            return self.field3(x)

        elif field_type == 'fieldM1':
            return self.fieldM1(x)

        elif field_type == 'fieldM2':
            return self.fieldM2(x)

        elif field_type == 'fieldDT':
            # Expect x to contain both spatial (xyz) and temporal (t) info
            # x should be (batch_size, 4) where last dim is time
            xyz = x[..., :3]
            t = x[..., 3:]
            return self.fieldDT(xyz, t)

        elif field_type == 'both':
            field1_output = self.field1(x) * self.max_time
            field2_output = self.field2(x)
            return field1_output, field2_output

        elif field_type == 'both_unscaled':
            field1_output = self.field1(x)
            field2_output = self.field2(x)
            return field1_output, field2_output

        elif field_type == 'masks':
            fieldM1_output = self.fieldM1(x)
            fieldM2_output = self.fieldM2(x)
            return fieldM1_output, fieldM2_output
        
        elif field_type == "timesAndMasks":
            field1_output = self.field1(x) * self.max_time
            field2_output = self.field2(x)
            fieldM1_output = self.fieldM1(x)
            fieldM2_output = self.fieldM2(x)
            return field1_output, field2_output, fieldM1_output, fieldM2_output

        elif field_type == 'all':
            field1_output = self.field1(x) * self.max_time
            field2_output = self.field2(x)
            field3_output = self.field3(x)
            fieldM1_output = self.fieldM1(x)
            fieldM2_output = self.fieldM2(x)
            return field1_output, field2_output, field3_output, fieldM1_output, fieldM2_output
        
        else:
            raise ValueError("field_type must be 'field1', 'field2', 'field3', 'fieldM1', 'fieldM2', 'fieldDT', 'both', or 'all'")
    
    # def get_field_gradients(self, x):
    #     x_local = x.detach().requires_grad_(True)

    #     time1_output, alphaT2_output = self.forward(x_local, field_type='both_unscaled')
    #     time2 = time1_output + alphaT2_output * (1.0 - time1_output)

    #     gradTime2 = torch.autograd.grad(time2.sum(), x_local, create_graph=True)[0]
    #     modified_gradTime2 = -1 * gradTime2
    #     unit_vec = F.normalize(modified_gradTime2, p=2, dim=-1, eps=1e-12)

    #     return time1_output * self.max_time, alphaT2_output, unit_vec

    # def get_field_gradients(self, x):
    #     """
    #     Compute gradients of both fields at given positions.
    #     Useful for analyzing field variation rates.
    #     """
    #     x_local = x.detach().requires_grad_(True)
        
    #     field1_vals, field2_vals = self.forward(x_local, field_type='both')

    #     gradTime2 = torch.autograd.grad((field1_vals+field2_vals).sum(), x_local, create_graph=True)[0]
    #     modified_gradTime2 = -1 * gradTime2
    #     unit_vec = F.normalize(modified_gradTime2, p=2, dim=-1, eps=1e-12)

    #     return field1_vals, field2_vals, unit_vec

    def set_active_field(self, field_type):
        """
        Sets which field is active for training, freezing the other.
        """
        if field_type == 'field1':
            for param in self.field1.parameters():
                param.requires_grad = True
            for param in self.field2.parameters():
                param.requires_grad = False
            for param in self.field3.parameters():
                param.requires_grad = False
            for param in self.fieldM1.parameters():
                param.requires_grad = False
            for param in self.fieldM2.parameters():
                param.requires_grad = False
            # for param in self.fieldDT.parameters():
            #     param.requires_grad = False

        elif field_type == 'field2':
            for param in self.field1.parameters():
                param.requires_grad = False
            for param in self.field2.parameters():
                param.requires_grad = True
            for param in self.field3.parameters():
                param.requires_grad = False
            for param in self.fieldM1.parameters():
                param.requires_grad = False
            for param in self.fieldM2.parameters():
                param.requires_grad = False
            # for param in self.fieldDT.parameters():
            #     param.requires_grad = False

        elif field_type == 'field3':
            for param in self.field1.parameters():
                param.requires_grad = False
            for param in self.field2.parameters():
                param.requires_grad = False
            for param in self.field3.parameters():
                param.requires_grad = True
            for param in self.fieldM1.parameters():
                param.requires_grad = False
            for param in self.fieldM2.parameters():
                param.requires_grad = False
            # for param in self.fieldDT.parameters():
            #     param.requires_grad = False

        elif field_type == 'fieldM1':
            for param in self.field1.parameters():
                param.requires_grad = False
            for param in self.field2.parameters():
                param.requires_grad = False
            for param in self.field3.parameters():
                param.requires_grad = False
            for param in self.fieldM1.parameters():
                param.requires_grad = True
            for param in self.fieldM2.parameters():
                param.requires_grad = False
            # for param in self.fieldDT.parameters():
            #     param.requires_grad = False
        
        elif field_type == 'fieldM2':
            for param in self.field1.parameters():
                param.requires_grad = False
            for param in self.field2.parameters():
                param.requires_grad = False
            for param in self.field3.parameters():
                param.requires_grad = False
            for param in self.fieldM1.parameters():
                param.requires_grad = False
            for param in self.fieldM2.parameters():
                param.requires_grad = True
            # for param in self.fieldDT.parameters():
            #     param.requires_grad = False

        # elif field_type == 'fieldDT':
        #     for param in self.field1.parameters():
        #         param.requires_grad = False
        #     for param in self.field2.parameters():
        #         param.requires_grad = False
        #     for param in self.field3.parameters():
        #         param.requires_grad = False
        #     for param in self.fieldM1.parameters():
        #         param.requires_grad = False
        #     for param in self.fieldM2.parameters():
        #         param.requires_grad = False
        #     for param in self.fieldDT.parameters():
        #         param.requires_grad = True

        elif field_type == 'all':
            for param in self.parameters():
                param.requires_grad = True
        
        else:
            raise ValueError("field_type must be 'field1', 'field2', 'field3', 'fieldM1', 'fieldM2', 'fieldDT', or 'all'") 

    
