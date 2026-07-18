import torch
import torch.nn as nn

class Softsign01(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return nn.functional.softsign(x) * 0.5 + 0.5

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons, n_hidden_layers, activation='relu', output_activation=None, dropout_rate=0.0):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        layers = []
        current_dim = input_dim
        
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(current_dim, n_neurons))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'softplus':
                layers.append(nn.Softplus())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            if dropout_rate > 0:
                # Store the index of the dropout layer for dynamic adjustment
                dropout_layer = nn.Dropout(p=dropout_rate)
                layers.append(dropout_layer)

            current_dim = n_neurons
        
        if output_activation is not None:
            if output_activation == 'sigmoid':
                layers.append(nn.Linear(current_dim, output_dim))
                layers.append(nn.Sigmoid())
            elif output_activation == 'relu':
                layers.append(nn.Linear(current_dim, output_dim))
                layers.append(nn.ReLU())
            elif output_activation == 'leaky_relu':
                layers.append(nn.Linear(current_dim, output_dim))
                layers.append(nn.LeakyReLU())
            elif output_activation == 'softplus':
                layers.append(nn.Linear(current_dim, output_dim))
                layers.append(nn.Softplus())
            elif output_activation == "softsign01":
                final_linear = nn.Linear(current_dim, output_dim)
                nn.init.xavier_uniform_(final_linear.weight, gain=0.1)
                nn.init.zeros_(final_linear.bias)
                layers.append(final_linear)
                layers.append(Softsign01()) # 2nd gradient not continuous
            elif output_activation == "forF3":
                final_linear = nn.Linear(current_dim, output_dim)
                nn.init.xavier_uniform_(final_linear.weight, gain=0.01)
                nn.init.zeros_(final_linear.bias)
                with torch.no_grad():
                    final_linear.bias[2].fill_(1.0) 
                layers.append(final_linear)
            else:
                raise ValueError(f"Unsupported output activation: {output_activation}")
        else:
            layers.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def set_dropout_rate(self, dropout_rate):
        """Dynamically set the dropout rate"""
        self.dropout_rate = dropout_rate
        for module in self.mlp.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
    
    def get_dropout_rate(self):
        """Get the current dropout rate"""
        return self.dropout_rate

    def forward(self, x):
        return self.mlp(x) 