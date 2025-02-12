import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """
    Builds a feedforward network with arbitrary hidden layers.

    Arguments:
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer
        hidden_layers: list of integers (one for each hidden layer), the sizes of the hidden layers
    """

    def __init__(self, input_size, output_size, hidden_layers) -> None:
        super().__init__()
        
        input_size = 784
        hidden_layers = [128, 64, 32]
        output_size = 10

        # model layers
        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.output = nn.Linear(hidden_layers[2], output_size)

        # dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass with dropout through the network, returns the output logits.
        """
        # flatten input tensor
        x = x.view(x.shape[0], -1)

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        
        # output
        x = self.output(x)

        return nn.functional.log_softmax(x, dim=1)
    
    def save_checkpoint(state_dict) -> None:
        """
        Save the model checkpoint.
        """
        model_checkpoint = {
            "input_size": 784,
            "output_size": 10,
            "hidden_layers": [128, 64, 32],
            "state_dict": state_dict,
        }
        torch.save(model_checkpoint, "model_checkpoint.pth")
    
if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
