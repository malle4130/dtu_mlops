import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
from torch import nn, optim

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # training loop
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 5
    steps = 0

    model.train()

    train_losses = []
    for _ in range(epochs):
        running_loss = 0
        for images, labels in train_dataloader:
            # Flatten images into a 784 long vector
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            running_loss += loss.item()

    print("Our model: \n\n", model, "\n")
    print("The state dict keys: \n\n", model.state_dict().keys())

    model_checkpoint = {
            "input_size": 784,
            "output_size": 10,
            "hidden_layers": [128, 64, 32],
            "state_dict": model.state_dict(),
        }
    torch.save(model_checkpoint, "model_checkpoint.pth")


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = load_checkpoint(model_checkpoint)
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total * 100}%")

def load_checkpoint(filepath):
    """Load checkpoint and rebuild the model."""
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel(checkpoint["input_size"], checkpoint["output_size"], checkpoint["hidden_layers"])
    model.load_state_dict(checkpoint["state_dict"])

    return model

if __name__ == "__main__":
    app()
