import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = BaselineCNN(num_classes=6, dropout_rate=0.5)

    dummy = torch.zeros(4, 3, 224, 224)
    output = model(dummy)

    print(model)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")