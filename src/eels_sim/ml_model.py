"""PyTorch model kept separate so the base simulator has no ML dependency."""


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required only for ML training; install requirements-ml.txt "
            "in an isolated environment on the designated workstation") from exc
    return torch, nn


def build_alexnet_regressor(input_channels, output_count):
    """Build the AlexNet-style 101 px regressor shown in the reference slide."""
    _, nn = require_torch()

    class AlexNetRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6)),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.25), nn.Linear(256 * 6 * 6, 2048),
                nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(2048, 512),
                nn.ReLU(inplace=True), nn.Linear(512, output_count),
            )

        def forward(self, image):
            return self.regressor(self.features(image))

    return AlexNetRegressor()


def select_device(torch, requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
