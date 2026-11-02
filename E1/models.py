"""E1: ba bộ phân loại ảnh 28x28 xám, 10 lớp.

Tất cả đều trả về LOGITS (chưa qua softmax). Softmax nằm trong nn.CrossEntropyLoss
khi huấn luyện, và được gọi tường minh (torch.softmax) khi cần xác suất lúc suy luận.
Nếu đặt thêm softmax trong forward rồi dùng CrossEntropyLoss thì softmax bị áp hai lần.
"""
import torch.nn as nn

IN_DIM = 28 * 28
NUM_CLASSES = 10


class SoftmaxClassifier(nn.Module):
    """Hồi quy softmax: flatten -> một ánh xạ tuyến tính W x + b -> (softmax trong loss)."""

    def __init__(self, in_dim: int = IN_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):               # x: (B, 1, 28, 28)
        return self.fc(self.flatten(x))  # (B, 10)


class MLPClassifier(nn.Module):
    """MLP trên vector flatten: 784 -> 512 -> 256 -> 10, ReLU + Dropout ở các tầng ẩn."""

    def __init__(self, in_dim: int = IN_DIM, hidden=(512, 256), num_classes: int = NUM_CLASSES,
                 dropout: float = 0.2):
        super().__init__()
        layers, prev = [nn.Flatten()], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def conv_bn_relu(c_in: int, c_out: int) -> nn.Sequential:
    # bias=False vì BatchNorm ngay sau đã có tham số dịch (beta), bias của conv là thừa.
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CNNClassifier(nn.Module):
    """CNN nhỏ kiểu VGG cho ảnh 28x28:
        [conv-bn-relu x2, maxpool]  1  -> 32 kênh, 28x28 -> 14x14
        [conv-bn-relu x2, maxpool]  32 -> 64 kênh, 14x14 -> 7x7
        flatten (64*7*7 = 3136) -> FC 128 -> dropout -> FC 10
    """

    def __init__(self, in_ch: int = 1, channels=(32, 64), fc_dim: int = 128,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        blocks, prev = [], in_ch
        for c in channels:
            blocks += [conv_bn_relu(prev, c), conv_bn_relu(c, c), nn.MaxPool2d(2)]
            prev = c
        self.features = nn.Sequential(*blocks)
        spatial = 28 // (2 ** len(channels))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev * spatial * spatial, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


MODELS = {
    "softmax": SoftmaxClassifier,
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
}


def build_model(name: str) -> nn.Module:
    return MODELS[name]()
