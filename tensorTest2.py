from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs')
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convl = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.convl(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


model = Model()
images = torch.randn(1, 1, 28, 28)
writer.add_graph(model, images)
writer.close()
