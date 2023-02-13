from qc.TorchCircuit import TorchCircuit
import torch.nn.functional as F
import torch.nn as nn
import torch

from config import *

class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(2304, NUM_LAYERS * NUM_QUBITS)
        self.qc = TorchCircuit.apply
        self.fc2 = nn.Linear(NUM_QC_OUTPUTS, CLASSES)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(1,-1)#(-1, 2304)
        x = self.fc1(x)
        x = np.pi * torch.tanh(x)
        x = self.qc(x[0])  # QUANTUM LAYER
        x = F.relu(x)
        x = self.fc2(x.float())
        #x = F.softmax(x, 1)
        return x

    def predict(self, x):
        pred = self.forward(x)
        ans = torch.argmax(pred[0]).item()
        return torch.tensor(ans)