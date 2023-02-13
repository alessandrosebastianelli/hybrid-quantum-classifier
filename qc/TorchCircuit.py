from qc.QiskitCircuit import QiskitCircuit
from torch.autograd import Function
import torch

from config import *

class TorchCircuit(Function):
    @staticmethod
    def forward(ctx, i):
        if not hasattr(ctx, 'QiskitCirc'):
            ctx.QiskitCirc = QiskitCircuit(NUM_QUBITS, SIMULATOR, shots=NUM_SHOTS)

        exp_value = ctx.QiskitCirc.run(i)
        result = torch.tensor([exp_value])
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        forward_tensor, i = ctx.saved_tensors
        input_numbers = i
        gradients = torch.Tensor()

        for k in range(NUM_LAYERS * NUM_QUBITS):
            shift_right = input_numbers.detach().clone()
            shift_right[k] = shift_right[k] + SHIFT
            shift_left = input_numbers.detach().clone()
            shift_left[k] = shift_left[k] - SHIFT
            expectation_right = ctx.QiskitCirc.run(shift_right)
            expectation_left = ctx.QiskitCirc.run(shift_left)
            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            #gradient = gradient / torch.norm(gradient) # rescale gradient
            gradients = torch.cat((gradients, gradient.float()))

        result = torch.Tensor(gradients)

        return (result.float() * grad_output.float()).T