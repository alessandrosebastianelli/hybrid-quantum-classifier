from config import *
from utils.DatasetHandler import DatasetHandler
from models.HybridNet import HybridNet
from qc.QiskitCircuit import QiskitCircuit
from utils.utils import *

from sklearn.metrics import confusion_matrix, classification_report
import torch.optim as optim
import torch.nn as nn
import torch
import os

# Suppressing warning
import warnings
warnings.filterwarnings('ignore')

#=======================================================================================================================
print('\n[%] Checking for the availability of GPUs')
gpu = False
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu = True
else:
    device = torch.device("cpu")
print('\t [*] Running on device: {}'.format(device))

#=======================================================================================================================
print('\n[%] Initialize Quantum Hybrid Neural Network')
if gpu:
    network = HybridNet()
else:
    network = HybridNet()
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)#, momentum = MOMENTUM)

#=======================================================================================================================
print('\n[%] Printing Quantum Circuit')
circuit = QiskitCircuit(NUM_QUBITS, SIMULATOR, NUM_SHOTS)
print(circuit.circuit.draw(output='text', scale=1/NUM_LAYERS))

#=======================================================================================================================
print('\n[%] Printing Quantum Circuit Parameters')
print('\t [*] Number of Qubits:   {}'.format(NUM_QUBITS))
print('\t [*] Number of R Layers: {}'.format(NUM_LAYERS))
print('\t [*] Number of Outputs:  {}'.format(NUM_QC_OUTPUTS))
print('\t [*] Number of Shots:    {}'.format(NUM_SHOTS))

#=======================================================================================================================
print('\n[%] Loading Dataset')
handler_train = DatasetHandler(training_root)
handler_val   = DatasetHandler(validation_root)
classes = []
for i, c in enumerate(handler_train.classes):
    cl = c.split(os.path.sep)[-1]
    classes.append(cl)
classes.sort()

print('\t [*] Training classes: ',classes)
train_imgs, train_labels = handler_train.load_paths_labels(training_root, classes=classes)
val_imgs, val_labels = handler_val.load_paths_labels(validation_root, classes=classes)
train_loader = iter(handler_train.qcnn_data_loader(train_imgs, train_labels, batch_size=1, img_shape=(64,64,3)))
test_loader  = iter(handler_val.qcnn_data_loader(val_imgs, val_labels, batch_size=1, img_shape=(64,64,3)))

print('\t [*] Training size:  ', len(train_imgs))
print('\t [*] Validation size:', len(val_imgs))

#=======================================================================================================================
print('\n[%] Starting Training')
if LOAD_CHECKPOINT:
    print('\t[%] Loading Checkpoint')
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('\t\t [*] Checkpoint loaded, starting from:')
        print('\t\t\t - Epoch {}'.format(epoch))
        print('\t\t\t - Loss {}'.format(loss))
    except:
        print('\t\t [!] Checkpoint not found, training from scratch')
else:
  print('\t [!] Checkpoint not activated, training from scratch')

#=======================================================================================================================
if TRAINING:
    
    print('\t [*] Training ...')
    train_loss_list = []
    val_loss_list = []
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = []
        for batch_idx in range(len(train_labels)):
            data, target = next(train_loader)
            optimizer.zero_grad()
            # Forward pass
            output = network(data)
            # Calculating loss
            loss = loss_func(output, target)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()
            total_loss.append(loss.item())

            print('\r\t\t [*] [Epoch %d/%d] [Batch %d/%d] [Train Loss %f] ' % (epoch, EPOCHS, batch_idx, len(train_imgs) - 1, loss.item()),
                  end='\t\t')

        with torch.no_grad():
            val_loss = []
            targets = []
            predictions = []
            for batch_idx in range(len(val_imgs)):
                data, target = next(test_loader)
                output = network(data)
                loss = loss_func(output, target)
                val_loss.append(loss.item())
                targets.append(target.item())
                predictions.append(network.predict(data).item())

        train_loss_list.append(sum(total_loss) / len(total_loss))
        val_loss_list.append(sum(val_loss) / len(val_loss))

        print('[Val Loss %f] ' % (val_loss_list[-1]))

        if epoch % 3 == 1:
            cf = confusion_matrix(targets, predictions, normalize='true')
            cr = classification_report(targets, predictions, target_names=classes, digits=4)

            print('\t\t [*] Confusion Matrix:')
            print_CF(cf, classes)
            print('\t\t [*] Classification Report:')
            print(cr)

            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_list[-1],
            }, MODEL_SAVE_PATH)
