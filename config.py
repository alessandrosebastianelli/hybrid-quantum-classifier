import numpy as np
import itertools
import qiskit
import glob
import os

#----------------------- Quantum Circuit Settings -----------------------
NUM_QUBITS     = 4
NUM_SHOTS      = 3000
NUM_LAYERS     = 1
SHIFT          = np.pi/4

def create_QC_OUTPUTS():
    measurements = list(itertools.product([0, 1], repeat=NUM_QUBITS))
    return [''.join([str(bit) for bit in measurement]) for measurement in measurements]

QC_OUTPUTS = create_QC_OUTPUTS()
NUM_QC_OUTPUTS = len(QC_OUTPUTS)

SIMULATOR = qiskit.Aer.get_backend('qasm_simulator')

#----------------------- Dataset Settings -----------------------
training_root   = os.path.join( 'datasets', 'VolcanicEruptionDetection', 'training')
validation_root = os.path.join( 'datasets', 'VolcanicEruptionDetection', 'validation')

indices, classes = [],[]
for i, c in enumerate(glob.glob(os.path.join(training_root, '*'))):
    indices.append(i)
    classes.append(c.split(os.sep)[-1])

CLASS_DICT = {classes[i] : indices[i] for i in range(len(indices))}

#----------------------- Training Settings -----------------------
TRAINING        = True
LOAD_CHECKPOINT = False
EPOCHS          = 100
LEARNING_RATE   = 0.0002
#MOMENTUM        = 0.5
BATCH_SIZE      = 16
CLASSES         = len(CLASS_DICT)

MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'model-Q{}-L{}.pt'.format(NUM_QUBITS, NUM_LAYERS))
