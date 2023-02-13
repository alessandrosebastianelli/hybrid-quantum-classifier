from qiskit.circuit import Parameter
from qiskit import execute
import qiskit

from config import *

class QiskitCircuit():
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.thetas = {k: Parameter('Theta' + str(k)) for k in range(NUM_LAYERS * self.n_qubits)}
        self.backend = backend

        all_qubits = [i for i in range(n_qubits)]
        self.circuit.h(all_qubits)
        self.circuit.barrier()

        for N in range(NUM_LAYERS):
            for k in range(0, self.n_qubits):
                self.circuit.ry(self.thetas[k+ N*self.n_qubits] , k)

            if NUM_LAYERS==1 or N < NUM_LAYERS-1:
                for k in range(0, self.n_qubits):
                    for k2 in range(0, self.n_qubits):
                        if k < k2:
                            self.circuit.cx(k2, k)

        self.circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def N_qubit_expectation_Z(self, counts, shots, nr_qubits):
        expects = np.zeros(NUM_QC_OUTPUTS)
        for k in range(NUM_QC_OUTPUTS):
            key = QC_OUTPUTS[k]
            perc = counts.get(key, 0) / shots
            expects[k] = perc
        return expects

    def run(self, i):
        params = i
        job_sim = execute(self.circuit,
                          self.backend,
                          shots=self.shots,
                          parameter_binds=[{self.thetas[k]: params[k].item() for k in range(NUM_LAYERS * NUM_QUBITS)}])
        #
        result_sim = job_sim.result()
        counts = result_sim.get_counts(self.circuit)
        return self.N_qubit_expectation_Z(counts, self.shots, NUM_QUBITS)