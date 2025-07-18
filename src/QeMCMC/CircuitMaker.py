import numpy as np
from .energy_models import IsingEnergyFunction
from typing import Union
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_qulacs.qulacs_backend import QulacsBackend
import qiskit 
#from qiskit_aer import AerSimulator
#import time as time_




class CircuitMaker:

    """
    Class which initialises a circuit for a given problem.

    Can be initialised, then tasked with making a circuit given a new input string s.

    Taken mostly from https://github.com/pafloxy/quMCMC but restructured

    """

    def __init__(self, model: IsingEnergyFunction, gamma: Union[float, tuple] , time: Union[int, tuple]):

        """
        Initialise Circuit_Maker object.

        Args:

        model: obj
            energy model object, normally IsingEnergyFunction from energy_models
        gamma: float or tuple
            parameter in QeMCMC circuit
        time: int or tuple
            parameter in QeMCMC circuit






        """
        
        
        self.time = time
        self.gamma = gamma
        self.model = model
        
        # To fix sign issues
        # Very helpful when using different forms of Ising models
        if self.model.cost_function_signs[0] == 1:
            self.h = -self.model.h
        else:
            self.h = self.model.h
        if self.model.cost_function_signs[1] == 1:
            self.J = -self.model.J
        else:
            self.J = self.model.J
            
            
        self.delta_time = 0.8
        self.n_spins = model.num_spins   
        self.alpha = model.alpha
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

        #create trotter circuit that is irrelevant of the input string
        self.qc_evol_h1 = self.fn_qc_h1()  
        self.qc_evol_h2 = self.fn_qc_h2()
        
        

        

        self.trotter_ckt = self.trottered_qc_for_transition(self.qc_evol_h1, self.qc_evol_h2, self.num_trotter_steps)
        
        # QISKIT TRANSPILING IS SO SLOW
        
        #simulator = AerSimulator()
        #self.trotter_ckt = qiskit.transpile(self.trotter_ckt, simulator)
        
        #self.trotter_ckt.decompose(16)
        #backend = AerSimulator(method='statevector')
        #backend = AerSimulator(method="automatic")
        #backend = AerSimulator(method='extended_stabilizer')
        #backend = AerSimulator(method='unitary')
        #backend = AerSimulator(method='superop')
        #backend = AerSimulator(method='stabilizer')
        #backend = AerSimulator(method='density_matrix')

        
        backend = QulacsBackend()
        config = backend.configuration()
        fixed_basis_gates = np.delete(np.array(config.basis_gates),-4)
        
        self.pm = qiskit.transpiler.generate_preset_pass_manager(optimization_level=0,
            basis_gates = fixed_basis_gates)
        self.trotter_ckt = self.pm.run(self.trotter_ckt)
        
        #config = backend.configuration()
        #self.pm = qiskit.transpiler.generate_preset_pass_manager(optimization_level=3,
        #basis_gates = config.basis_gates)
        #self.trotter_ckt = self.pm.run(self.trotter_ckt)
        
        

    def build_circuit(self, s:str) -> QuantumCircuit:
        """
        Build a quantum circuit for a given bitstring.
        Args:
            s (str): The bitstring for which the quantum circuit is to be built.
        Returns:
            QuantumCircuit: The combined quantum circuit for the given bitstring.
        """
        
        if type(s) is not str:
            raise TypeError("s must be a string in build_circuit in CircuitMaker")
        
        #build a circuit for a given bitstring
        qc_s = self.initialise_qc(s)
        qc_s.compose(self.trotter_ckt, inplace=True)
        
        
        #qc_for_s = self.combine_2_qc(qc_s, self.trotter_ckt)# i can get rid of this!

        return qc_s

    def get_state_obtained_binary(self, s: str) -> str:
        """
        Get the output bitstring s' given an input bitstring s.
        This method builds a quantum circuit based on the input bitstring `s`, 
        initializes a quantum state, updates the quantum 
        state with the circuit, and then samples the resulting state to obtain the 
        output bitstring in binary format.
        Args:
            s (str): The input bitstring.
        Returns:
            str: The output bitstring in binary format.
        """
        
        if type(s) is not str:
            raise TypeError("s must be a string in get_state_obtained_binary in CircuitMaker")
        
        #get the output bitstring s' given s

        qc_for_s = self.build_circuit(s)
        
        #q_state= QuantumState(qubit_count=self.n_spins)
        #q_state.set_zero_state()
        #qc_for_s.update_quantum_state(q_state)

        #state_obtained=q_state.sampling(sampling_count=1)[0]
        #state_obtained_binary=f"{state_obtained:0{self.n_spins}b}"
        
        
        # Use Qiskit-Qulacs to run the circuit
        
        
        
        backend = QulacsBackend()
        
        result = backend.run(qc_for_s, shots = 1).result()
        state_obtained_binary = list(result.data()["counts"].keys())[0]
        
        
        #backend = AerSimulator(method='statevector')
        # Add measurement to all qubits
        #qc_for_s.measure_all()
        #result = backend.run(qc_for_s, shots = 1).result()
        
        #result_data = result.data()
                
        #state_obtained_binary = list(result_data["counts"].keys())[0]
        #state_obtained_binary = int(state_obtained_binary,0)
        #state_obtained_binary = f"{state_obtained_binary:0{self.n_spins}b}"
        
        
        #print("s:", s)
        #print("s_prime:", state_obtained_binary)
        
        return state_obtained_binary


    def initialise_qc(self,s : str) -> QuantumCircuit :
        """
        Initializes a quantum circuit based on a given binary string.
        Args:
            s (str): A binary string where each character represents the initial state of a qubit.
                    '1' indicates that the qubit should be in the |1⟩ state, and '0' indicates that
                    the qubit should be in the |0⟩ state.
        Returns:
            QuantumCircuit: A quantum circuit with the specified initial states for each qubit.
        Raises:
            AssertionError: If the length of the input string `s` does not match the number of qubits.
        """
        
        if type(s) is not str:
            raise TypeError("s must be a string in initialise_qc in CircuitMaker")
        

        qc_in=QuantumCircuit(self.n_spins)
        len_str_in = len(s)

        for i in range(0,len(s)):
            if s[i]=="1":
                qc_in.x(len_str_in - 1 - i)
        return qc_in



    def fn_qc_h1(self) -> QuantumCircuit :
        """
        Create a Quantum Circuit for time-evolution under Hamiltonian H1.
        The Hamiltonian H1 is described as:
        H1 = -(1-gamma)*alpha*sum_{j=1}^{n}[(h_j*Z_j)] + gamma *sum_{j=1}^{n}[(X_j)]
        This function constructs a quantum circuit that simulates the time evolution
        under the Hamiltonian H1 for a given time step `delta_time`.
        Returns:
            QuantumCircuit: A quantum circuit representing the time evolution under H1.
        """

        a=self.gamma
        b_list = ((self.gamma-1)*self.alpha)* np.array(self.h)
        qc_h1 = QuantumCircuit(self.n_spins)
        
        X = SparsePauliOp("X")
        Z = SparsePauliOp("Z")
        
        for j in range(0, self.n_spins):
            
            #Matrix = np.round(np.expm(-1j*self.delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)

            #unitary_gate=DenseMatrix(index=self.n_spins-1-j,
            #                matrix = Matrix)
            
            #qc_h1.add_gate(unitary_gate)
            

            H = self.delta_time*(a*X(2)+b_list[j]*Z(2))
            #evo_gate = qiskit.synthesis.MatrixExponential(H).decompose(4)
            evo_gate = PauliEvolutionGate(H, time=1)




            #qc_h1.draw(output="mpl", filename='plots/qc_h1_before.png')

            
            qc_h1.append(evo_gate, [self.n_spins-1-j])
            
            
            #print("you havent checked this works.. code 14523689 circuit maker")
            
        return qc_h1




    def fn_qc_h2(self) -> QuantumCircuit :
        
        """
        Hamiltonian H2 (described in the paper).
        This function constructs a quantum circuit that simulates the 
        time evolution of a system under the Hamiltonian H2. The Hamiltonian 
        is represented by the interaction matrix `self.J`, and the evolution 
        parameters are determined by `self.gamma` and `self.delta_time`.
        Returns:
        -------
        QuantumCircuit
            A quantum circuit representing the time evolution under the Hamiltonian H2.
        """

        

        
        self.n_spins=np.shape(self.J)[0]
        qc_for_evol_h2=QuantumCircuit(self.n_spins)
        upper_triag_without_diag=np.triu(self.J,k=1)
        theta_array=(-2*(1-self.gamma)*self.alpha*self.delta_time)*upper_triag_without_diag
        for j in range(0,self.n_spins-1):
            for k in range(j+1,self.n_spins):
                target_list=[self.n_spins-1-j,self.n_spins-1-k]
                angle = theta_array[j,k]
                
                
                qc_for_evol_h2.rzz(angle, target_list[0], target_list[1])
                
                #qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle = angle)
                

        return qc_for_evol_h2


    def trottered_qc_for_transition(self, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
        """
        Returns a Trotterized quantum circuit.
        This method constructs a quantum circuit that approximates the evolution under the combined Hamiltonians
        H1 and H2 using the Trotter-Suzuki decomposition. The resulting circuit is of the form:
        (evolution_under_h2 X evolution_under_h1)^(num_trotter_steps-1) (evolution under h1).
        Args:
            qc_h1 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H1.
            qc_h2 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H2.
            num_trotter_steps (int): The number of Trotter steps to use in the decomposition.
        Returns:
            QuantumCircuit: The Trotterized quantum circuit representing the combined evolution.
        """
        
        
        qc_combine=QuantumCircuit(self.n_spins)
        for _ in range(0,num_trotter_steps-1):

            qc_combine.compose(qc_h1, inplace=True)
            qc_combine.compose(qc_h2, inplace=True)
            #qc_combine.merge_circuit(qc_h1)
            #qc_combine.merge_circuit(qc_h2)

        #qc_combine.merge_circuit(qc_h1)
        qc_combine.compose(qc_h1, inplace=True)

        return qc_combine



    
    def get_statevector_obtained(self, s: str) -> str:
        """
        Get the output statevector representing probability of s^{prime}s given an input bitstring s.
        This method builds a quantum circuit based on the input bitstring `s`, 
        initializes a quantum state, updates the quantum 
        state with the circuit, and then samples the resulting state to obtain the 
        output bitstring in binary format.
        Args:
            s (str): The input bitstring.
        Returns:
            str: The output bitstring in binary format.
        """
        
        if type(s) is not str:
            raise TypeError("s must be a string in get_state_obtained_binary in CircuitMaker")
        
        #get the output bitstring s' given s

        qc_for_s = self.build_circuit(s)
        
        #q_state= QuantumState(qubit_count=self.n_spins)
        #q_state.set_zero_state()
        #qc_for_s.update_quantum_state(q_state)

        #state_obtained=q_state.sampling(sampling_count=1)[0]
        #state_obtained_binary=f"{state_obtained:0{self.n_spins}b}"
        
        
        # Use Qiskit-Qulacs to run the circuit
        
        
        backend = QulacsBackend()
        #backend = AerSimulator(method='statevector')
            
        
        
        #qc_for_s.draw(output="text", filename='plots/qc_for_s.png')
        qc_for_s.save_statevector()

        result = backend.run(qc_for_s).result()
        statevector = result.get_statevector(qc_for_s)
        
        return statevector.probabilities()

