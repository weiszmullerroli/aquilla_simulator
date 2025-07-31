"""
Aquila Quantum Simulation Package

A comprehensive package for simulating quantum dynamics on hexagonal lattices
using the Aquila quantum system with optimized sparse matrix operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, kron, identity, diags, eye
from scipy.sparse.linalg import expm_multiply
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from typing import List, Tuple, Union, Callable, Dict, Optional
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import pickle
import os


@dataclass
class SimulationParams:
    """Parameters for quantum simulation."""
    d0: float  # Distance from center to each atom (μm)
    omega: float  # Rabi frequency (MHz)
    delta: float  # Detuning (MHz)
    runtime: float = 4.0  # Simulation time (μs)
    n_time_points: int = 200  # Number of time evaluation points
    
    def __post_init__(self):
        # Convert nanoseconds to microseconds if needed
        if self.runtime < 0.1:  # Assume it's in nanoseconds
            self.runtime = self.runtime / 1000.0
            

class QuantumOperator:
    """A class to represent quantum operators for the Aquila system."""
    
    def __init__(self, matrix: Union[np.ndarray, str], qubits: List[int], n_total: int = None):
        """
        Initialize a quantum operator.
        
        Args:
            matrix: Either a 2x2 numpy array or a string like "I", "X", "Y", "Z", "I-Z"
            qubits: List of qubit indices this operator acts on
            n_total: Total number of qubits in the system
        """
        self.qubits = qubits
        self.n_total = n_total
        
        if isinstance(matrix, str):
            self.matrix = self._parse_pauli_string(matrix)
        else:
            self.matrix = np.array(matrix, dtype=complex)
    
    def _parse_pauli_string(self, pauli_str: str) -> np.ndarray:
        """Parse Pauli string expressions like 'I-Z'."""
        # Define Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        if '-' in pauli_str:
            parts = pauli_str.split('-')
            if len(parts) == 2:
                return pauli_dict[parts[0]] - pauli_dict[parts[1]]
        elif '+' in pauli_str:
            parts = pauli_str.split('+')
            if len(parts) == 2:
                return pauli_dict[parts[0]] + pauli_dict[parts[1]]
        else:
            return pauli_dict[pauli_str]
        
        raise ValueError(f"Unknown Pauli string: {pauli_str}")
    
    def to_full_matrix(self, n_total: int) -> csr_matrix:
        """Convert to full matrix representation for n_total qubits using optimized sparse operations."""
        if len(self.qubits) == 1:
            # Single qubit operator - use optimized Kronecker product
            qubit_idx = self.qubits[0] - 1  # Convert to 0-indexed
            
            # Build identity matrices for left and right sides
            left_dim = 2**qubit_idx
            right_dim = 2**(n_total - qubit_idx - 1)
            
            if left_dim > 1:
                left_id = eye(left_dim, format='csr', dtype=complex)
            else:
                left_id = None
                
            if right_dim > 1:
                right_id = eye(right_dim, format='csr', dtype=complex)
            else:
                right_id = None
            
            op_matrix = csr_matrix(self.matrix, dtype=complex)
            
            # Optimized Kronecker product construction
            if left_id is not None and right_id is not None:
                result = kron(kron(left_id, op_matrix), right_id)
            elif left_id is not None:
                result = kron(left_id, op_matrix)
            elif right_id is not None:
                result = kron(op_matrix, right_id)
            else:
                result = op_matrix
                
            return result
            
        elif len(self.qubits) == 2:
            # Two qubit operator - optimized for van der Waals terms
            if np.array_equal(self.matrix, np.array([[0, 0], [0, 1]], dtype=complex)):
                # This is a projector onto |1⟩ state
                q1, q2 = self.qubits[0] - 1, self.qubits[1] - 1  # Convert to 0-indexed
                
                # Create diagonal matrix more efficiently
                diag_elements = np.zeros(2**n_total, dtype=complex)
                for state in range(2**n_total):
                    # Check if both qubits are in |1⟩ state
                    if (state >> (n_total - 1 - q1)) & 1 and (state >> (n_total - 1 - q2)) & 1:
                        diag_elements[state] = 1.0
                
                return diags(diag_elements, format='csr', dtype=complex)
            else:
                # General two-qubit case (fallback)
                q1, q2 = self.qubits[0] - 1, self.qubits[1] - 1
                matrices = []
                
                for i in range(n_total):
                    if i == q1 or i == q2:
                        matrices.append(csr_matrix(self.matrix))
                    else:
                        matrices.append(eye(2, format='csr', dtype=complex))
                
                result = matrices[0]
                for mat in matrices[1:]:
                    result = kron(result, mat)
                return result
        
        else:
            raise ValueError("Only single and two-qubit operators supported")


class LatticeGeometry:
    """Class for managing lattice geometries."""
    
    @staticmethod
    def hexagonal_lattice(d0: float) -> List[Tuple[float, float]]:
        """
        Generate a hexagonal lattice with 6 atoms.
        
        Args:
            d0: Distance from center to each atom
            
        Returns:
            List of (x, y) positions for the 6 atoms
        """
        positions = []
        for k in range(6):
            x = d0 * np.cos(k * np.pi / 3)
            y = d0 * np.sin(-k * np.pi / 3)  # Negative for clockwise orientation
            positions.append((x, y))
        
        return positions
    
    @staticmethod
    def euclidean_distance(pos1: Tuple[float, ...], pos2: Tuple[float, ...]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))


class DriveProtocol:
    """Class for managing time-dependent drive protocols."""
    
    def __init__(self, time_points: List[float], omega_values: List[float], 
                 delta_values: List[float], phi_values: List[float] = None):
        """
        Initialize drive protocol.
        
        Args:
            time_points: List of time points
            omega_values: Rabi frequency values
            delta_values: Detuning values
            phi_values: Phase values (default: all zeros)
        """
        self.time_points = time_points
        self.omega_values = omega_values
        self.delta_values = delta_values
        self.phi_values = phi_values if phi_values else [0] * len(time_points)
        
        # Create interpolated functions
        self.omega_func, self.delta_func, self.phi_func = self._create_interpolations()
    
    def _create_interpolations(self) -> Tuple[Callable, Callable, Callable]:
        """Create interpolated drive functions."""
        # Pad shorter lists to match time length
        max_len = len(self.time_points)
        
        def pad_list(lst, length):
            if len(lst) < length:
                padded = lst + [lst[-1] if lst else 0] * (length - len(lst))
                return padded
            return lst[:length]
        
        omega_padded = pad_list(self.omega_values, max_len)
        delta_padded = pad_list(self.delta_values, max_len)
        phi_padded = pad_list(self.phi_values, max_len)
        
        # Create interpolation functions
        omega_func = interp1d(self.time_points, omega_padded, kind='linear', 
                             bounds_error=False, fill_value='extrapolate')
        delta_func = interp1d(self.time_points, delta_padded, kind='linear', 
                             bounds_error=False, fill_value='extrapolate')
        phi_func = interp1d(self.time_points, phi_padded, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
        
        return omega_func, delta_func, phi_func
    
    @classmethod
    def simple_ramp(cls, runtime: float, omega: float, delta_start: float, delta_end: float):
        """Create a simple linear ramp protocol."""
        time_points = [0, 0.05, runtime - 0.05, runtime]
        omega_values = [0, omega, omega, 0]
        delta_values = [delta_start, delta_start, delta_end, delta_end]
        return cls(time_points, omega_values, delta_values)


class AquilaHamiltonian:
    """Class for constructing the Aquila Hamiltonian."""
    
    def __init__(self, sites: List[Tuple[float, ...]], c6: float = 2 * np.pi * 862690):
        """
        Initialize Hamiltonian constructor.
        
        Args:
            sites: List of qubit positions
            c6: van der Waals coefficient in MHz·μm^6
        """
        self.sites = sites
        self.c6 = c6
        self.n = len(sites)
        self._precompute_operators()
    
    def _precompute_operators(self):
        """Pre-compute time-independent operators for efficiency."""
        print(f"Pre-computing Hamiltonian operators for {self.n} atoms...")
        
        # Pre-compute all single-qubit X and Y operators
        self.rabi_ops_real = []  # X operators
        self.rabi_ops_imag = []  # Y operators
        
        for j in range(1, self.n + 1):
            x_op = QuantumOperator("X", [j])
            x_matrix = x_op.to_full_matrix(self.n)
            self.rabi_ops_real.append(x_matrix)
            
            y_op = QuantumOperator("Y", [j])
            y_matrix = y_op.to_full_matrix(self.n)
            self.rabi_ops_imag.append(y_matrix)
        
        # Pre-compute all detuning operators
        self.detuning_ops = []
        for j in range(1, self.n + 1):
            detuning_op = QuantumOperator("I-Z", [j])
            detuning_matrix = 0.5 * detuning_op.to_full_matrix(self.n)
            self.detuning_ops.append(detuning_matrix)
        
        # Pre-compute van der Waals interaction operators
        self.vdw_ops = []
        self.vdw_strengths = []
        
        for j in range(1, self.n + 1):
            for k in range(j + 1, self.n + 1):
                distance = LatticeGeometry.euclidean_distance(self.sites[j-1], self.sites[k-1])
                
                if distance > 0:
                    vdw_strength = self.c6 / (4 * distance**6)
                    self.vdw_strengths.append(vdw_strength)
                    
                    proj_matrix = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
                    vdw_op = QuantumOperator(proj_matrix, [j, k])
                    vdw_matrix = vdw_op.to_full_matrix(self.n)
                    self.vdw_ops.append(vdw_matrix)
        
        print(f"Pre-computed {len(self.rabi_ops_real)} Rabi operators")
        print(f"Pre-computed {len(self.detuning_ops)} detuning operators") 
        print(f"Pre-computed {len(self.vdw_ops)} van der Waals operators")
    
    def time_dependent_hamiltonian(self, drive_protocol: DriveProtocol) -> Callable:
        """
        Create time-dependent Hamiltonian function.
        
        Args:
            drive_protocol: Drive protocol object
            
        Returns:
            Function H(t) that returns sparse Hamiltonian at time t
        """
        def hamiltonian(t):
            omega = drive_protocol.omega_func(t)
            delta = drive_protocol.delta_func(t)
            phi = drive_protocol.phi_func(t)
            
            # Start with zero Hamiltonian
            H = csr_matrix((2**self.n, 2**self.n), dtype=complex)
            
            # Add Rabi terms with phase
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            for j in range(self.n):
                H += (omega / 2) * (cos_phi * self.rabi_ops_real[j] + sin_phi * self.rabi_ops_imag[j])
            
            # Add detuning terms
            for j in range(self.n):
                H -= delta * self.detuning_ops[j]
            
            # Add van der Waals interactions
            for i, vdw_strength in enumerate(self.vdw_strengths):
                H += vdw_strength * self.vdw_ops[i]
            
            return H
        
        return hamiltonian


class QuantumEvolution:
    """Class for quantum state evolution."""
    
    @staticmethod
    def evolve_sparse(hamiltonian_func: Callable, initial_state: np.ndarray, 
                     t_span: Tuple[float, float], t_eval: np.ndarray = None) -> dict:
        """
        Evolve quantum state using sparse matrix operations.
        
        Args:
            hamiltonian_func: Function H(t) returning sparse Hamiltonian
            initial_state: Initial quantum state vector
            t_span: (t_start, t_end) tuple
            t_eval: Time points to evaluate
            
        Returns:
            Dictionary with 't' and 'y' arrays
        """
        def schrodinger_eq_sparse(t, psi):
            H = hamiltonian_func(t)
            return -1j * H.dot(psi)
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)
        
        sol = solve_ivp(schrodinger_eq_sparse, t_span, initial_state, 
                       t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10,
                       max_step=0.01)
        
        return {'t': sol.t, 'y': sol.y.T}


class AquilaSimulator:
    """Main simulator class combining all components."""
    
    def __init__(self, d0: float, c6: float = 2 * np.pi * 862690):
        """
        Initialize the Aquila simulator.
        
        Args:
            d0: Distance from center to each atom (μm)
            c6: van der Waals coefficient
        """
        self.d0 = d0
        self.c6 = c6
        self.sites = LatticeGeometry.hexagonal_lattice(d0)
        self.hamiltonian_builder = AquilaHamiltonian(self.sites, c6)
        self.n_atoms = len(self.sites)
        
        # Ground state
        self.ground_state = np.zeros(2**self.n_atoms, dtype=complex)
        self.ground_state[0] = 1.0
    
    def run_simulation(self, params: SimulationParams) -> Dict:
        """
        Run a complete simulation.
        
        Args:
            params: Simulation parameters
            
        Returns:
            Dictionary containing results
        """
        # Create drive protocol
        drive = DriveProtocol.simple_ramp(
            params.runtime, params.omega, 
            params.delta, -params.delta
        )
        
        # Create Hamiltonian
        H_func = self.hamiltonian_builder.time_dependent_hamiltonian(drive)
        
        # Time evolution
        t_eval = np.linspace(0, params.runtime, params.n_time_points)
        evolution = QuantumEvolution.evolve_sparse(
            H_func, self.ground_state, (0, params.runtime), t_eval
        )
        
        # Extract final state populations
        final_state = evolution['y'][-1, :]
        final_populations = np.abs(final_state)**2
        
        return {
            'params': params,
            'evolution': evolution,
            'final_populations': final_populations,
            'sites': self.sites
        }
    
    def get_computational_basis_probabilities(self, final_populations: np.ndarray) -> Dict[str, float]:
        """
        Convert final populations to computational basis states.
        
        Args:
            final_populations: Array of final state probabilities
            
        Returns:
            Dictionary mapping basis states to probabilities
        """
        basis_probs = {}
        for i, prob in enumerate(final_populations):
            if prob > 1e-10:  # Only include significant probabilities
                state_str = format(i, f"0{self.n_atoms}b")
                basis_probs[state_str] = prob
        
        return basis_probs


class ParameterSweep:
    """Class for running parameter sweeps and generating phase diagrams."""
    
    def __init__(self, d0_range: List[float], omega_range: List[float], 
                 delta_range: List[float], runtime: float = 4.0):
        """
        Initialize parameter sweep.
        
        Args:
            d0_range: List of d0 values to test
            omega_range: List of omega values to test
            delta_range: List of delta values to test
            runtime: Simulation runtime
        """
        self.d0_range = d0_range
        self.omega_range = omega_range
        self.delta_range = delta_range
        self.runtime = runtime
        self.results = {}
    
    def run_sweep(self, n_processes: int = None, save_results: bool = True) -> Dict:
        """
        Run parameter sweep using multiprocessing.
        
        Args:
            n_processes: Number of processes (default: CPU count)
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing all results
        """
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        # Generate all parameter combinations
        param_combinations = []
        for d0 in self.d0_range:
            for omega in self.omega_range:
                for delta in self.delta_range:
                    params = SimulationParams(d0=d0, omega=omega, delta=delta, runtime=self.runtime)
                    param_combinations.append(params)
        
        print(f"Running {len(param_combinations)} simulations with {n_processes} processes...")
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = {executor.submit(self._run_single_simulation, params): params 
                      for params in param_combinations}
            
            results = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
                params = futures[future]
                try:
                    result = future.result()
                    key = (params.d0, params.omega, params.delta)
                    results[key] = result
                except Exception as e:
                    print(f"Simulation failed for {params}: {e}")
        
        self.results = results
        
        if save_results:
            self.save_results()
        
        return results
    
    @staticmethod
    def _run_single_simulation(params: SimulationParams) -> Dict:
        """Run a single simulation (for multiprocessing)."""
        simulator = AquilaSimulator(params.d0)
        result = simulator.run_simulation(params)
        
        # Extract key information to reduce memory usage
        final_pops = result['final_populations']
        basis_probs = simulator.get_computational_basis_probabilities(final_pops)
        
        return {
            'params': params,
            'final_populations': final_pops,
            'basis_probabilities': basis_probs,
            'max_population': np.max(final_pops),
            'entropy': -np.sum(final_pops * np.log2(final_pops + 1e-16))
        }
    
    def save_results(self, filename: str = "aquila_sweep_results.pkl"):
        """Save results to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename: str = "aquila_sweep_results.pkl"):
        """Load results from file."""
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Results loaded from {filename}")


class Visualization:
    """Class for creating visualizations and phase diagrams."""
    
    @staticmethod
    def plot_lattice(sites: List[Tuple[float, float]], d0: float):
        """Plot hexagonal lattice."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        x_coords = [pos[0] for pos in sites]
        y_coords = [pos[1] for pos in sites]
        
        ax.scatter(x_coords, y_coords, s=200, c='blue', alpha=0.7)
        
        for i, (x, y) in enumerate(sites):
            ax.annotate(str(i + 1), (x, y), xytext=(0, 15), 
                       textcoords='offset points', ha='center', va='bottom',
                       fontweight='bold', fontsize=12)
        
        ax.set_xlabel('x (μm)', fontsize=12)
        ax.set_ylabel('y (μm)', fontsize=12)
        ax.set_title(f'Hexagonal Lattice (d0 = {d0:.2f} μm)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_ground_state_population(evolution_data: Dict, params: SimulationParams = None, 
                                   title_suffix: str = ""):
        """
        Plot ground state population evolution over time.
        
        Args:
            evolution_data: Dictionary with 't' and 'y' arrays from evolution
            params: SimulationParams object for title information
            title_suffix: Additional text for title
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Ground state is the first component (index 0)
        ground_state_population = np.abs(evolution_data['y'][:, 0])**2
        time_points = evolution_data['t']
        
        # Plot with thick line
        ax.plot(time_points, ground_state_population, 'b-', linewidth=2.5, 
                label='Ground State |000000⟩')
        
        # Formatting to match the style in your image
        ax.set_xlabel('t (μs)', fontsize=14)
        ax.set_ylabel('|⟨ψ_f|ψ_0⟩|²', fontsize=14)
        
        # Create title with parameters if provided
        if params:
            title = f'Ground State Population Evolution{title_suffix}\n'
            title += f'd₀ = {params.d0:.2f} μm, Ω = {params.omega:.1f} MHz, Δ = {params.delta:.1f} MHz'
        else:
            title = f'Ground State Population Evolution{title_suffix}'
        
        ax.set_title(title, fontsize=12)
        
        # Grid and formatting
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)  # Probability should be between 0 and 1
        
        # Add legend
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_phase_diagram(results: Dict, fixed_param: str = 'd0', fixed_value: float = None):
        """
        Create phase diagram from parameter sweep results.
        
        Args:
            results: Results dictionary from parameter sweep
            fixed_param: Which parameter to fix ('d0', 'omega', or 'delta')
            fixed_value: Value of the fixed parameter
        """
        # Extract data for phase diagram
        param_sets = list(results.keys())
        
        if fixed_param == 'd0':
            if fixed_value is None:
                fixed_value = param_sets[0][0]  # Use first d0 value
            
            # Filter results for fixed d0
            filtered_results = {k: v for k, v in results.items() if abs(k[0] - fixed_value) < 1e-10}
            
            # Extract omega and delta values
            omegas = sorted(list(set([k[1] for k in filtered_results.keys()])))
            deltas = sorted(list(set([k[2] for k in filtered_results.keys()])))
            
            # Create meshgrid
            X, Y = np.meshgrid(omegas, deltas)
            Z = np.zeros_like(X)
            
            # Fill in ground state populations (final)
            for i, delta in enumerate(deltas):
                for j, omega in enumerate(omegas):
                    key = (fixed_value, omega, delta)
                    if key in filtered_results:
                        # Ground state is index 0
                        ground_state_pop = filtered_results[key]['final_populations'][0]
                        Z[i, j] = ground_state_pop
            
            xlabel = 'Omega (MHz)'
            ylabel = 'Delta (MHz)'
            title = f'Ground State Population Phase Diagram (d0 = {fixed_value:.2f} μm)'
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Final Ground State Population', fontsize=12)
        
        plt.tight_layout()
        return fig, ax


def aquila_analysis(mode: str = "single", 
                   d0: Union[float, List[float]] = 7.51,
                   omega: Union[float, List[float]] = 8.0, 
                   delta: Union[float, List[float]] = 30.0,
                   runtime: float = 4.0,
                   n_time_points: int = 400,
                   n_processes: int = None,
                   plot_evolution: bool = True,
                   plot_phase_diagram: bool = True,
                   save_results: bool = True,
                   show_plots: bool = True) -> Dict:
    """
    Unified function for Aquila quantum analysis - handles both single simulations and parameter sweeps.
    
    Args:
        mode: "single" for single simulation, "sweep" for parameter sweep
        d0: Distance parameter(s) in μm. Single float or list of floats
        omega: Rabi frequency(s) in MHz. Single float or list of floats  
        delta: Detuning(s) in MHz. Single float or list of floats
        runtime: Simulation time in μs
        n_time_points: Number of time evaluation points
        n_processes: Number of processes for parameter sweep (default: CPU count)
        plot_evolution: Whether to plot ground state evolution
        plot_phase_diagram: Whether to create phase diagrams (sweep mode only)
        save_results: Whether to save results to file
        show_plots: Whether to display plots
        
    Returns:
        Dictionary containing results and analysis
        
    Examples:
        # Single simulation
        result = aquila_analysis(mode="single", d0=7.5, omega=8.0, delta=30.0)
        
        # Parameter sweep
        result = aquila_analysis(
            mode="sweep", 
            d0=[7.0, 8.0], 
            omega=[6.0, 8.0, 10.0], 
            delta=[20.0, 30.0, 40.0]
        )
    """
    
    print("=" * 80)
    print("AQUILA QUANTUM SIMULATION ANALYSIS")
    print("=" * 80)
    
    if mode == "single":
        # Single simulation mode
        print(f"Mode: Single Simulation")
        print(f"Parameters: d0={d0} μm, Ω={omega} MHz, Δ={delta} MHz")
        print(f"Runtime: {runtime} μs")
        
        # Create simulator
        simulator = AquilaSimulator(d0=d0)
        
        # Set up parameters
        params = SimulationParams(
            d0=d0, omega=omega, delta=delta, 
            runtime=runtime, n_time_points=n_time_points
        )
        
        # Run simulation
        print("Running simulation...")
        result = simulator.run_simulation(params)
        
        # Extract ground state population evolution
        ground_state_pops = np.abs(result['evolution']['y'][:, 0])**2
        final_ground_state_pop = ground_state_pops[-1]
        
        print(f"Final ground state population: {final_ground_state_pop:.6f}")
        
        # Plot ground state evolution
        if plot_evolution:
            fig, ax = Visualization.plot_ground_state_population(
                result['evolution'], params
            )
            if show_plots:
                plt.show()
            if save_results:
                fig.savefig(f'ground_state_evolution_d0_{d0}_omega_{omega}_delta_{delta}.png', 
                           dpi=300, bbox_inches='tight')
        
        # Get computational basis probabilities
        basis_probs = simulator.get_computational_basis_probabilities(result['final_populations'])
        
        # Show top states
        print(f"\nTop 5 computational basis states:")
        sorted_states = sorted(basis_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (state, prob) in enumerate(sorted_states, 1):
            n_excited = state.count('1')
            print(f"  {i}. |{state}⟩ ({n_excited} excited): {prob:.6f}")
        
        return {
            'mode': 'single',
            'params': params,
            'evolution': result['evolution'],
            'final_populations': result['final_populations'],
            'ground_state_evolution': ground_state_pops,
            'final_ground_state_pop': final_ground_state_pop,
            'basis_probabilities': basis_probs,
            'simulator': simulator
        }
    
    elif mode == "sweep":
        # Parameter sweep mode
        print(f"Mode: Parameter Sweep")
        
        # Convert single values to lists
        d0_range = [d0] if isinstance(d0, (int, float)) else d0
        omega_range = [omega] if isinstance(omega, (int, float)) else omega
        delta_range = [delta] if isinstance(delta, (int, float)) else delta
        
        print(f"Parameter ranges:")
        print(f"  d0: {d0_range} μm")
        print(f"  Omega: {omega_range} MHz")
        print(f"  Delta: {delta_range} MHz")
        print(f"  Total combinations: {len(d0_range) * len(omega_range) * len(delta_range)}")
        print(f"  Runtime: {runtime} μs")
        
        # Create parameter sweep
        sweep = ParameterSweep(d0_range, omega_range, delta_range, runtime=runtime)
        
        # Run sweep
        print(f"Running parameter sweep with {n_processes or 'auto'} processes...")
        results = sweep.run_sweep(n_processes=n_processes, save_results=save_results)
        
        print(f"Parameter sweep completed with {len(results)} simulations!")
        
        # Analyze results
        ground_state_data = {}
        final_ground_pops = {}
        
        for params_key, result in results.items():
            d0_val, omega_val, delta_val = params_key
            # Ground state is index 0
            final_ground_pop = result['final_populations'][0]
            final_ground_pops[params_key] = final_ground_pop
            
            # Store for potential plotting
            ground_state_data[params_key] = final_ground_pop
        
        # Find best and worst ground state retention
        best_params = max(final_ground_pops.items(), key=lambda x: x[1])
        worst_params = min(final_ground_pops.items(), key=lambda x: x[1])
        
        print(f"\nGround state population analysis:")
        print(f"Best retention: d0={best_params[0][0]}, Ω={best_params[0][1]}, Δ={best_params[0][2]} → {best_params[1]:.6f}")
        print(f"Worst retention: d0={worst_params[0][0]}, Ω={worst_params[0][1]}, Δ={worst_params[0][2]} → {worst_params[1]:.6f}")
        print(f"Average ground state population: {np.mean(list(final_ground_pops.values())):.6f}")
        
        # Create phase diagrams for each d0 value
        if plot_phase_diagram and len(d0_range) > 0:
            print(f"\nCreating phase diagrams...")
            phase_figs = []
            
            for d0_val in d0_range:
                print(f"Creating phase diagram for d0 = {d0_val} μm")
                fig, ax = Visualization.plot_phase_diagram(results, fixed_param='d0', fixed_value=d0_val)
                phase_figs.append((fig, ax, d0_val))
                
                if save_results:
                    fig.savefig(f'ground_state_phase_diagram_d0_{d0_val:.1f}.png', 
                               dpi=300, bbox_inches='tight')
                
                if show_plots:
                    plt.show()
        
        # Plot evolution for best parameters if requested
        if plot_evolution and len(results) > 0:
            print(f"Plotting ground state evolution for best parameters...")
            best_key = best_params[0]
            
            # Re-run simulation with full evolution data for best parameters
            simulator = AquilaSimulator(best_key[0])
            params = SimulationParams(
                d0=best_key[0], omega=best_key[1], delta=best_key[2],
                runtime=runtime, n_time_points=n_time_points
            )
            best_result = simulator.run_simulation(params)
            
            fig, ax = Visualization.plot_ground_state_population(
                best_result['evolution'], params, 
                title_suffix=" (Best Parameters)"
            )
            
            if save_results:
                fig.savefig(f'best_ground_state_evolution.png', dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
        
        return {
            'mode': 'sweep',
            'results': results,
            'ground_state_populations': final_ground_pops,
            'best_params': best_params,
            'worst_params': worst_params,
            'average_ground_pop': np.mean(list(final_ground_pops.values())),
            'parameter_ranges': {
                'd0': d0_range,
                'omega': omega_range, 
                'delta': delta_range
            }
        }
    
    else:
        raise ValueError("Mode must be 'single' or 'sweep'")


# Quick access functions for common use cases
def quick_test(d0: float = 7.51, omega: float = 8.0, delta: float = 30.0, 
               runtime: float = 4.0, show_plots: bool = True) -> Dict:
    """
    Quick test function for single simulation focusing on ground state.
    
    Args:
        d0: Distance parameter (μm)
        omega: Rabi frequency (MHz)
        delta: Detuning (MHz)
        runtime: Simulation time (μs)
        show_plots: Whether to display plots
        
    Returns:
        Analysis results dictionary
    """
    return aquila_analysis(
        mode="single", 
        d0=d0, omega=omega, delta=delta, runtime=runtime,
        show_plots=show_plots
    )


def parameter_scan(d0_range: List[float], omega_range: List[float], delta_range: List[float],
                  runtime: float = 4.0, n_processes: int = None, show_plots: bool = True) -> Dict:
    """
    Parameter scan function focusing on ground state populations.
    
    Args:
        d0_range: List of distance values (μm)
        omega_range: List of Rabi frequency values (MHz)
        delta_range: List of detuning values (MHz)
        runtime: Simulation time (μs)
        n_processes: Number of processes for parallel execution
        show_plots: Whether to display plots
        
    Returns:
        Analysis results dictionary
    """
    return aquila_analysis(
        mode="sweep",
        d0=d0_range, omega=omega_range, delta=delta_range,
        runtime=runtime, n_processes=n_processes, show_plots=show_plots
    )
    # Test single simulation
    print("Testing single simulation...")
    simulator = AquilaSimulator(d0=7.51)
    params = SimulationParams(d0=7.51, omega=8.0, delta=30.0, runtime=4.0)
    result = simulator.run_simulation(params)
    
    print(f"Simulation completed. Final state has {len(result['final_populations'])} components")
    
    # Get computational basis probabilities
    basis_probs = simulator.get_computational_basis_probabilities(result['final_populations'])
    print(f"\nTop 5 basis states:")
    sorted_states = sorted(basis_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for state, prob in sorted_states:
        print(f"|{state}⟩: {prob:.4f}")
    
    # Test parameter sweep (small example)
    print("\nTesting parameter sweep...")
    d0_range = [7.0, 8.0]
    omega_range = [6.0, 8.0]
    delta_range = [25.0, 35.0]
    
    sweep = ParameterSweep(d0_range, omega_range, delta_range, runtime=1.0)  # Short runtime for testing
    results = sweep.run_sweep(n_processes=2)
    
    print(f"Parameter sweep completed with {len(results)} results")
    
    # Create visualization
    Visualization.plot_lattice(simulator.sites, simulator.d0)
    plt.show()