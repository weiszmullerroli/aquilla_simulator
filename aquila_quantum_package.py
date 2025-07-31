"""
Aquila Quantum Simulation Package - Computational States Analysis

A comprehensive package for simulating quantum dynamics on hexagonal lattices
using the Aquila quantum system with focus on computational basis state probabilities.
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
import pandas as pd
import seaborn as sns
from collections import defaultdict


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
    
    def get_computational_basis_probabilities(self, final_populations: np.ndarray, 
                                            threshold: float = 1e-10) -> Dict[str, float]:
        """
        Convert final populations to computational basis states.
        
        Args:
            final_populations: Array of final state probabilities
            threshold: Minimum probability to include
            
        Returns:
            Dictionary mapping basis states to probabilities
        """
        basis_probs = {}
        for i, prob in enumerate(final_populations):
            if prob > threshold:
                state_str = format(i, f"0{self.n_atoms}b")
                basis_probs[state_str] = prob
        
        return basis_probs


class ComputationalStateAnalyzer:
    """Class for analyzing computational basis states and creating visualizations."""
    
    def __init__(self, n_atoms: int = 6):
        """Initialize analyzer."""
        self.n_atoms = n_atoms
        self.state_names = self._generate_state_names()
    
    def _generate_state_names(self) -> List[str]:
        """Generate all possible computational basis state names."""
        return [format(i, f"0{self.n_atoms}b") for i in range(2**self.n_atoms)]
    
    def categorize_states(self, basis_probs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Categorize states by number of excited atoms.
        
        Args:
            basis_probs: Dictionary of state probabilities
            
        Returns:
            Dictionary categorized by excitation number
        """
        categories = defaultdict(dict)
        
        for state, prob in basis_probs.items():
            n_excited = state.count('1')
            categories[f"{n_excited}_excited"][state] = prob
        
        return dict(categories)
    
    def get_top_states(self, basis_probs: Dict[str, float], top_n: int = 20) -> Dict[str, float]:
        """Get top N states by probability."""
        sorted_states = sorted(basis_probs.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_states[:top_n])
    
    def create_state_barplot(self, basis_probs: Dict[str, float], 
                           title: str = "Computational Basis State Probabilities",
                           figsize: Tuple[int, int] = (15, 8),
                           top_n: int = 20,
                           show_excitation_groups: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar plot of computational basis state probabilities.
        
        Args:
            basis_probs: Dictionary of state probabilities
            title: Plot title
            figsize: Figure size
            top_n: Number of top states to show
            show_excitation_groups: Whether to color by excitation number
            
        Returns:
            Figure and axes objects
        """
        # Get top states
        top_states = self.get_top_states(basis_probs, top_n)
        
        if not top_states:
            raise ValueError("No states found with significant probability")
        
        states = list(top_states.keys())
        probs = list(top_states.values())
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if show_excitation_groups:
            # Color by excitation number
            colors = []
            excitation_colors = {
                0: '#1f77b4',  # Blue
                1: '#ff7f0e',  # Orange
                2: '#2ca02c',  # Green
                3: '#d62728',  # Red
                4: '#9467bd',  # Purple
                5: '#8c564b',  # Brown
                6: '#e377c2',  # Pink
            }
            
            for state in states:
                n_excited = state.count('1')
                colors.append(excitation_colors.get(n_excited, '#7f7f7f'))
            
            bars = ax.bar(range(len(states)), probs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        else:
            bars = ax.bar(range(len(states)), probs, color='steelblue', alpha=0.7, 
                         edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Computational Basis States', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([f"|{state}⟩" for state in states], rotation=45, ha='right')
        
        # Add probability values on top of bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{prob:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add legend for excitation groups if colored
        if show_excitation_groups:
            legend_elements = []
            for n_excited, color in excitation_colors.items():
                if any(state.count('1') == n_excited for state in states):
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7,
                                                       label=f'{n_excited} excited'))
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', title='Excitation Level')
        
        # Grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(probs) * 1.15)
        
        plt.tight_layout()
        return fig, ax
    
    def create_excitation_summary_plot(self, basis_probs: Dict[str, float],
                                     title: str = "Probability by Excitation Level",
                                     figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a summary plot showing total probability by excitation level.
        
        Args:
            basis_probs: Dictionary of state probabilities
            title: Plot title
            figsize: Figure size
            
        Returns:
            Figure and axes objects
        """
        # Calculate total probability for each excitation level
        excitation_probs = defaultdict(float)
        for state, prob in basis_probs.items():
            n_excited = state.count('1')
            excitation_probs[n_excited] += prob
        
        # Sort by excitation level
        excitation_levels = sorted(excitation_probs.keys())
        total_probs = [excitation_probs[level] for level in excitation_levels]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bars = ax.bar(excitation_levels, total_probs, 
                     color=[colors[i % len(colors)] for i in excitation_levels],
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, total_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Number of Excited Atoms', fontsize=14)
        ax.set_ylabel('Total Probability', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(excitation_levels)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(total_probs) * 1.2)
        
        plt.tight_layout()
        return fig, ax


class ParameterSweepAnalyzer:
    """Class for running parameter sweeps and analyzing computational states."""
    
    def __init__(self, d0_range: List[float], omega_range: List[float], 
                 delta_range: List[float], runtime: float = 4.0):
        """
        Initialize parameter sweep analyzer.
        
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
        self.analyzer = ComputationalStateAnalyzer()
    
    def run_sweep(self, n_processes: int = None, save_results: bool = True) -> Dict:
        """Run parameter sweep using multiprocessing."""
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
        
        # Extract computational basis probabilities
        basis_probs = simulator.get_computational_basis_probabilities(result['final_populations'])
        
        return {
            'params': params,
            'final_populations': result['final_populations'],
            'basis_probabilities': basis_probs,
        }
    
    def create_comparison_plots(self, parameter_sets: List[Tuple[float, float, float]], 
                              figsize: Tuple[int, int] = (20, 12),
                              top_n: int = 15) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create comparison plots for multiple parameter sets.
        
        Args:
            parameter_sets: List of (d0, omega, delta) tuples to compare
            figsize: Figure size
            top_n: Number of top states to show in each plot
            
        Returns:
            Figure and axes array
        """
        n_sets = len(parameter_sets)
        n_cols = min(3, n_sets)
        n_rows = (n_sets - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_sets == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (d0, omega, delta) in enumerate(parameter_sets):
            key = (d0, omega, delta)
            if key not in self.results:
                print(f"Warning: No results found for parameters {key}")
                continue
            
            result = self.results[key]
            basis_probs = result['basis_probabilities']
            
            if i < len(axes):
                ax = axes[i]
                
                # Get top states for this parameter set
                top_states = self.analyzer.get_top_states(basis_probs, top_n)
                states = list(top_states.keys())
                probs = list(top_states.values())
                
                if states:
                    # Color by excitation number
                    colors = []
                    excitation_colors = {
                        0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 
                        3: '#d62728', 4: '#9467bd', 5: '#8c564b', 6: '#e377c2'
                    }
                    
                    for state in states:
                        n_excited = state.count('1')
                        colors.append(excitation_colors.get(n_excited, '#7f7f7f'))
                    
                    bars = ax.bar(range(len(states)), probs, color=colors, alpha=0.7, 
                                 edgecolor='black', linewidth=0.5)
                    
                    # Customize subplot
                    ax.set_title(f'd₀={d0:.1f}μm, Ω={omega:.1f}MHz, Δ={delta:.1f}MHz', 
                               fontsize=12, fontweight='bold')
                    ax.set_ylabel('Probability', fontsize=10)
                    ax.set_xticks(range(len(states)))
                    ax.set_xticklabels([f"|{state}⟩" for state in states], 
                                     rotation=45, ha='right', fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add probability values on bars for significant ones
                    for bar, prob in zip(bars, probs):
                        if prob > 0.01:  # Only label significant probabilities
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                   f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_sets, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def save_results(self, filename: str = "aquila_computational_states_results.pkl"):
        """Save results to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename: str = "aquila_computational_states_results.pkl"):
        """Load results from file."""
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Results loaded from {filename}")


class Visualization:
    """Enhanced visualization class for computational states analysis."""
    
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
    def plot_ground_state_evolution(evolution_data: Dict, params: SimulationParams = None, 
                                   title_suffix: str = ""):
        """Plot ground state population evolution over time."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Ground state is the first component (index 0)
        ground_state_population = np.abs(evolution_data['y'][:, 0])**2
        time_points = evolution_data['t']
        
        # Plot with thick line
        ax.plot(time_points, ground_state_population, 'b-', linewidth=2.5, 
                label='Ground State |000000⟩')
        
        # Formatting
        ax.set_xlabel('t (μs)', fontsize=14)
        ax.set_ylabel('|⟨ψ_f|ψ_0⟩|²', fontsize=14)
        
        # Create title with parameters if provided
        if params:
            title = f'Ground State Population Evolution{title_suffix}\n'
            title += f'd₀ = {params.d0:.2f} μm, Ω = {params.omega:.1f} MHz, Δ = {params.delta:.1f} MHz'
        else:
            title = f'Ground State Population Evolution{title_suffix}'
        
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        return fig, ax


def aquila_computational_analysis(mode: str = "single", 
                                d0: Union[float, List[float]] = 7.51,
                                omega: Union[float, List[float]] = 8.0, 
                                delta: Union[float, List[float]] = 30.0,
                                runtime: float = 4.0,
                                n_time_points: int = 400,
                                n_processes: int = None,
                                top_n_states: int = 20,
                                plot_evolution: bool = True,
                                plot_states: bool = True,
                                plot_excitation_summary: bool = True,
                                save_results: bool = True,
                                show_plots: bool = True) -> Dict:
    """
    Unified function for Aquila quantum computational states analysis.
    
    Args:
        mode: "single" for single simulation, "sweep" for parameter sweep
        d0: Distance parameter(s) in μm
        omega: Rabi frequency(s) in MHz
        delta: Detuning(s) in MHz
        runtime: Simulation time in μs
        n_time_points: Number of time evaluation points
        n_processes: Number of processes for parameter sweep
        top_n_states: Number of top states to show in plots
        plot_evolution: Whether to plot ground state evolution
        plot_states: Whether to plot computational states
        plot_excitation_summary: Whether to plot excitation level summary
        save_results: Whether to save results to file
        show_plots: Whether to display plots
        
    Returns:
        Dictionary containing results and analysis
    """
    
    print("=" * 80)
    print("AQUILA QUANTUM COMPUTATIONAL STATES ANALYSIS")
    print("=" * 80)
    
    if mode == "single":
        # Single simulation mode
        print(f"Mode: Single Simulation")
        print(f"Parameters: d0={d0} μm, Ω={omega} MHz, Δ={delta} MHz")
        print(f"Runtime: {runtime} μs")
        
        # Create simulator
        simulator = AquilaSimulator(d0=d0)
        analyzer = ComputationalStateAnalyzer()
        
        # Set up parameters
        params = SimulationParams(
            d0=d0, omega=omega, delta=delta, 
            runtime=runtime, n_time_points=n_time_points
        )
        
        # Run simulation
        print("Running simulation...")
        result = simulator.run_simulation(params)
        
        # Get computational basis probabilities
        basis_probs = simulator.get_computational_basis_probabilities(result['final_populations'])
        
        print(f"Found {len(basis_probs)} significant computational basis states")
        
        # Show top states
        top_states = analyzer.get_top_states(basis_probs, 10)
        print(f"\nTop 10 computational basis states:")
        for i, (state, prob) in enumerate(top_states.items(), 1):
            n_excited = state.count('1')
            print(f"  {i}. |{state}⟩ ({n_excited} excited): {prob:.6f}")
        
        # Create visualizations
        figs = {}
        
        if plot_evolution:
            fig_evol, ax_evol = Visualization.plot_ground_state_evolution(
                result['evolution'], params
            )
            figs['evolution'] = fig_evol
            if show_plots:
                plt.show()
            if save_results:
                fig_evol.savefig(f'ground_state_evolution_d0_{d0}_omega_{omega}_delta_{delta}.png', 
                               dpi=300, bbox_inches='tight')
        
        if plot_states:
            fig_states, ax_states = analyzer.create_state_barplot(
                basis_probs, 
                title=f"Computational States (d₀={d0}μm, Ω={omega}MHz, Δ={delta}MHz)",
                top_n=top_n_states
            )
            figs['states'] = fig_states
            if show_plots:
                plt.show()
            if save_results:
                fig_states.savefig(f'computational_states_d0_{d0}_omega_{omega}_delta_{delta}.png', 
                                  dpi=300, bbox_inches='tight')
        
        if plot_excitation_summary:
            fig_exc, ax_exc = analyzer.create_excitation_summary_plot(
                basis_probs,
                title=f"Excitation Summary (d₀={d0}μm, Ω={omega}MHz, Δ={delta}MHz)"
            )
            figs['excitation_summary'] = fig_exc
            if show_plots:
                plt.show()
            if save_results:
                fig_exc.savefig(f'excitation_summary_d0_{d0}_omega_{omega}_delta_{delta}.png', 
                              dpi=300, bbox_inches='tight')
        
        return {
            'mode': 'single',
            'params': params,
            'basis_probabilities': basis_probs,
            'top_states': top_states,
            'excitation_categories': analyzer.categorize_states(basis_probs),
            'figures': figs,
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
        
        # Create parameter sweep analyzer
        sweep_analyzer = ParameterSweepAnalyzer(d0_range, omega_range, delta_range, runtime=runtime)
        
        # Run sweep
        print(f"Running parameter sweep...")
        results = sweep_analyzer.run_sweep(n_processes=n_processes, save_results=save_results)
        
        print(f"Parameter sweep completed with {len(results)} simulations!")
        
        # Create comparison plots for all parameter combinations
        parameter_sets = list(results.keys())
        
        if plot_states and parameter_sets:
            print("Creating computational states comparison plots...")
            fig_comp, axes_comp = sweep_analyzer.create_comparison_plots(
                parameter_sets, top_n=top_n_states
            )
            
            if show_plots:
                plt.show()
            if save_results:
                fig_comp.savefig('computational_states_comparison.png', dpi=300, bbox_inches='tight')
        
        # Analyze most interesting results
        interesting_analysis = {}
        for key, result in results.items():
            basis_probs = result['basis_probabilities']
            
            # Calculate some metrics
            ground_state_prob = basis_probs.get('000000', 0.0)
            max_excited_states = max([state.count('1') for state in basis_probs.keys()] + [0])
            total_prob_accounted = sum(basis_probs.values())
            
            interesting_analysis[key] = {
                'ground_state_prob': ground_state_prob,
                'max_excited_states': max_excited_states,
                'total_prob_accounted': total_prob_accounted,
                'n_significant_states': len(basis_probs)
            }
        
        # Find most interesting cases
        min_ground = min(interesting_analysis.items(), key=lambda x: x[1]['ground_state_prob'])
        max_excited = max(interesting_analysis.items(), key=lambda x: x[1]['max_excited_states'])
        most_diverse = max(interesting_analysis.items(), key=lambda x: x[1]['n_significant_states'])
        
        print(f"\nInteresting parameter combinations:")
        print(f"Lowest ground state: d0={min_ground[0][0]}, Ω={min_ground[0][1]}, Δ={min_ground[0][2]} → {min_ground[1]['ground_state_prob']:.4f}")
        print(f"Highest excitation: d0={max_excited[0][0]}, Ω={max_excited[0][1]}, Δ={max_excited[0][2]} → {max_excited[1]['max_excited_states']} atoms")
        print(f"Most diverse states: d0={most_diverse[0][0]}, Ω={most_diverse[0][1]}, Δ={most_diverse[0][2]} → {most_diverse[1]['n_significant_states']} states")
        
        return {
            'mode': 'sweep',
            'results': results,
            'interesting_analysis': interesting_analysis,
            'parameter_ranges': {
                'd0': d0_range,
                'omega': omega_range, 
                'delta': delta_range
            },
            'min_ground_state': min_ground,
            'max_excited_states': max_excited,
            'most_diverse_states': most_diverse
        }
    
    else:
        raise ValueError("Mode must be 'single' or 'sweep'")


# Quick access functions
def quick_computational_test(d0: float = 7.51, omega: float = 8.0, delta: float = 30.0, 
                           runtime: float = 4.0, show_plots: bool = True) -> Dict:
    """Quick test focusing on computational basis states."""
    return aquila_computational_analysis(
        mode="single", 
        d0=d0, omega=omega, delta=delta, runtime=runtime,
        show_plots=show_plots
    )


def computational_parameter_scan(d0_range: List[float], omega_range: List[float], 
                               delta_range: List[float], runtime: float = 4.0, 
                               n_processes: int = None, show_plots: bool = True) -> Dict:
    """Parameter scan focusing on computational basis states."""
    return aquila_computational_analysis(
        mode="sweep",
        d0=d0_range, omega=omega_range, delta=delta_range,
        runtime=runtime, n_processes=n_processes, show_plots=show_plots
    )


if __name__ == "__main__":
    # Example usage
    print("Testing Aquila Computational States Analysis...")
    
    # Quick single test
    result = quick_computational_test(d0=7.5, omega=8.0, delta=30.0)
    print(f"Found {len(result['top_states'])} top computational states")
    
    # Small parameter sweep example
    d0_vals = [7.0, 8.0]
    omega_vals = [6.0, 10.0]
    delta_vals = [25.0, 35.0]
    
    sweep_result = computational_parameter_scan(d0_vals, omega_vals, delta_vals, runtime=2.0)
    print(f"Completed sweep with {len(sweep_result['results'])} parameter combinations")