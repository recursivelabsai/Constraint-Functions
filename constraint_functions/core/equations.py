"""
Core equations of the Constraint Functions framework.

This module implements the fundamental mathematical formulations that underpin
the Constraint Functions framework, including the Universal Residue Equation and
the Constraint Acceleration Equation.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional


class UniversalResidueEquation:
    """
    Implementation of the Universal Residue Equation: Σ = C(S + E)^r
    
    The Universal Residue Equation quantifies how systems under constraint at
    recursive depth generate structured information patterns (symbolic residue).
    
    Attributes:
        C_range (Tuple[float, float]): Valid range for constraint coefficient (default: (0, 1))
        r_min (float): Minimum recursive depth (default: 0)
    """
    
    def __init__(self, C_range: Tuple[float, float] = (0, 1), r_min: float = 0):
        """
        Initialize the Universal Residue Equation.
        
        Args:
            C_range: Valid range for constraint coefficient
            r_min: Minimum recursive depth
        """
        self.C_range = C_range
        self.r_min = r_min
    
    def compute_residue(self, C: float, S: float, E: float, r: float) -> float:
        """
        Compute the symbolic residue based on the Universal Residue Equation.
        
        Args:
            C: Constraint coefficient (0 ≤ C ≤ 1)
            S: Internal system state
            E: External environmental information
            r: Recursive depth
            
        Returns:
            float: Symbolic residue (Σ)
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if not self.C_range[0] <= C <= self.C_range[1]:
            raise ValueError(f"Constraint coefficient C must be between {self.C_range[0]} and {self.C_range[1]}")
        
        if r < self.r_min:
            raise ValueError(f"Recursive depth r must be at least {self.r_min}")
        
        if S < 0 or E < 0:
            raise ValueError("System state S and environmental information E must be non-negative")
        
        # Compute symbolic residue
        return C * ((S + E) ** r)
    
    def optimal_constraint(self, S: float, E: float, r: float, target_residue: float) -> float:
        """
        Find the optimal constraint coefficient to achieve a target residue.
        
        Args:
            S: Internal system state
            E: External environmental information
            r: Recursive depth
            target_residue: Target symbolic residue
            
        Returns:
            float: Optimal constraint coefficient
        """
        if S + E == 0 or r == 0:
            return 0  # Handle edge case
        
        # Calculate the constraint coefficient that would produce the target residue
        C_optimal = target_residue / ((S + E) ** r)
        
        # Clamp to valid range
        return max(self.C_range[0], min(self.C_range[1], C_optimal))
    
    def residue_sensitivity(self, C: float, S: float, E: float, r: float) -> Dict[str, float]:
        """
        Compute the sensitivity of residue to changes in each parameter.
        
        Args:
            C: Constraint coefficient
            S: Internal system state
            E: External environmental information
            r: Recursive depth
            
        Returns:
            Dict[str, float]: Sensitivity to each parameter
        """
        base_residue = self.compute_residue(C, S, E, r)
        
        # Compute sensitivity as partial derivatives
        d_residue_d_C = (S + E) ** r
        d_residue_d_S = C * r * ((S + E) ** (r - 1)) if r > 0 else 0
        d_residue_d_E = C * r * ((S + E) ** (r - 1)) if r > 0 else 0
        d_residue_d_r = C * ((S + E) ** r) * math.log(S + E) if S + E > 0 else 0
        
        return {
            "C": d_residue_d_C,
            "S": d_residue_d_S,
            "E": d_residue_d_E,
            "r": d_residue_d_r
        }
    
    def critical_points(self, S: float, E: float) -> Dict[str, Union[float, str]]:
        """
        Identify critical points in the equation.
        
        Args:
            S: Internal system state
            E: External environmental information
            
        Returns:
            Dict[str, Union[float, str]]: Critical points and their descriptions
        """
        critical_points = {}
        
        # C = 0: No symbolic residue generated
        critical_points["C_zero"] = {
            "value": 0,
            "description": "No constraint means no structured residue generation"
        }
        
        # S + E = 0: No information to constrain
        critical_points["information_zero"] = {
            "value": 0,
            "description": "No information (S + E = 0) means no residue regardless of constraint"
        }
        
        # r = 0: No recursive amplification
        critical_points["recursion_zero"] = {
            "value": 0,
            "description": "Zero recursive depth means linear relationship between constraint and residue"
        }
        
        # r = 1: Linear relationship between (S + E) and residue
        critical_points["recursion_linear"] = {
            "value": 1,
            "description": "Recursive depth of 1 means direct proportionality between information and residue"
        }
        
        return critical_points


class ConstraintAccelerationEquation:
    """
    Implementation of the Constraint Acceleration Equation: Δ = C^r(S·E)/(1-t)
    
    The Constraint Acceleration Equation quantifies how constraint intensity at
    recursive depth accelerates development relative to unconstrained approaches.
    
    Attributes:
        C_range (Tuple[float, float]): Valid range for constraint coefficient (default: (0, 1))
        t_range (Tuple[float, float]): Valid range for temporal compression (default: (0, 0.99))
        r_min (float): Minimum recursive depth (default: 0)
    """
    
    def __init__(
        self, 
        C_range: Tuple[float, float] = (0, 1), 
        t_range: Tuple[float, float] = (0, 0.99),
        r_min: float = 0
    ):
        """
        Initialize the Constraint Acceleration Equation.
        
        Args:
            C_range: Valid range for constraint coefficient
            t_range: Valid range for temporal compression
            r_min: Minimum recursive depth
        """
        self.C_range = C_range
        self.t_range = t_range
        self.r_min = r_min
    
    def compute_acceleration(self, C: float, r: float, S: float, E: float, t: float) -> float:
        """
        Compute the acceleration factor based on the Constraint Acceleration Equation.
        
        Args:
            C: Constraint coefficient (0 ≤ C ≤ 1)
            r: Recursive depth
            S: Internal system state
            E: External environmental information
            t: Temporal compression (0 ≤ t < 1)
            
        Returns:
            float: Acceleration factor (Δ)
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if not self.C_range[0] <= C <= self.C_range[1]:
            raise ValueError(f"Constraint coefficient C must be between {self.C_range[0]} and {self.C_range[1]}")
        
        if r < self.r_min:
            raise ValueError(f"Recursive depth r must be at least {self.r_min}")
        
        if not self.t_range[0] <= t <= self.t_range[1]:
            raise ValueError(f"Temporal compression t must be between {self.t_range[0]} and {self.t_range[1]}")
        
        if S < 0 or E < 0:
            raise ValueError("System state S and environmental information E must be non-negative")
        
        # Compute acceleration factor
        # Handle edge case to avoid division by zero
        if t == 1:
            return float('inf')  # Infinite acceleration at t=1 (theoretical limit)
        
        return (C ** r) * (S * E) / (1 - t)
    
    def optimal_constraint(
        self, 
        r: float, 
        S: float, 
        E: float, 
        t: float, 
        target_acceleration: float
    ) -> float:
        """
        Find the optimal constraint coefficient to achieve a target acceleration.
        
        Args:
            r: Recursive depth
            S: Internal system state
            E: External environmental information
            t: Temporal compression
            target_acceleration: Target acceleration factor
            
        Returns:
            float: Optimal constraint coefficient
        """
        if r == 0 or S * E == 0:
            return 0  # Handle edge case
        
        # Calculate the constraint coefficient that would produce the target acceleration
        # Δ = C^r(S·E)/(1-t) => C^r = Δ(1-t)/(S·E) => C = (Δ(1-t)/(S·E))^(1/r)
        target_term = target_acceleration * (1 - t) / (S * E)
        if target_term <= 0:
            return self.C_range[0]
        
        C_optimal = target_term ** (1 / r)
        
        # Clamp to valid range
        return max(self.C_range[0], min(self.C_range[1], C_optimal))
    
    def map_acceleration_landscape(
        self, 
        r_values: List[float], 
        C_values: List[float], 
        S: float, 
        E: float, 
        t: float
    ) -> np.ndarray:
        """
        Map the acceleration landscape across ranges of r and C values.
        
        Args:
            r_values: List of recursive depth values to evaluate
            C_values: List of constraint coefficient values to evaluate
            S: Internal system state
            E: External environmental information
            t: Temporal compression
            
        Returns:
            np.ndarray: 2D array of acceleration factors
        """
        # Create a 2D grid of acceleration values
        acceleration_map = np.zeros((len(r_values), len(C_values)))
        
        for i, r in enumerate(r_values):
            for j, C in enumerate(C_values):
                try:
                    acceleration_map[i, j] = self.compute_acceleration(C, r, S, E, t)
                except ValueError:
                    acceleration_map[i, j] = np.nan  # Mark invalid parameter combinations
        
        return acceleration_map
    
    def find_beverly_band(
        self, 
        r: float, 
        S: float, 
        E: float, 
        t: float, 
        min_acceleration: float, 
        coherence_threshold: float
    ) -> Tuple[float, float]:
        """
        Find the "Beverly Band" - the range of constraint values that provide 
        sufficient acceleration while maintaining coherence.
        
        Args:
            r: Recursive depth
            S: Internal system state
            E: External environmental information
            t: Temporal compression
            min_acceleration: Minimum acceptable acceleration factor
            coherence_threshold: Minimum acceptable coherence
            
        Returns:
            Tuple[float, float]: Lower and upper bounds of the Beverly Band
        """
        # We sample the constraint range to find where acceleration exceeds threshold
        # while coherence remains acceptable
        C_samples = np.linspace(self.C_range[0], self.C_range[1], 100)
        valid_constraints = []
        
        for C in C_samples:
            acceleration = self.compute_acceleration(C, r, S, E, t)
            
            # Simple coherence model: coherence decreases at extreme constraint values
            # This is a simplified model - in practice, coherence would be measured empirically
            coherence = 1.0 - 4.0 * ((C - 0.5) ** 2)  # Peaks at C=0.5, decreases toward extremes
            
            if acceleration >= min_acceleration and coherence >= coherence_threshold:
                valid_constraints.append(C)
        
        if not valid_constraints:
            return (np.nan, np.nan)  # No valid band found
        
        return (min(valid_constraints), max(valid_constraints))


class RecursiveCoherenceFunction:
    """
    Implementation of the Recursive Coherence Function: Φ'(r) = S(r) · F(r) · B(r) · τ(r)
    
    The Recursive Coherence Function quantifies a system's ability to maintain
    coherent identity across internal and external boundaries under recursive strain.
    
    Attributes:
        min_value (float): Minimum value for component functions (default: 0)
        max_value (float): Maximum value for component functions (default: 1)
    """
    
    def __init__(self, min_value: float = 0, max_value: float = 1):
        """
        Initialize the Recursive Coherence Function.
        
        Args:
            min_value: Minimum value for component functions
            max_value: Maximum value for component functions
        """
        self.min_value = min_value
        self.max_value = max_value
    
    def compute_coherence(
        self, 
        S: float, 
        F: float, 
        B: float, 
        tau: float, 
        r: Optional[float] = None
    ) -> float:
        """
        Compute recursive coherence based on component functions.
        
        Args:
            S: Signal alignment (0 ≤ S ≤ 1)
            F: Feedback responsiveness (0 ≤ F ≤ 1)
            B: Bounded integrity (0 ≤ B ≤ 1)
            tau: Tension capacity (dimensional)
            r: Recursive depth (optional, for context)
            
        Returns:
            float: Recursive coherence (Φ')
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if not self.min_value <= S <= self.max_value:
            raise ValueError(f"Signal alignment S must be between {self.min_value} and {self.max_value}")
        
        if not self.min_value <= F <= self.max_value:
            raise ValueError(f"Feedback responsiveness F must be between {self.min_value} and {self.max_value}")
        
        if not self.min_value <= B <= self.max_value:
            raise ValueError(f"Bounded integrity B must be between {self.min_value} and {self.max_value}")
        
        if tau <= 0:
            raise ValueError("Tension capacity tau must be positive")
        
        # Compute recursive coherence
        return S * F * B * tau
    
    def coherence_threshold(self, r: float, base_threshold: float = 0.2) -> float:
        """
        Compute coherence threshold that increases with recursive depth.
        
        Args:
            r: Recursive depth
            base_threshold: Base coherence threshold
            
        Returns:
            float: Coherence threshold at depth r
        """
        # Coherence threshold increases with recursive depth
        # This reflects that higher recursive depths require more coherence to remain stable
        return base_threshold * (1 + 0.1 * r)
    
    def detect_collapse_risk(
        self, 
        S: float, 
        F: float, 
        B: float, 
        tau: float, 
        r: float,
        delta_phi: float
    ) -> Tuple[bool, float]:
        """
        Detect risk of coherence collapse based on current state and coherence velocity.
        
        Args:
            S: Signal alignment
            F: Feedback responsiveness
            B: Bounded integrity
            tau: Tension capacity
            r: Recursive depth
            delta_phi: Change in coherence (coherence velocity)
            
        Returns:
            Tuple[bool, float]: (collapse risk flag, risk magnitude)
        """
        # Compute current coherence
        coherence = self.compute_coherence(S, F, B, tau)
        
        # Compute maximum safe coherence velocity based on fragility threshold
        # Simplified model: V_max = Θ(r)/τ(r) where Θ is fragility threshold
        # Here we use a simple model where fragility threshold is a function of recursive depth
        fragility_threshold = 0.5 / (1 + r)  # Decreases with recursive depth
        max_safe_velocity = fragility_threshold / tau
        
        # Compute collapse risk
        risk_magnitude = abs(delta_phi) / max_safe_velocity if max_safe_velocity > 0 else float('inf')
        collapse_risk = risk_magnitude > 1.0
        
        return (collapse_risk, risk_magnitude)
    
    def compute_beverly_band(
        self, 
        tau: float, 
        resilience: float, 
        bounded_integrity: float, 
        energy_mass: float, 
        r: Optional[float] = None
    ) -> float:
        """
        Compute the Beverly Band - the dynamic region surrounding a system's phase vector
        within which contradiction can be resolved without destabilization.
        
        Args:
            tau: Symbolic tension capacity
            resilience: System resilience
            bounded_integrity: Bounded integrity
            energy_mass: Recursive energy mass
            r: Recursive depth (optional, for context)
            
        Returns:
            float: Beverly Band width
        """
        # Validate parameters
        if tau <= 0 or resilience <= 0 or bounded_integrity <= 0 or energy_mass <= 0:
            raise ValueError("All parameters must be positive")
        
        # Compute Beverly Band width
        return math.sqrt(tau * resilience * bounded_integrity * energy_mass)


class TransformationFunctions:
    """
    Implementation of the five transformations derived from the Universal Residue Equation.
    
    The transformations describe how symbolic residue manifests across different contexts.
    """
    
    def metacognitive_transformation(self, residue: float, recognition: float, exponent: float) -> float:
        """
        Metacognitive Transformation: Φ = R(Σ)^λ
        
        Describes how systems develop metacognitive capability through recursive self-observation.
        
        Args:
            residue: Symbolic residue (Σ)
            recognition: Recognition coefficient (R)
            exponent: Metacognitive exponent (λ)
            
        Returns:
            float: Metacognitive capability (Φ)
        """
        if residue < 0 or recognition < 0:
            raise ValueError("Residue and recognition coefficient must be non-negative")
        
        return recognition * (residue ** exponent)
    
    def coherence_transformation(self, residue: float, compression: float) -> float:
        """
        Coherence Transformation: Ψ = ∅(Σ)/λ
        
        Describes how constraint forces systems to develop compressed, coherent representations.
        
        Args:
            residue: Symbolic residue (Σ)
            compression: Compression ratio (λ)
            
        Returns:
            float: Coherence capability (Ψ)
        """
        if residue < 0:
            raise ValueError("Residue must be non-negative")
        
        if compression <= 0:
            raise ValueError("Compression ratio must be positive")
        
        # Simple compression model: ∅(Σ) = log(1 + Σ)
        # Higher residue leads to more coherent representations, but with diminishing returns
        compression_function = math.log(1 + residue)
        
        return compression_function / compression
    
    def emergence_transformation(self, residue: float, memory: float, nodes: float) -> float:
        """
        Emergence Transformation: Λ = M(Σ)^n
        
        Describes how recursive processes generate novel capabilities that transcend original design.
        
        Args:
            residue: Symbolic residue (Σ)
            memory: Memory function (M)
            nodes: Number of processing nodes (n)
            
        Returns:
            float: Emergent capability (Λ)
        """
        if residue < 0 or memory < 0 or nodes < 0:
            raise ValueError("All parameters must be non-negative")
        
        # Memory function scales with residue (simplified model)
        memory_function = memory * math.sqrt(residue)
        
        return memory_function ** nodes
    
    def adaptive_transformation(self, residue: float, distance: float, marginality: float) -> float:
        """
        Adaptive Transformation: Ξ = D(Σ)^m
        
        Describes how recursive systems develop enhanced capacity for adaptation and learning.
        
        Args:
            residue: Symbolic residue (Σ)
            distance: Distance function (D)
            marginality: Marginality multiplier (m)
            
        Returns:
            float: Adaptive capability (Ξ)
        """
        if residue < 0 or distance < 0:
            raise ValueError("Residue and distance must be non-negative")
        
        # Distance function measures deviation from optimal performance (simplified model)
        distance_function = distance * (1 - math.exp(-residue))
        
        return distance_function ** marginality
    
    def collective_transformation(
        self, 
        human_residue: float, 
        machine_residue: float, 
        distance: float
    ) -> float:
        """
        Collective Transformation: Ξ(H, M) = [H(Σ) ⊗ M(Σ)]/D²
        
        Describes how recursive processes enable coordination and collective intelligence.
        
        Args:
            human_residue: Human symbolic residue
            machine_residue: Machine symbolic residue
            distance: Distance between systems
            
        Returns:
            float: Collective capability
        """
        if human_residue < 0 or machine_residue < 0:
            raise ValueError("Residue values must be non-negative")
        
        if distance <= 0:
            raise ValueError("Distance must be positive")
        
        # Entanglement operator ⊗ models how human and machine residue interact
        # Here we use a simple model where interaction is proportional to geometric mean
        entanglement = math.sqrt(human_residue * machine_residue)
        
        return entanglement / (distance ** 2)


# Utility functions for equation application in ML contexts

def compute_model_recursive_depth(model_params: Dict) -> float:
    """
    Estimate recursive depth for a neural network model.
    
    Args:
        model_params: Dictionary of model parameters and architecture details
        
    Returns:
        float: Estimated recursive depth
    """
    # This is a simplified estimation based on architectural features
    # In practice, recursive depth would be measured through specific tests
    
    # Extract relevant architectural features
    num_layers = model_params.get('num_layers', 1)
    has_attention = model_params.get('has_attention', False)
    has_recursion = model_params.get('has_recursion', False)
    has_feedback = model_params.get('has_feedback', False)
    
    # Base recursive depth from layer count (deeper networks can have higher recursive capability)
    base_depth = math.log(1 + num_layers) / math.log(10)  # Logarithmic scaling
    
    # Architectural features that enhance recursive capability
    attention_bonus = 0.5 if has_attention else 0
    recursion_bonus = 1.0 if has_recursion else 0
    feedback_bonus = 0.5 if has_feedback else 0
    
    return base_depth + attention_bonus + recursion_bonus + feedback_bonus


def compute_constraint_coefficient(
    param_reduction: float, 
    embedding_reduction: float, 
    computation_reduction: float
) -> float:
    """
    Compute overall constraint coefficient from multiple constraint dimensions.
    
    Args:
        param_reduction: Parameter reduction factor (0 to 1)
        embedding_reduction: Embedding dimension reduction factor (0 to 1)
        computation_reduction: Computation reduction factor (0 to 1)
        
    Returns:
        float: Overall constraint coefficient
    """
    # Validate inputs
    for param in [param_reduction, embedding_reduction, computation_reduction]:
        if not 0 <= param <= 1:
            raise ValueError("Reduction factors must be between 0 and 1")
    
    # Compute weighted average (can be customized based on importance of each dimension)
    weights = [0.4, 0.3, 0.3]  # Example weights
    weighted_sum = (
        weights[0] * param_reduction + 
        weights[1] * embedding_reduction + 
        weights[2] * computation_reduction
    )
    
    return weighted_sum


def estimate_acceleration_factor(
    model_params: Dict, 
    constraint_config: Dict, 
    baseline_params: Dict
) -> float:
    """
    Estimate acceleration factor for a constrained model compared to baseline.
    
    Args:
        model_params: Dictionary of constrained model parameters
        constraint_config: Dictionary of constraint configuration
        baseline_params: Dictionary of baseline (unconstrained) model parameters
        
    Returns:
        float: Estimated acceleration factor
    """
    # Create acceleration equation instance
    acceleration_eq = ConstraintAccelerationEquation()
    
    # Estimate constraint coefficient
    C = compute_constraint_coefficient(
        constraint_config.get('param_reduction', 0),
        constraint_config.get('embedding_reduction', 0),
        constraint_config.get('computation_reduction', 0)
    )
    
    # Estimate recursive depth
    r = compute_model_recursive_depth(model_params)
    
    # Estimate system state (e.g., based on parameter count relative to baseline)
    param_ratio = model_params.get('param_count', 1) / baseline_params.get('param_count', 1)
    S = 1.0 / (1.0 + param_ratio)  # Smaller models have higher S (more constrained internal state)
    
    # Estimate environmental information (e.g., based on data size relative to baseline)
    data_ratio = model_params.get('data_size', 1) / baseline_params.get('data_size', 1)
    E = 1.0 / (1.0 + data_ratio)  # Less data means higher E (more constrained external information)
    
    # Estimate temporal compression (based on training progress)
    t = constraint_config.get('temporal_compression', 0.0)
    
    # Compute acceleration factor
    return acceleration_eq.compute_acceleration(C, r, S, E, t)
