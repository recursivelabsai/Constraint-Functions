"""
Constraint types and their properties in the Constraint Functions framework.

This module defines the various types of constraints that can be applied to AI systems,
their properties, and how they interact to generate specific capabilities.
"""

from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Callable
import numpy as np
import math


class ConstraintDimension(Enum):
    """Enumeration of fundamental constraint dimensions."""
    
    COMPUTATIONAL = "computational"  # Constraints on computational resources
    REPRESENTATIONAL = "representational"  # Constraints on representation capacity
    TEMPORAL = "temporal"  # Constraints on processing time or sequence length
    KNOWLEDGE = "knowledge"  # Constraints on available information
    FEEDBACK = "feedback"  # Constraints on feedback specificity or frequency
    ACTION = "action"  # Constraints on available actions
    ARCHITECTURAL = "architectural"  # Constraints on network architecture


class ConstraintType:
    """
    Base class for defining specific constraint types.
    
    Attributes:
        name (str): Name of the constraint type
        dimension (ConstraintDimension): Primary constraint dimension
        intensity_range (Tuple[float, float]): Valid range for constraint intensity
        description (str): Description of the constraint type
        acceleration_profile (str): Characteristic acceleration pattern
    """
    
    def __init__(
        self,
        name: str,
        dimension: ConstraintDimension,
        intensity_range: Tuple[float, float] = (0.0, 1.0),
        description: str = "",
        acceleration_profile: str = "linear"
    ):
        """
        Initialize a constraint type.
        
        Args:
            name: Name of the constraint type
            dimension: Primary constraint dimension
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint type
            acceleration_profile: Characteristic acceleration pattern
        """
        self.name = name
        self.dimension = dimension
        self.intensity_range = intensity_range
        self.description = description
        self.acceleration_profile = acceleration_profile
        
        # Optimal intensity range (where acceleration is maximized)
        # Default to middle of range, will be refined based on empirical data
        self._optimal_intensity = (intensity_range[0] + intensity_range[1]) / 2
        self._optimal_range = (
            max(intensity_range[0], self._optimal_intensity - 0.1),
            min(intensity_range[1], self._optimal_intensity + 0.1)
        )
    
    def validate_intensity(self, intensity: float) -> bool:
        """
        Validate that the given intensity is within the allowed range.
        
        Args:
            intensity: Constraint intensity value
            
        Returns:
            bool: True if valid, False otherwise
        """
        return self.intensity_range[0] <= intensity <= self.intensity_range[1]
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply the constraint to a target object.
        
        Args:
            target: Object to apply constraint to
            intensity: Constraint intensity
            **kwargs: Additional constraint-specific parameters
            
        Returns:
            object: Constrained object
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement apply()")
    
    def compute_acceleration_factor(
        self, 
        intensity: float, 
        recursive_depth: float, 
        base_factor: float = 1.0
    ) -> float:
        """
        Compute the acceleration factor for this constraint type at given intensity.
        
        Args:
            intensity: Constraint intensity
            recursive_depth: Recursive depth of the system
            base_factor: Base acceleration factor
    """
Constraint types and their properties in the Constraint Functions framework.

This module defines the various types of constraints that can be applied to AI systems,
their properties, and how they interact to generate specific capabilities.
"""

from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Callable
import numpy as np
import math


class ConstraintDimension(Enum):
    """Enumeration of fundamental constraint dimensions."""
    
    COMPUTATIONAL = "computational"  # Constraints on computational resources
    REPRESENTATIONAL = "representational"  # Constraints on representation capacity
    TEMPORAL = "temporal"  # Constraints on processing time or sequence length
    KNOWLEDGE = "knowledge"  # Constraints on available information
    FEEDBACK = "feedback"  # Constraints on feedback specificity or frequency
    ACTION = "action"  # Constraints on available actions
    ARCHITECTURAL = "architectural"  # Constraints on network architecture


class ConstraintType:
    """
    Base class for defining specific constraint types.
    
    Attributes:
        name (str): Name of the constraint type
        dimension (ConstraintDimension): Primary constraint dimension
        intensity_range (Tuple[float, float]): Valid range for constraint intensity
        description (str): Description of the constraint type
        acceleration_profile (str): Characteristic acceleration pattern
    """
    
    def __init__(
        self,
        name: str,
        dimension: ConstraintDimension,
        intensity_range: Tuple[float, float] = (0.0, 1.0),
        description: str = "",
        acceleration_profile: str = "linear"
    ):
        """
        Initialize a constraint type.
        
        Args:
            name: Name of the constraint type
            dimension: Primary constraint dimension
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint type
            acceleration_profile: Characteristic acceleration pattern
        """
        self.name = name
        self.dimension = dimension
        self.intensity_range = intensity_range
        self.description = description
        self.acceleration_profile = acceleration_profile
        
        # Optimal intensity range (where acceleration is maximized)
        # Default to middle of range, will be refined based on empirical data
        self._optimal_intensity = (intensity_range[0] + intensity_range[1]) / 2
        self._optimal_range = (
            max(intensity_range[0], self._optimal_intensity - 0.1),
            min(intensity_range[1], self._optimal_intensity + 0.1)
        )
    
    def validate_intensity(self, intensity: float) -> bool:
        """
        Validate that the given intensity is within the allowed range.
        
        Args:
            intensity: Constraint intensity value
            
        Returns:
            bool: True if valid, False otherwise
        """
        return self.intensity_range[0] <= intensity <= self.intensity_range[1]
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply the constraint to a target object.
        
        Args:
            target: Object to apply constraint to
            intensity: Constraint intensity
            **kwargs: Additional constraint-specific parameters
            
        Returns:
            object: Constrained object
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement apply()")
    
    def compute_acceleration_factor(
        self, 
        intensity: float, 
        recursive_depth: float, 
        base_factor: float = 1.0
    ) -> float:
        """
        Compute the acceleration factor for this constraint type at given intensity.
        
        Args:
            intensity: Constraint intensity
            recursive_depth: Recursive depth of the system
            base_factor: Base acceleration factor
            
        Returns:
            float: Acceleration factor
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Different acceleration profiles based on constraint type
        if self.acceleration_profile == "linear":
            # Linear acceleration with intensity
            return base_factor * (1 + intensity * recursive_depth)
        
        elif self.acceleration_profile == "quadratic":
            # Quadratic acceleration with intensity
            return base_factor * (1 + (intensity ** 2) * recursive_depth)
        
        elif self.acceleration_profile == "exponential":
            # Exponential acceleration with intensity
            return base_factor * (1 + math.exp(intensity * recursive_depth) - 1)
        
        elif self.acceleration_profile == "inverted_u":
            # Inverted U-shaped curve with peak at optimal intensity
            distance_from_optimal = abs(intensity - self._optimal_intensity)
            return base_factor * (1 + (1 - 4 * (distance_from_optimal ** 2)) * recursive_depth)
        
        else:
            # Default to linear if unknown profile
            return base_factor * (1 + intensity * recursive_depth)
    
    def get_optimal_intensity(self, recursive_depth: float = 1.0) -> float:
        """
        Get the optimal constraint intensity for maximum acceleration.
        
        Args:
            recursive_depth: Recursive depth of the system
            
        Returns:
            float: Optimal constraint intensity
        """
        # For most constraint types, optimal intensity is fixed
        # For some types, it may vary with recursive depth
        return self._optimal_intensity
    
    def is_compatible_with(self, other_constraint: 'ConstraintType') -> bool:
        """
        Check if this constraint type is compatible with another.
        
        Args:
            other_constraint: Another constraint type to check compatibility with
            
        Returns:
            bool: True if compatible, False otherwise
        """
        # Default implementation assumes constraints are compatible
        # Subclasses can override for specific incompatibilities
        return True
    
    def __str__(self) -> str:
        """String representation of the constraint type."""
        return f"{self.name} ({self.dimension.value})"
    
    def __repr__(self) -> str:
        """Detailed representation of the constraint type."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"dimension={self.dimension}, "
                f"intensity_range={self.intensity_range}, "
                f"acceleration_profile='{self.acceleration_profile}')")


class ComputationalConstraint(ConstraintType):
    """
    Constraints on computational resources (e.g., parameter count, FLOPs).
    
    Computational constraints force systems to develop more efficient algorithms
    and representations by limiting raw computational capacity.
    """
    
    def __init__(
        self,
        name: str = "Computational Constraint",
        intensity_range: Tuple[float, float] = (0.0, 0.9),
        description: str = "Constraints on computational resources",
        acceleration_profile: str = "inverted_u"
    ):
        """
        Initialize a computational constraint.
        
        Args:
            name: Name of the constraint
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint
            acceleration_profile: Characteristic acceleration pattern
        """
        super().__init__(
            name=name,
            dimension=ConstraintDimension.COMPUTATIONAL,
            intensity_range=intensity_range,
            description=description,
            acceleration_profile=acceleration_profile
        )
        
        # Computational constraints have optimal value around 0.5-0.7
        # (moderate constraint drives innovation, excessive constraint prevents progress)
        self._optimal_intensity = 0.6
        self._optimal_range = (0.5, 0.7)
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply computational constraint to a target object.
        
        Args:
            target: Object to apply constraint to (e.g., model, layer)
            intensity: Constraint intensity (0 to 1)
            **kwargs: Additional parameters:
                - param_reduction_fn: Function to reduce parameters
                - flop_reduction_fn: Function to reduce FLOPs
                - memory_reduction_fn: Function to reduce memory usage
                
        Returns:
            object: Computationally constrained object
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Extract constraint functions from kwargs or use defaults
        param_reduction_fn = kwargs.get('param_reduction_fn', None)
        flop_reduction_fn = kwargs.get('flop_reduction_fn', None)
        memory_reduction_fn = kwargs.get('memory_reduction_fn', None)
        
        # Create constrained copy of target
        constrained_target = target
        
        # Apply parameter reduction if function provided
        if param_reduction_fn:
            constrained_target = param_reduction_fn(constrained_target, intensity)
        
        # Apply FLOP reduction if function provided
        if flop_reduction_fn:
            constrained_target = flop_reduction_fn(constrained_target, intensity)
        
        # Apply memory reduction if function provided
        if memory_reduction_fn:
            constrained_target = memory_reduction_fn(constrained_target, intensity)
        
        return constrained_target
    
    def get_optimal_intensity(self, recursive_depth: float = 1.0) -> float:
        """
        Get optimal computational constraint intensity based on recursive depth.
        
        For computational constraints, optimal intensity increases with recursive depth
        as more recursive systems can handle stronger constraints effectively.
        
        Args:
            recursive_depth: Recursive depth of the system
            
        Returns:
            float: Optimal constraint intensity
        """
        # Increase optimal intensity with recursive depth, but cap at 0.8
        base_optimal = self._optimal_intensity
        depth_adjustment = min(0.2, 0.05 * (recursive_depth - 1))  # +0.05 per recursion level above 1
        
        return min(0.8, base_optimal + (depth_adjustment if recursive_depth > 1 else 0))


class RepresentationalConstraint(ConstraintType):
    """
    Constraints on representation capacity (e.g., embedding dimensions, attention heads).
    
    Representational constraints force systems to develop more efficient encodings
    by limiting the dimensions of internal representations.
    """
    
    def __init__(
        self,
        name: str = "Representational Constraint",
        intensity_range: Tuple[float, float] = (0.0, 0.8),
        description: str = "Constraints on representation capacity",
        acceleration_profile: str = "inverted_u"
    ):
        """
        Initialize a representational constraint.
        
        Args:
            name: Name of the constraint
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint
            acceleration_profile: Characteristic acceleration pattern
        """
        super().__init__(
            name=name,
            dimension=ConstraintDimension.REPRESENTATIONAL,
            intensity_range=intensity_range,
            description=description,
            acceleration_profile=acceleration_profile
        )
        
        # Representational constraints have optimal value around 0.4-0.6
        self._optimal_intensity = 0.5
        self._optimal_range = (0.4, 0.6)
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply representational constraint to a target object.
        
        Args:
            target: Object to apply constraint to (e.g., model, embeddings)
            intensity: Constraint intensity (0 to 1)
            **kwargs: Additional parameters:
                - embedding_reduction_fn: Function to reduce embedding dimensions
                - attention_reduction_fn: Function to reduce attention mechanisms
                - hidden_reduction_fn: Function to reduce hidden dimensions
                
        Returns:
            object: Representationally constrained object
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Extract constraint functions from kwargs or use defaults
        embedding_reduction_fn = kwargs.get('embedding_reduction_fn', None)
        attention_reduction_fn = kwargs.get('attention_reduction_fn', None)
        hidden_reduction_fn = kwargs.get('hidden_reduction_fn', None)
        
        # Create constrained copy of target
        constrained_target = target
        
        # Apply embedding dimension reduction if function provided
        if embedding_reduction_fn:
            constrained_target = embedding_reduction_fn(constrained_target, intensity)
        
        # Apply attention mechanism reduction if function provided
        if attention_reduction_fn:
            constrained_target = attention_reduction_fn(constrained_target, intensity)
        
        # Apply hidden dimension reduction if function provided
        if hidden_reduction_fn:
            constrained_target = hidden_reduction_fn(constrained_target, intensity)
        
        return constrained_target


class TemporalConstraint(ConstraintType):
    """
    Constraints on processing time or sequence length.
    
    Temporal constraints force systems to develop more efficient processing by
    limiting the time or sequence length available for computation.
    """
    
    def __init__(
        self,
        name: str = "Temporal Constraint",
        intensity_range: Tuple[float, float] = (0.0, 0.7),
        description: str = "Constraints on processing time or sequence length",
        acceleration_profile: str = "quadratic"
    ):
        """
        Initialize a temporal constraint.
        
        Args:
            name: Name of the constraint
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint
            acceleration_profile: Characteristic acceleration pattern
        """
        super().__init__(
            name=name,
            dimension=ConstraintDimension.TEMPORAL,
            intensity_range=intensity_range,
            description=description,
            acceleration_profile=acceleration_profile
        )
        
        # Temporal constraints have optimal value around 0.3-0.5
        # (too high causes information loss rather than compression)
        self._optimal_intensity = 0.4
        self._optimal_range = (0.3, 0.5)
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply temporal constraint to a target object.
        
        Args:
            target: Object to apply constraint to (e.g., model, sequence)
            intensity: Constraint intensity (0 to 1)
            **kwargs: Additional parameters:
                - context_length_fn: Function to reduce context window
                - processing_steps_fn: Function to reduce processing steps
                - sequence_truncation_fn: Function to truncate sequences
                
        Returns:
            object: Temporally constrained object
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Extract constraint functions from kwargs or use defaults
        context_length_fn = kwargs.get('context_length_fn', None)
        processing_steps_fn = kwargs.get('processing_steps_fn', None)
        sequence_truncation_fn = kwargs.get('sequence_truncation_fn', None)
        
        # Create constrained copy of target
        constrained_target = target
        
        # Apply context length reduction if function provided
        if context_length_fn:
            constrained_target = context_length_fn(constrained_target, intensity)
        
        # Apply processing steps reduction if function provided
        if processing_steps_fn:
            constrained_target = processing_steps_fn(constrained_target, intensity)
        
        # Apply sequence truncation if function provided
        if sequence_truncation_fn:
            constrained_target = sequence_truncation_fn(constrained_target, intensity)
        
        return constrained_target


class KnowledgeConstraint(ConstraintType):
    """
    Constraints on available information or knowledge.
    
    Knowledge constraints force systems to develop more robust generalization
    by limiting the information available during training or inference.
    """
    
    def __init__(
        self,
        name: str = "Knowledge Constraint",
        intensity_range: Tuple[float, float] = (0.0, 0.8),
        description: str = "Constraints on available information",
        acceleration_profile: str = "exponential"
    ):
        """
        Initialize a knowledge constraint.
        
        Args:
            name: Name of the constraint
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint
            acceleration_profile: Characteristic acceleration pattern
        """
        super().__init__(
            name=name,
            dimension=ConstraintDimension.KNOWLEDGE,
            intensity_range=intensity_range,
            description=description,
            acceleration_profile=acceleration_profile
        )
        
        # Knowledge constraints have optimal value around 0.3-0.5
        # (moderate data limitation drives generalization, excessive prevents learning)
        self._optimal_intensity = 0.4
        self._optimal_range = (0.3, 0.5)
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply knowledge constraint to a target object.
        
        Args:
            target: Object to apply constraint to (e.g., dataset, inputs)
            intensity: Constraint intensity (0 to 1)
            **kwargs: Additional parameters:
                - data_reduction_fn: Function to reduce data amount
                - feature_masking_fn: Function to mask features
                - label_sparsity_fn: Function to create label sparsity
                
        Returns:
            object: Knowledge-constrained object
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Extract constraint functions from kwargs or use defaults
        data_reduction_fn = kwargs.get('data_reduction_fn', None)
        feature_masking_fn = kwargs.get('feature_masking_fn', None)
        label_sparsity_fn = kwargs.get('label_sparsity_fn', None)
        
        # Create constrained copy of target
        constrained_target = target
        
        # Apply data reduction if function provided
        if data_reduction_fn:
            constrained_target = data_reduction_fn(constrained_target, intensity)
        
        # Apply feature masking if function provided
        if feature_masking_fn:
            constrained_target = feature_masking_fn(constrained_target, intensity)
        
        # Apply label sparsity if function provided
        if label_sparsity_fn:
            constrained_target = label_sparsity_fn(constrained_target, intensity)
        
        return constrained_target


class FeedbackConstraint(ConstraintType):
    """
    Constraints on feedback specificity or frequency.
    
    Feedback constraints force systems to develop more robust self-evaluation
    by limiting the specificity or frequency of external feedback.
    """
    
    def __init__(
        self,
        name: str = "Feedback Constraint",
        intensity_range: Tuple[float, float] = (0.0, 0.9),
        description: str = "Constraints on feedback specificity or frequency",
        acceleration_profile: str = "exponential"
    ):
        """
        Initialize a feedback constraint.
        
        Args:
            name: Name of the constraint
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint
            acceleration_profile: Characteristic acceleration pattern
        """
        super().__init__(
            name=name,
            dimension=ConstraintDimension.FEEDBACK,
            intensity_range=intensity_range,
            description=description,
            acceleration_profile=acceleration_profile
        )
        
        # Feedback constraints have optimal value around 0.5-0.7
        # (sparse feedback drives self-evaluation capability)
        self._optimal_intensity = 0.6
        self._optimal_range = (0.5, 0.7)
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply feedback constraint to a target object.
        
        Args:
            target: Object to apply constraint to (e.g., training loop, feedback)
            intensity: Constraint intensity (0 to 1)
            **kwargs: Additional parameters:
                - feedback_frequency_fn: Function to reduce feedback frequency
                - feedback_specificity_fn: Function to reduce feedback specificity
                - reward_sparsity_fn: Function to create reward sparsity
                
        Returns:
            object: Feedback-constrained object
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Extract constraint functions from kwargs or use defaults
        feedback_frequency_fn = kwargs.get('feedback_frequency_fn', None)
        feedback_specificity_fn = kwargs.get('feedback_specificity_fn', None)
        reward_sparsity_fn = kwargs.get('reward_sparsity_fn', None)
        
        # Create constrained copy of target
        constrained_target = target
        
        # Apply feedback frequency reduction if function provided
        if feedback_frequency_fn:
            constrained_target = feedback_frequency_fn(constrained_target, intensity)
        
        # Apply feedback specificity reduction if function provided
        if feedback_specificity_fn:
            constrained_target = feedback_specificity_fn(constrained_target, intensity)
        
        # Apply reward sparsity if function provided
        if reward_sparsity_fn:
            constrained_target = reward_sparsity_fn(constrained_target, intensity)
        
        return constrained_target


class ActionConstraint(ConstraintType):
    """
    Constraints on available actions.
    
    Action constraints force systems to develop more compositional and
    strategic behaviors by limiting the available action space.
    """
    
    def __init__(
        self,
        name: str = "Action Constraint",
        intensity_range: Tuple[float, float] = (0.0, 0.8),
        description: str = "Constraints on available actions",
        acceleration_profile: str = "inverted_u"
    ):
        """
        Initialize an action constraint.
        
        Args:
            name: Name of the constraint
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint
            acceleration_profile: Characteristic acceleration pattern
        """
        super().__init__(
            name=name,
            dimension=ConstraintDimension.ACTION,
            intensity_range=intensity_range,
            description=description,
            acceleration_profile=acceleration_profile
        )
        
        # Action constraints have optimal value around 0.4-0.6
        self._optimal_intensity = 0.5
        self._optimal_range = (0.4, 0.6)
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply action constraint to a target object.
        
        Args:
            target: Object to apply constraint to (e.g., agent, policy)
            intensity: Constraint intensity (0 to 1)
            **kwargs: Additional parameters:
                - action_space_fn: Function to reduce action space
                - action_rate_fn: Function to limit action rate
                - action_precision_fn: Function to reduce action precision
                
        Returns:
            object: Action-constrained object
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Extract constraint functions from kwargs or use defaults
        action_space_fn = kwargs.get('action_space_fn', None)
        action_rate_fn = kwargs.get('action_rate_fn', None)
        action_precision_fn = kwargs.get('action_precision_fn', None)
        
        # Create constrained copy of target
        constrained_target = target
        
        # Apply action space reduction if function provided
        if action_space_fn:
            constrained_target = action_space_fn(constrained_target, intensity)
        
        # Apply action rate limitation if function provided
        if action_rate_fn:
            constrained_target = action_rate_fn(constrained_target, intensity)
        
        # Apply action precision reduction if function provided
        if action_precision_fn:
            constrained_target = action_precision_fn(constrained_target, intensity)
        
        return constrained_target


class ArchitecturalConstraint(ConstraintType):
    """
    Constraints on network architecture.
    
    Architectural constraints force systems to develop more efficient information
    flow by limiting the architectural components available.
    """
    
    def __init__(
        self,
        name: str = "Architectural Constraint",
        intensity_range: Tuple[float, float] = (0.0, 0.7),
        description: str = "Constraints on network architecture",
        acceleration_profile: str = "inverted_u"
    ):
        """
        Initialize an architectural constraint.
        
        Args:
            name: Name of the constraint
            intensity_range: Valid range for constraint intensity
            description: Description of the constraint
            acceleration_profile: Characteristic acceleration pattern
        """
        super().__init__(
            name=name,
            dimension=ConstraintDimension.ARCHITECTURAL,
            intensity_range=intensity_range,
            description=description,
            acceleration_profile=acceleration_profile
        )
        
        # Architectural constraints have optimal value around 0.4-0.6
        self._optimal_intensity = 0.5
        self._optimal_range = (0.4, 0.6)
    
    def apply(self, target: object, intensity: float, **kwargs) -> object:
        """
        Apply architectural constraint to a target object.
        
        Args:
            target: Object to apply constraint to (e.g., model architecture)
            intensity: Constraint intensity (0 to 1)
            **kwargs: Additional parameters:
                - layer_reduction_fn: Function to reduce number of layers
                - connection_sparsity_fn: Function to create connection sparsity
                - component_simplification_fn: Function to simplify components
                
        Returns:
            object: Architecturally constrained object
        """
        if not self.validate_intensity(intensity):
            raise ValueError(f"Intensity {intensity} is outside valid range {self.intensity_range}")
        
        # Extract constraint functions from kwargs or use defaults
        layer_reduction_fn = kwargs.get('layer_reduction_fn', None)
        connection_sparsity_fn = kwargs.get('connection_sparsity_fn', None)
        component_simplification_fn = kwargs.get('component_simplification_fn', None)
        
        # Create constrained copy of target
        constrained_target = target
        
        # Apply layer reduction if function provided
        if layer_reduction_fn:
            constrained_target = layer_reduction_fn(constrained_target, intensity)
        
        # Apply connection sparsity if function provided
        if connection_sparsity_fn:
            constrained_target = connection_sparsity_fn(constrained_target, intensity)
        
        # Apply component simplification if function provided
        if component_simplification_fn:
            constrained_target = component_simplification_fn(constrained_target, intensity)
        
        return constrained_target


class MultiDimensionalConstraint:
    """
    Composite constraint that combines multiple constraint dimensions.
    
    Multi-dimensional constraints enable more sophisticated constraint engineering
    by combining different constraint types with varying intensities.
    
    Attributes:
        name (str): Name of the multi-dimensional constraint
        constraints (Dict[ConstraintDimension, ConstraintType]): Component constraints
        description (str): Description of the multi-dimensional constraint
    """
    
    def __init__(
        self,
        name: str,
        constraints: Dict[ConstraintDimension, Tuple[ConstraintType, float]],
        description: str = ""
    ):
        """
        Initialize a multi-dimensional constraint.
        
        Args:
            name: Name of the multi-dimensional constraint
            constraints: Dictionary mapping dimensions to (constraint, intensity) tuples
            description: Description of the multi-dimensional constraint
        """
        self.name = name
        self.constraints = constraints
        self.description = description
    
    def apply(self, target: object, **kwargs) -> object:
        """
        Apply all component constraints to a target object.
        
        Args:
            target: Object to apply constraints to
            **kwargs: Additional parameters for constraint application
            
        Returns:
            object: Constrained object
        """
        constrained_target = target
        
        # Apply each constraint in sequence
        for dimension, (constraint, intensity) in self.constraints.items():
            # Filter kwargs for this constraint dimension
            dimension_kwargs = {k: v for k, v in kwargs.items() if k.startswith(f"{dimension.value}_")}
            
            # Apply constraint with its intensity
            constrained_target = constraint.apply(constrained_target, intensity, **dimension_kwargs)
        
        return constrained_target
    
    def compute_acceleration_factor(self, recursive_depth: float, base_factor: float = 1.0) -> float:
        """
        Compute the combined acceleration factor for all constraints.
        
        Args:
            recursive_depth: Recursive depth of the system
            base_factor: Base acceleration factor
            
        Returns:
            float: Combined acceleration factor
        """
        # Start with base factor
        acceleration = base_factor
        
        # Multiply by acceleration factor for each constraint
        for dimension, (constraint, intensity) in self.constraints.items():
            constraint_acceleration = constraint.compute_acceleration_factor(
                intensity, recursive_depth, base_factor=1.0
            )
            
            # Apply diminishing returns for multiple constraints
            # (to avoid unrealistic multiplication effects)
            acceleration += (constraint_acceleration - 1.0)
        
        return max(1.0, acceleration)  # Ensure acceleration is at least 1.0
    
    def get_optimal_configuration(self, recursive_depth: float = 1.0) -> Dict[ConstraintDimension, float]:
        """
        Get the optimal constraint intensities for all dimensions.
        
        Args:
            recursive_depth: Recursive depth of the system
            
        Returns:
            Dict[ConstraintDimension, float]: Optimal intensities by dimension
        """
        optimal_config = {}
        
        for dimension, (constraint, _) in self.constraints.items():
            optimal_config[dimension] = constraint.get_optimal_intensity(recursive_depth)
        
        return optimal_config
    
    def __str__(self) -> str:
        """String representation of the multi-dimensional constraint."""
        return f"{self.name} ({len(self.constraints)} dimensions)"
    
    def __repr__(self) -> str:
        """Detailed representation of the multi-dimensional constraint."""
        dimensions = ", ".join(dim.value for dim in self.constraints.keys())
        return f"{self.__class__.__name__}(name='{self.name}', dimensions=[{dimensions}])"


# Factory functions for creating common constraint configurations

def create_computational_constraint(
    intensity: float = 0.5,
    name: str = "Parameter Reduction",
    **kwargs
) -> ComputationalConstraint:
    """
    Create a computational constraint with specified intensity.
    
    Args:
        intensity: Constraint intensity (0 to 1)
        name: Name of the constraint
        **kwargs: Additional constraint parameters
        
    Returns:
        ComputationalConstraint: Configured constraint
    """
    constraint = ComputationalConstraint(name=name, **kwargs)
    if not constraint.validate_intensity(intensity):
        raise ValueError(f"Intensity {intensity} is outside valid range {constraint.intensity_range}")
    return constraint


def create_representational_constraint(
    intensity: float = 0.5,
    name: str = "Embedding Compression",
    **kwargs
) -> RepresentationalConstraint:
    """
    Create a representational constraint with specified intensity.
    
    Args:
        intensity: Constraint intensity (0 to 1)
        name: Name of the constraint
        **kwargs: Additional constraint parameters
        
    Returns:
        RepresentationalConstraint: Configured constraint
    """
    constraint = Represent
