"""
Constraint Functions Benchmark Framework

This module provides a comprehensive benchmarking framework for evaluating the
acceleration effects of constraint functions across different model architectures,
tasks, and constraint configurations. The framework enables systematic comparison
between constrained and unconstrained approaches, measuring acceleration factors,
resource efficiency, and capability emergence.
"""

import os
import time
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional imports for different frameworks
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as fnn
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import constraint functions library
from constraint_functions.core.equations import (
    UniversalResidueEquation,
    ConstraintAccelerationEquation
)
from constraint_functions.core.constraint_types import (
    ConstraintDimension,
    ComputationalConstraint,
    RepresentationalConstraint,
    TemporalConstraint
)
from constraint_functions.architectures.transformers import (
    TransformerConstraints,
    ConstrainedTransformerConfig
)
from constraint_functions.engineering.patterns import (
    RecursiveScaffoldPattern,
    CompressionFunnelPattern,
    BoundaryExplorationPattern
)
from constraint_functions.frameworks.pytorch import (
    apply_constraints as pytorch_apply_constraints,
    ConstraintConfig as PyTorchConstraintConfig
)
if TENSORFLOW_AVAILABLE:
    from constraint_functions.frameworks.tensorflow import (
        apply_constraints as tf_apply_constraints,
        ConstraintConfig as TFConstraintConfig
    )
if JAX_AVAILABLE:
    from constraint_functions.frameworks.jax import (
        apply_constraints as jax_apply_constraints,
        ConstraintConfig as JAXConstraintConfig
    )
if TRANSFORMERS_AVAILABLE:
    from constraint_functions.frameworks.huggingface import (
        apply_transformer_constraints,
        ConstraintConfig as HFConstraintConfig
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("constraint_benchmark")


class CapabilityTracker:
    """
    Tracks the emergence of capabilities during training.
    """
    
    def __init__(self, capability_tests: Dict[str, callable], threshold: float = 0.7):
        """
        Initialize capability tracker.
        
        Args:
            capability_tests: Dictionary mapping capability names to test functions
            threshold: Threshold for considering a capability emerged
        """
        self.capability_tests = capability_tests
        self.threshold = threshold
        self.history = {name: [] for name in capability_tests.keys()}
        self.emergence_points = {}
    
    def evaluate(self, model: nn.Module, step: int):
        """
        Evaluate capabilities at current step.
        
        Args:
            model: Model to evaluate
            step: Current training step
        """
        for name, test_fn in self.capability_tests.items():
            score = test_fn(model)
            self.history[name].append((step, score))
            
            # Check if capability has emerged
            if name not in self.emergence_points and score >= self.threshold:
                self.emergence_points[name] = step
                logger.info(f"Capability emerged at step {step}: {name} (score: {score:.4f})")
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """
        Generate capability emergence report.
        
        Returns:
            Dict: Report containing emergence points and histories
        """
        return {
            "emergence_points": self.emergence_points,
            "history": self.history
        }
    
    def plot_emergence(self, output_path: str = "capability_emergence.png"):
        """
        Plot capability emergence over time.
        
        Args:
            output_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for name, history in self.history.items():
            if not history:
                continue
                
            steps, scores = zip(*history)
            plt.plot(steps, scores, label=name)
            
            # Mark emergence point if exists
            if name in self.emergence_points:
                emergence_step = self.emergence_points[name]
                # Find the score at emergence
                emergence_score = next(score for step, score in history if step == emergence_step)
                plt.scatter([emergence_step], [emergence_score], marker='o', s=100)
                plt.annotate(
                    f"{name} emerged",
                    (emergence_step, emergence_score),
                    xytext=(10, 5),
                    textcoords='offset points'
                )
        
        plt.axhline(y=self.threshold, color='r', linestyle='--', alpha=0.5, label=f"Threshold ({self.threshold})")
        plt.xlabel("Training Steps")
        plt.ylabel("Capability Score")
        plt.title("Capability Emergence Over Training")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(output_path)
        plt.close()


class ResourceTracker:
    """
    Tracks resource usage during training.
    """
    
    def __init__(self):
        """Initialize resource tracker."""
        self.parameter_counts = []
        self.memory_usage = []
        self.compute_usage = []
        self.training_times = []
        self.inference_times = []
    
    def track_parameters(self, model: nn.Module):
        """
        Track parameter count.
        
        Args:
            model: Model to track
        """
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.parameter_counts.append(param_count)
        return param_count
    
    def track_memory(self):
        """Track memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            self.memory_usage.append(memory_allocated)
            return memory_allocated
        return 0
    
    def track_training_time(self, start_time: float, end_time: float):
        """
        Track training time.
        
        Args:
            start_time: Start time
            end_time: End time
        """
        duration = end_time - start_time
        self.training_times.append(duration)
        return duration
    
    def track_inference_time(self, model: nn.Module, input_data, num_runs: int = 10):
        """
        Track inference time.
        
        Args:
            model: Model to track
            input_data: Input data for inference
            num_runs: Number of runs for averaging
        """
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(input_data)
            
            # Measure
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(input_data)
            end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        self.inference_times.append(avg_time)
        return avg_time
    
    def estimate_compute_usage(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """
        Estimate compute usage in FLOPs.
        
        Args:
            model: Model to track
            input_shape: Input shape for estimation
        """
        try:
            from thop import profile
            input_tensor = torch.zeros(input_shape, device=next(model.parameters()).device)
            flops, _ = profile(model, inputs=(input_tensor,))
            self.compute_usage.append(flops)
            return flops
        except ImportError:
            logger.warning("thop not available for FLOPs calculation. Install with pip install thop")
            return 0
    
    def get_resource_report(self) -> Dict[str, Any]:
        """
        Generate resource usage report.
        
        Returns:
            Dict: Report containing resource usage statistics
        """
        return {
            "parameter_counts": self.parameter_counts,
            "memory_usage": self.memory_usage,
            "compute_usage": self.compute_usage,
            "training_times": self.training_times,
            "inference_times": self.inference_times,
            "summary": {
                "avg_parameter_count": np.mean(self.parameter_counts) if self.parameter_counts else 0,
                "avg_memory_usage": np.mean(self.memory_usage) if self.memory_usage else 0,
                "avg_compute_usage": np.mean(self.compute_usage) if self.compute_usage else 0,
                "avg_training_time": np.mean(self.training_times) if self.training_times else 0,
                "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0
            }
        }


class ConstraintBenchmark:
    """
    Main benchmark class for evaluating constraint acceleration.
    """
    
    def __init__(
        self,
        task: str = "language_modeling",
        model_type: str = "transformer",
        dataset_name: str = "wikitext-2",
        constraint_levels: List[float] = [0.0, 0.3, 0.5, 0.7],
        constraint_patterns: List[str] = ["compression_funnel", "recursive_scaffold"],
        metrics: List[str] = ["perplexity", "training_time", "parameter_count"],
        capability_tests: Optional[Dict[str, callable]] = None,
        output_dir: str = "benchmark_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize benchmark.
        
        Args:
            task: Task to benchmark
            model_type: Model architecture type
            dataset_name: Dataset name
            constraint_levels: Constraint levels to test
            constraint_patterns: Constraint patterns to test
            metrics: Metrics to track
            capability_tests: Dictionary of capability test functions
            output_dir: Output directory for results
            device: Device to use
        """
        self.task = task
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.constraint_levels = constraint_levels
        self.constraint_patterns = constraint_patterns
        self.metrics = metrics
        self.capability_tests = capability_tests or {}
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize trackers
        self.capability_tracker = CapabilityTracker(self.capability_tests) if self.capability_tests else None
        self.resource_tracker = ResourceTracker()
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized benchmark for {task} with {model_type} on {dataset_name}")
        logger.info(f"Testing constraint levels: {constraint_levels}")
        logger.info(f"Testing constraint patterns: {constraint_patterns}")
    
    def _create_model(self, constraint_level: float, constraint_pattern: Optional[str] = None) -> nn.Module:
        """
        Create model with specified constraints.
        
        Args:
            constraint_level: Constraint intensity
            constraint_pattern: Constraint pattern to use
            
        Returns:
            nn.Module: Created model
        """
        # Model creation depends on model type
        if self.model_type == "transformer":
            if constraint_level == 0.0:
                # Unconstrained baseline
                config = ConstrainedTransformerConfig(
                    vocab_size=10000,  # Placeholder, will be set by dataset
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    constraint_intensity=0.0  # No constraint
                )
            else:
                # Constrained model
                config = ConstrainedTransformerConfig(
                    vocab_size=10000,  # Placeholder, will be set by dataset
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    constraint_intensity=constraint_level
                )
            
            # Create model based on config
            from constraint_functions.architectures.transformers import SimpleTransformer
            model = SimpleTransformer(config).to(self.device)
            
            # Apply constraint pattern if specified
            if constraint_pattern and constraint_level > 0.0:
                if constraint_pattern == "compression_funnel":
                    self.pattern = CompressionFunnelPattern(stages=5)
                elif constraint_pattern == "recursive_scaffold":
                    self.pattern = RecursiveScaffoldPattern(max_levels=3)
                elif constraint_pattern == "boundary_exploration":
                    self.pattern = BoundaryExplorationPattern(exploration_phase_ratio=0.6)
                else:
                    logger.warning(f"Unknown constraint pattern: {constraint_pattern}")
                    self.pattern = None
            else:
                self.pattern = None
        
        elif self.model_type == "mlp":
            # Simple MLP model
            if constraint_level == 0.0:
                # Unconstrained baseline
                model = nn.Sequential(
                    nn.Linear(784, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 10)
                ).to(self.device)
            else:
                # Constrained model
                hidden_size = int(1024 * (1 - constraint_level))
                model = nn.Sequential(
                    nn.Linear(784, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 10)
                ).to(self.device)
                
                # Apply constraint pattern if specified
                if constraint_pattern:
                    logger.warning("Constraint patterns not implemented for MLP models")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def _load_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load dataset for the benchmark task.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation data loaders
        """
        # Dataset loading depends on task and dataset name
        if self.task == "language_modeling":
            if self.dataset_name == "wikitext-2":
                # Simplified example for demonstration
                from torch.utils.data import TensorDataset
                
                # Dummy data for demonstration
                train_data = TensorDataset(
                    torch.randint(0, 10000, (1000, 128)).to(self.device),
                    torch.randint(0, 10000, (1000, 128)).to(self.device)
                )
                val_data = TensorDataset(
                    torch.randint(0, 10000, (100, 128)).to(self.device),
                    torch.randint(0, 10000, (100, 128)).to(self.device)
                )
                
                train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=16)
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        elif self.task == "image_classification":
            if self.dataset_name == "mnist":
                from torchvision.datasets import MNIST
                from torchvision.transforms import ToTensor
                
                train_data = MNIST(root="./data", train=True, transform=ToTensor(), download=True)
                val_data = MNIST(root="./data", train=False, transform=ToTensor())
                
                train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=64)
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return train_loader, val_loader
    
    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        constraint_level: float = 0.0,
        constraint_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train model and track metrics.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            constraint_level: Constraint intensity
            constraint_pattern: Constraint pattern to use
            
        Returns:
            Dict[str, Any]: Training results
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        if self.task == "language_modeling":
            criterion = nn.CrossEntropyLoss()
        elif self.task == "image_classification":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Track resources
        param_count = self.resource_tracker.track_parameters(model)
        logger.info(f"Model parameter count: {param_count:,}")
        
        # Estimate compute
        if self.task == "language_modeling":
            input_shape = (16, 128)  # Batch size, sequence length
        elif self.task == "image_classification":
            input_shape = (64, 1, 28, 28)  # Batch size, channels, height, width
        
        flops = self.resource_tracker.estimate_compute_usage(model, input_shape)
        logger.info(f"Estimated FLOPs: {flops:,}")
        
        # Training loop
        training_start = time.time()
        train_losses = []
        val_metrics = []
        step = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                # Get batch data based on task
                if self.task == "language_modeling":
                    inputs, targets = batch
                elif self.task == "image_classification":
                    inputs, targets = batch
                    inputs = inputs.view(inputs.size(0), -1)  # Flatten for MLP
                
                # Update constraints if using a pattern
                if constraint_pattern and constraint_level > 0.0 and self.pattern:
                    total_steps = epochs * len(train_loader)
                    try:
                        self.pattern.apply(
                            model,
                            recursive_depth=1.0,
                            step=step,
                            total_steps=total_steps
                        )
                    except Exception as e:
                        logger.warning(f"Error applying constraint pattern: {e}")
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss based on task
                if self.task == "language_modeling":
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                elif self.task == "image_classification":
                    loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                step += 1
                
                # Track memory usage periodically
                if batch_idx % 10 == 0:
                    self.resource_tracker.track_memory()
                
                # Evaluate capabilities periodically
                if self.capability_tracker and batch_idx % 50 == 0:
                    self.capability_tracker.evaluate(model, step)
            
            # Epoch metrics
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            val_metric = self._evaluate_model(model, val_loader)
            val_metrics.append(val_metric)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                       f"Validation: {val_metric:.4f}, Time: {epoch_time:.2f}s")
        
        # Track total training time
        training_end = time.time()
        training_time = self.resource_tracker.track_training_time(training_start, training_end)
        logger.info(f"Total training time: {training_time:.2f}s")
        
        # Track inference time
        if self.task == "language_modeling":
            inference_input = torch.randint(0, 10000, (1, 128)).to(self.device)
        elif self.task == "image_classification":
            inference_input = torch.randn(1, 784).to(self.device)
        
        inference_time = self.resource_tracker.track_inference_time(model, inference_input)
        logger.info(f"Average inference time: {inference_time*1000:.2f}ms")
        
        # Compile results
        results = {
            "constraint_level": constraint_level,
            "constraint_pattern": constraint_pattern,
            "final_val_metric": val_metrics[-1],
            "best_val_metric": min(val_metrics) if self.task == "language_modeling" else max(val_metrics),
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "parameter_count": param_count,
            "training_time": training_time,
            "inference_time": inference_time,
            "flops": flops,
            "memory_usage": np.mean(self.resource_tracker.memory_usage) if self.resource_tracker.memory_usage else 0
        }
        
        # Add capability emergence if available
        if self.capability_tracker:
            results["capability_emergence"] = self.capability_tracker.get_emergence_report()
        
        return results
    
    def _evaluate_model(self, model: nn.Module, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation data.
        
        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            
        Returns:
            float: Validation metric
        """
        model.eval()
        
        if self.task == "language_modeling":
            # Calculate perplexity
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            total_tokens = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    total_loss += loss.item() * targets.numel()
                    total_tokens += targets.numel()
            
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            return perplexity
        
        elif self.task == "image_classification":
            # Calculate accuracy
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.view(inputs.size(0), -1)  # Flatten for MLP
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = correct / total
            return accuracy
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Returns:
            Dict[str, Any]: Benchmark results
        """
        logger.info(f"Starting benchmark: {self.task} with {self.model_type} on {self.dataset_name}")
        
        # Load dataset
        train_loader, val_loader = self._load_dataset()
        
        # Run benchmarks for each constraint level and pattern
        for constraint_level in self.constraint_levels:
            level_results = {}
            
            if constraint_level == 0.0:
                # Unconstrained baseline (no pattern)
                logger.info(f"Benchmarking unconstrained baseline")
                
                model = self._create_model(constraint_level=constraint_level)
                results = self._train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    constraint_level=constraint_level
                )
                level_results["baseline"] = results
            else:
                # Constrained with different patterns
                for pattern in self.constraint_patterns:
                    logger.info(f"Benchmarking constraint level {constraint_level} with pattern {pattern}")
                    
                    model = self._create_model(constraint_level=constraint_level, constraint_pattern=pattern)
                    results = self._train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        constraint_level=constraint_level,
                        constraint_pattern=pattern
                    )
                    level_results[pattern] = results
            
            self.results[constraint_level] = level_results
        
        # Calculate acceleration factors relative to baseline
        self._calculate_acceleration_factors()
        
        # Save results
        self._save_results()
        
        # Generate visualization
        self._generate_visualizations()
        
        return self.results
    
    def _calculate_acceleration_factors(self):
        """Calculate acceleration factors relative to unconstrained baseline."""
        if 0.0 not in self.results:
            logger.warning("Baseline (constraint_level=0.0) not found in results")
            return
        
        baseline = self.results[0.0]["baseline"]
        
        for level, level_results in self.results.items():
            if level == 0.0:
                continue
                
            for pattern, results in level_results.items():
                # Training time acceleration
                time_acceleration = baseline["training_time"] / results["training_time"]
                results["time_acceleration"] = time_acceleration
                
                # Parameter efficiency
                param_efficiency = (baseline["parameter_count"] / results["parameter_count"]) * (
                    results["best_val_metric"] / baseline["best_val_metric"]
                )
                results["param_efficiency"] = param_efficiency
                
                # Compute efficiency
                compute_efficiency = (baseline["flops"] / results["flops"]) * (
                    results["best_val_metric"] / baseline["best_val_metric"]
                )
                results["compute_efficiency"] = compute_efficiency
                
                # Calculate theoretical acceleration using the Constraint Acceleration Equation
                acceleration_eq = ConstraintAccelerationEquation()
                theoretical_acceleration = acceleration_eq.compute_acceleration(
                    C=level,  # Constraint level
                    r=1.5,    # Estimated recursive depth
                    S=1.0,    # Normalized system state
                    E=1.0,    # Normalized environmental information
                    t=0.3     # Estimated temporal compression
                )
                results["theoretical_acceleration"] = theoretical_acceleration
                
                # Capability emergence acceleration (if available)
                if "capability_emergence" in baseline and "capability_emergence" in results:
                    baseline_emergence = baseline["capability_emergence"]["emergence_points"]
                    results_emergence = results["capability_emergence"]["emergence_points"]
                    
                    capability_acceleration = {}
                    for capability, step in results_emergence.items():
                        if capability in baseline_emergence:
                            acceleration = baseline_emergence[capability] / step
                            capability_acceleration[capability] = acceleration
                    
                    results["capability_acceleration"] = capability_acceleration
                    
                    # Average capability acceleration
                    if capability_acceleration:
                        results["avg_capability_acceleration"] = np.mean(list(capability_acceleration.values()))
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"benchmark_results_{self.task}_{self.model_type}_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for level, level_results in self.results.items():
            serializable_results[str(level)] = {}
            for pattern, results in level_results.items():
                serializable_results[str(level)][pattern] = {
                    k: v if not isinstance(v, np.ndarray) and not isinstance(v, list) or not v else 
                       v.tolist() if isinstance(v, np.ndarray) else
                       [float(x) if isinstance(x, np.float32) else x for x in v] if isinstance(v, list) else v
                    for k, v in results.items() if k not in ["model", "optimizer"]
                }
        
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary as CSV
        summary_file = self.output_dir / f"benchmark_summary_{self.task}_{self.model_type}_{timestamp}.csv"
        
        summary_data = []
        for level, level_results in self.results.items():
            for pattern, results in level_results.items():
                row = {
                    "Task": self.task,
                    "Model": self.model_type,
                    "Dataset": self.dataset_name,
                    "Constraint Level": level,
                    "Constraint Pattern": pattern,
                    "Parameter Count": results.get("parameter_count", 0),
                    "Training Time (s)": results.get("training_time", 0),
                    "Inference Time (ms)": results.get("inference_time", 0) * 1000,
                    "Best Validation Metric": results.get("best_val_metric", 0),
                    "Time Acceleration": results.get("time_acceleration", 1.0),
                    "Parameter Efficiency": results.get("param_efficiency", 1.0),
                    "Compute Efficiency": results.get("compute_efficiency", 1.0),
                    "Theoretical
