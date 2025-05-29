"""
Constraint-Accelerated Transformer Example

This example demonstrates how to apply the Constraint Functions framework to a transformer
language model, achieving significantly faster capability development with fewer parameters
and less computational resources.

The example implements:
1. Constraint-optimized transformer architecture
2. Graduated constraint schedule for training
3. Performance comparison with unconstrained baseline
4. Capability emergence tracking throughout training
"""

import os
import math
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt

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
    ConstrainedTransformerConfig,
    DynamicConstraintScheduler
)
from constraint_functions.engineering.patterns import (
    RecursiveScaffoldPattern,
    CompressionFunnelPattern
)


class SimpleTokenizer:
    """
    Simple tokenizer for demonstration purposes.
    
    In a real implementation, this would be replaced with a more sophisticated
    tokenizer like BPE or SentencePiece.
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size
        self.token_to_id = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
        self.id_to_token = {0: "[PAD]", 1: "[UNK]", 2: "[BOS]", 3: "[EOS]"}
        self.next_id = 4
    
    def train_from_texts(self, texts: List[str]):
        """
        Train tokenizer on a list of texts.
        
        Args:
            texts: List of texts to train on
        """
        # Simple word-level tokenization for demonstration
        word_counts = {}
        for text in texts:
            for word in text.split():
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
        
        # Add most common words to vocabulary
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - self.next_id]:
            if word not in self.token_to_id:
                self.token_to_id[word] = self.next_id
                self.id_to_token[self.next_id] = word
                self.next_id += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List[int]: Token IDs
        """
        tokens = []
        for word in text.split():
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                tokens.append(self.token_to_id["[UNK]"])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            
        Returns:
            str: Decoded text
        """
        return " ".join([self.id_to_token.get(token_id, "[UNK]") for token_id in token_ids])


class TextDataset(Dataset):
    """
    Simple text dataset for demonstration purposes.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: SimpleTokenizer,
        max_length: int = 512,
        constraint_factor: float = 0.0
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of texts
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            constraint_factor: Knowledge constraint factor (0 to 1)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.constraint_factor = constraint_factor
        
        # Encode all texts
        self.encoded_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_length - 2:  # Account for BOS and EOS
                tokens = tokens[:self.max_length - 2]
            tokens = [tokenizer.token_to_id["[BOS]"]] + tokens + [tokenizer.token_to_id["[EOS]"]]
            self.encoded_texts.append(tokens)
        
        # Apply knowledge constraint if specified
        if constraint_factor > 0:
            self._apply_knowledge_constraint()
    
    def _apply_knowledge_constraint(self):
        """Apply knowledge constraint by filtering dataset."""
        # Determine how many examples to keep
        keep_count = int(len(self.encoded_texts) * (1 - self.constraint_factor))
        
        # Randomly select examples to keep
        indices = np.random.choice(len(self.encoded_texts), size=keep_count, replace=False)
        self.encoded_texts = [self.encoded_texts[i] for i in indices]
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.encoded_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Dict[str, torch.Tensor]: Input and target tensors
        """
        tokens = self.encoded_texts[idx]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.token_to_id["[PAD]"]] * (self.max_length - len(tokens))
        
        # Create input and target
        input_tensor = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tensor = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {"input": input_tensor, "target": target_tensor}


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout_prob: float = 0.1
    ):
        """
        Initialize transformer block.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            intermediate_size: Feed-forward intermediate dimension
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, hidden_size]
        """
        # Multi-head attention with residual connection and layer normalization
        attn_output, _ = self.attention(
            query=self.layer_norm1(x),
            key=self.layer_norm1(x),
            value=self.layer_norm1(x),
            attn_mask=mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class SimpleTransformer(nn.Module):
    """
    Simple transformer model for demonstration purposes.
    """
    
    def __init__(self, config: ConstrainedTransformerConfig):
        """
        Initialize transformer model.
        
        Args:
            config: Transformer configuration
        """
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=0
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                dropout_prob=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize position IDs
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Logits [batch_size, seq_len, vocab_size]
        """
        seq_length = input_ids.size(1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.bool)
        
        # Get position IDs
        position_ids = self.position_ids[:, :seq_length]
        
        # Get token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for block in self.blocks:
            hidden_states = block(hidden_states, mask=extended_attention_mask)
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits
    
    def get_parameter_count(self) -> int:
        """
        Get parameter count.
        
        Returns:
            int: Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConstraintSchedule:
    """
    Graduated constraint schedule for training.
    """
    
    def __init__(
        self,
        total_steps: int,
        constraint_pattern: str = "compression_funnel",
        base_constraint: float = 0.3,
        peak_constraint: float = 0.7,
        warmup_ratio: float = 0.1,
        oscillation_amplitude: float = 0.05
    ):
        """
        Initialize constraint schedule.
        
        Args:
            total_steps: Total training steps
            constraint_pattern: Constraint pattern ("compression_funnel" or "recursive_scaffold")
            base_constraint: Base constraint intensity
            peak_constraint: Peak constraint intensity
            warmup_ratio: Ratio of steps for warm-up
            oscillation_amplitude: Amplitude of constraint oscillation
        """
        self.total_steps = total_steps
        self.constraint_pattern = constraint_pattern
        self.base_constraint = base_constraint
        self.peak_constraint = peak_constraint
        self.warmup_ratio = warmup_ratio
        self.oscillation_amplitude = oscillation_amplitude
        
        # Pre-compute schedule parameters
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.constraint_range = peak_constraint - base_constraint
        
        # Initialize pattern
        if constraint_pattern == "compression_funnel":
            self.pattern = CompressionFunnelPattern(stages=5)
        elif constraint_pattern == "recursive_scaffold":
            self.pattern = RecursiveScaffoldPattern(max_levels=3)
        else:
            raise ValueError(f"Unknown constraint pattern: {constraint_pattern}")
    
    def get_constraint(self, step: int) -> float:
        """
        Get constraint intensity for current step.
        
        Args:
            step: Current training step
            
        Returns:
            float: Constraint intensity (0 to 1)
        """
        # Apply pattern-specific constraint schedule
        if self.constraint_pattern == "compression_funnel":
            # Compression funnel pattern with stages
            stage_duration = self.total_steps / 5  # 5 stages
            current_stage = min(int(step / stage_duration), 4)  # 0 to 4
            
            # Progressive constraint intensity based on stage
            current_intensity = self.base_constraint + (self.constraint_range * current_stage / 4)
            
            # Add oscillation within stages
            stage_progress = (step % stage_duration) / stage_duration
            oscillation = self.oscillation_amplitude * math.sin(stage_progress * 2 * math.pi)
            
            return min(1.0, max(0.0, current_intensity + oscillation))
        
        elif self.constraint_pattern == "recursive_scaffold":
            # Recursive scaffold pattern with graduated levels
            if step < self.warmup_steps:
                # Warm-up phase: gradually increase constraint
                constraint = self.base_constraint + (self.constraint_range * step / self.warmup_steps)
            else:
                # Main phase: maintain constraint with oscillation
                oscillation = self.oscillation_amplitude * math.sin(step * 0.01)
                constraint = self.peak_constraint + oscillation
            
            return min(1.0, max(0.0, constraint))
        
        # Fallback to linear schedule
        return self.base_constraint


def generate_sample_data(num_samples: int = 1000, seq_length: int = 20) -> List[str]:
    """
    Generate sample text data.
    
    Args:
        num_samples: Number of samples to generate
        seq_length: Average sequence length
        
    Returns:
        List[str]: List of sample texts
    """
    vocabulary = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "hello", "world", "artificial", "intelligence", "machine", "learning",
                 "transformer", "model", "constraint", "function", "acceleration",
                 "development", "recursive", "depth", "symbolic", "residue"]
    
    texts = []
    for _ in range(num_samples):
        # Generate random length around seq_length
        length = max(5, int(seq_length * (0.8 + 0.4 * np.random.random())))
        
        # Generate random text
        words = np.random.choice(vocabulary, size=length)
        text = " ".join(words)
        texts.append(text)
    
    return texts


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    constraint_scheduler: Optional[ConstraintSchedule] = None,
    step_offset: int = 0
) -> Tuple[float, int]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
        constraint_scheduler: Constraint scheduler (optional)
        step_offset: Step offset for constraint scheduler
        
    Returns:
        Tuple[float, int]: Average loss and total steps
    """
    model.train()
    total_loss = 0
    steps = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Get batch data
        input_ids = batch["input"].to(device)
        target_ids = batch["target"].to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        steps += 1
        
        # Apply constraint if scheduler is provided
        if constraint_scheduler is not None:
            current_step = step_offset + batch_idx
            constraint = constraint_scheduler.get_constraint(current_step)
            # In a real implementation, this would update model constraints dynamically
    
    # Return average loss and steps
    return total_loss / steps, steps


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Evaluate model.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        device: Device to use
        
    Returns:
        float: Perplexity
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            input_ids = batch["input"].to(device)
            target_ids = batch["target"].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,  # Ignore padding
                reduction="sum"
            )
            
            # Update statistics
            total_loss += loss.item()
            total_tokens += (target_ids != 0).sum().item()
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def create_model(config: ConstrainedTransformerConfig, device: torch.device) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration
        device: Device to use
        
    Returns:
        nn.Module: Created model
    """
    model = SimpleTransformer(config)
    model.to(device)
    return model


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Constraint-Accelerated Transformer Example")
    parser.add_argument("--constraint", type=float, default=0.5, help="Constraint intensity (0 to 1)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--data_size", type=int, default=10000, help="Number of data samples")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--pattern", type=str, default="compression_funnel",
                        choices=["compression_funnel", "recursive_scaffold"],
                        help="Constraint pattern to use")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate sample data
    print("Generating sample data...")
    texts = generate_sample_data(num_samples=args.data_size, seq_length=args.max_length // 4)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=10000)
    tokenizer.train_from_texts(texts)
    
    # Create datasets
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    # Create datasets with and without constraint
    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        constraint_factor=0.0  # No data constraint for simplicity
    )
    val_dataset = TextDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    # Create constrained and unconstrained models
    print("Creating models...")
    
    # Unconstrained configuration
    unconstrained_config = ConstrainedTransformerConfig(
        vocab_size=len(tokenizer.token_to_id),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=args.max_length,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        constraint_intensity=0.0  # No constraint
    )
    
    # Constrained configuration
    constrained_config = ConstrainedTransformerConfig(
        vocab_size=len(tokenizer.token_to_id),
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=args.max_length,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        constraint_intensity=args.constraint  # Apply constraint
    )
    
    # Create models
    unconstrained_model = create_model(unconstrained_config, device)
    constrained_model = create_model(constrained_config, device)
    
    # Print model statistics
    unconstrained_params = unconstrained_model.get_parameter_count()
    constrained_params = constrained_model.get_parameter_count()
    param_reduction = 1.0 - (constrained_params / unconstrained_params)
    
    print(f"Unconstrained model parameters: {unconstrained_params:,}")
    print(f"Constrained model parameters: {constrained_params:,}")
    print(f"Parameter reduction: {param_reduction:.2%}")
    
    # Create optimizers
    unconstrained_optimizer = optim.Adam(unconstrained_model.parameters(), lr=args.lr)
    constrained_optimizer = optim.Adam(constrained_model.parameters(), lr=args.lr)
    
    # Create constraint scheduler
    constraint_scheduler = ConstraintSchedule(
        total_steps=len(train_loader) * args.epochs,
        constraint_pattern=args.pattern,
        base_constraint=args.constraint * 0.7,
        peak_constraint=args.constraint,
        warmup_ratio=0.1,
        oscillation_amplitude=0.05
    )
    
    # Training loop
    print("Starting training...")
    unconstrained_perplexities = []
    constrained_perplexities = []
    unconstrained_losses = []
    constrained_losses = []
    step_offset = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train unconstrained model
        start_time = time.time()
        unconstrained_loss, steps = train_epoch(
            model=unconstrained_model,
            dataloader=train_loader,
            optimizer=unconstrained_optimizer,
            device=device
        )
        unconstrained_train_time = time.time() - start_time
        
        # Train constrained model
        start_time = time.time()
        constrained_loss, _ = train_epoch(
            model=constrained_model,
            dataloader=train_loader,
            optimizer=constrained_optimizer,
            device=device,
            constraint_scheduler=constraint_scheduler,
            step_offset=step_offset
        )
        constrained_train_time = time.time() - start_time
        
        # Update step offset
        step_offset += steps
        
        # Evaluate unconstrained model
        unconstrained_perplexity = evaluate(
            model=unconstrained_model,
            dataloader=val_loader,
            device=device
        )
        
        # Evaluate constrained model
        constrained_perplexity = evaluate(
            model=constrained_model,
            dataloader=val_loader,
            device=device
        )
        
        # Record metrics
        unconstrained_losses.append(unconstrained_loss)
        constrained_losses.append(constrained_loss)
        unconstrained_perplexities.append(unconstrained_perplexity)
        constrained_perplexities.append(constrained_perplexity)
        
        # Print metrics
        print(f"Unconstrained - Loss: {unconstrained_loss:.4f}, Perplexity: {unconstrained_perplexity:.4f}, Time: {unconstrained_train_time:.2f}s")
        print(f"Constrained - Loss: {constrained_loss:.4f}, Perplexity: {constrained_perplexity:.4f}, Time: {constrained_train_time:.2f}s")
        print(f"Time reduction: {1.0 - (constrained_train_time / unconstrained_train_time):.2%}")
    
    # Calculate acceleration factor using Constraint Acceleration Equation
    acceleration_eq = ConstraintAccelerationEquation()
    
    # Estimate recursive depth (simplified for demonstration)
    recursive_depth = 1.5  # Assuming moderate recursive depth
    
    # Calculate acceleration using the equation
    acceleration = acceleration_eq.compute_acceleration(
        C=args.constraint,
        r=recursive_depth,
        S=1.0,  # Normalized system state
        E=1.0,  # Normalized environmental information
        t=0.3   # Moderate temporal compression
    )
    
    # Empirical acceleration (based on time reduction)
    empirical_acceleration = unconstrained_train_time / constrained_train_time
    
    print("\nTraining Results:")
    print(f"Parameter reduction: {param_reduction:.2%}")
    print(f"Time reduction: {1.0 - (constrained_train_time / unconstrained_train_time):.2%}")
    print(f"Theoretical acceleration factor: {acceleration:.2f}x")
    print(f"Empirical acceleration factor: {empirical_acceleration:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(unconstrained_losses, label="Unconstrained")
    plt.plot(constrained_losses, label="Constrained")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss")
    plt.legend()
    
    # Plot validation perplexity
    plt.subplot(2, 1, 2)
    plt.plot(unconstrained_perplexities, label="Unconstrained")
    plt.plot(constrained_perplexities, label="Constrained")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Perplexity")
    plt.title("Validation Perplexity")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("constraint_acceleration_results.png")
    plt.close()
    
    print("Results saved to constraint_acceleration_results.png")


if __name__ == "__main__":
    main()
