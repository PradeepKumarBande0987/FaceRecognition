"""Callbacks for training monitoring and control."""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, metrics: Dict[str, Any]):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore weights from epoch with best performance
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.wait_count = 0
        self.best_weights = None
        self.stop_training = False
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        val_loss = metrics.get('val_loss', float('inf'))
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait_count = 0
            if self.restore_best_weights:
                self.best_weights = None  # Will be set by trainer
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                self.stop_training = True
    
    def on_batch_end(self, batch: int, metrics: Dict[str, Any]):
        """Called at the end of each batch."""
        pass


class LearningRateMonitor(Callback):
    """Monitor learning rate changes."""
    
    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.batch_count = 0
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        self.batch_count = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_end(self, batch: int, metrics: Dict[str, Any]):
        """Called at the end of each batch."""
        self.batch_count += 1
        if self.batch_count % self.log_interval == 0:
            lr = metrics.get('learning_rate', None)
            if lr is not None:
                logger.info(f"Batch {batch} - Learning Rate: {lr:.6f}")


class ModelCheckpoint(Callback):
    """Save model checkpoints."""
    
    def __init__(self, save_path: str, monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True):
        """
        Args:
            save_path: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best checkpoint
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        current_value = metrics.get(self.monitor, None)
        
        if current_value is None:
            return
        
        should_save = False
        if self.mode == 'min' and current_value < self.best_value:
            should_save = True
            self.best_value = current_value
        elif self.mode == 'max' and current_value > self.best_value:
            should_save = True
            self.best_value = current_value
        
        if should_save:
            logger.info(f"Saving checkpoint to {self.save_path}")
    
    def on_batch_end(self, batch: int, metrics: Dict[str, Any]):
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback):
        """Add callback."""
        self.callbacks.append(callback)
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_start(epoch)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)
    
    def on_batch_end(self, batch: int, metrics: Dict[str, Any]):
        """Called at the end of each batch."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, metrics)
