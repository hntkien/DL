class EarlyStopping:
    """Stop training when a monitored loss stops improving.

    Example usage::

        stopper = EarlyStopping(patience=15, min_delta=1e-4, min_epochs=20)
        for epoch in range(1, max_epochs + 1):
            loss = run_epoch(...)
            if stopper.step(epoch, loss):
                break
            if stopper.improved:
                save_best_checkpoint(...)

    Args:
        patience (int): Epochs to wait after the last improvement before
            stopping. Defaults to 15.
        min_delta (float): Minimum absolute decrease in loss that counts as an
            improvement. Defaults to 1e-4.
        min_epochs (int): Never stop before this many epochs, even if patience
            is exhausted. Defaults to 20.
    """
    def __init__(
            self, 
            patience: int = 15, 
            min_delta: float = 1e-4, 
            min_epochs: int = 20,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_epochs < 0:
            raise ValueError("min_epochs must be non-negative.")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_loss = float("inf")
        self._counter = 0 
        self._improved = False 

    def step(self, epoch: int, loss: float) -> bool:
        """Evaluate whether training should stop after this epoch.

        Behaviour:
            - Always tracks the best loss seen so far.
            - Within the ``min_epochs`` protection window the patience counter is never incremented, so stopping cannot trigger early.
            - After ``min_epochs``, the counter increments on every epoch that fails to improve by at least ``min_delta``.

        Args:
            epoch (int): Current epoch number (1-indexed).
            loss (float): Epoch-averaged training loss.

        Returns:
            bool: ``True`` if training should stop, ``False`` otherwise.
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss 
            self._counter = 0 
            self._improved = True
        else:
            self._improved = False
            if epoch > self.min_epochs:
                self._counter += 1

        return self._counter >= self.patience
    
    @property
    def improved(self) -> bool:
        """Whether the most recent loss was an improvement over the best loss."""
        return self._improved
    
    @property
    def counter(self) -> int:
        """Number of epochs since the last improvement."""
        return self._counter
    
    def state_dict(self) -> dict:
        """Return the full state for checkpointing.

        Returns:
            dict: Contains ``best_loss``, ``counter``, ``patience``,
                ``min_delta``, and ``min_epochs``.
        """
        return {
            "best_loss":  self.best_loss,
            "counter":    self._counter,
            "patience":   self.patience,
            "min_delta":  self.min_delta,
            "min_epochs": self.min_epochs,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state from a checkpoint.

        Only the mutable runtime fields (``best_loss``, ``counter``) are
        restored; hyperparameters (``patience``, ``min_delta``, ``min_epochs``)
        are overwritten from the checkpoint so that resuming is fully
        reproducible.

        Args:
            state_dict (dict): State dict previously returned by :meth:`state_dict`.
        """
        self.best_loss  = state_dict["best_loss"]
        self._counter   = state_dict["counter"]
        self.patience   = state_dict.get("patience",   self.patience)
        self.min_delta  = state_dict.get("min_delta",  self.min_delta)
        self.min_epochs = state_dict.get("min_epochs", self.min_epochs)
        self._improved  = False  # reset; we don't persist this across epochs

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"min_delta={self.min_delta}, min_epochs={self.min_epochs}, "
            f"best_loss={self.best_loss:.6f}, counter={self._counter})"
        )