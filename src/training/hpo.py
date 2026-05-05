"""
DataFlix — Optuna Hyperparameter Optimisation
TPE search over k, λ, η₀, H, d.
"""

import optuna
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    OPTUNA_N_TRIALS, OPTUNA_LATENT_DIMS, OPTUNA_REG_RANGE,
    OPTUNA_LR_RANGE, OPTUNA_HEADS, OPTUNA_EMBED_DIMS,
    RESULTS_DIR
)
from src.models.hybrid import DataFlixModel
from src.training.trainer import PathATrainer


def create_objective(train_dataset, val_dataset, n_users, n_items):
    """Create Optuna objective function."""
    
    def objective(trial):
        # Sample hyperparameters
        k = trial.suggest_categorical("latent_dim_k", OPTUNA_LATENT_DIMS)
        d = trial.suggest_categorical("embed_dim_d", OPTUNA_EMBED_DIMS)
        n_heads = trial.suggest_categorical("n_heads", OPTUNA_HEADS)
        lr = trial.suggest_float("lr", *OPTUNA_LR_RANGE, log=True)
        reg = trial.suggest_float("weight_decay", *OPTUNA_REG_RANGE, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        
        # Ensure d is divisible by n_heads
        while d % n_heads != 0:
            n_heads = trial.suggest_categorical("n_heads_retry", [h for h in OPTUNA_HEADS if d % h == 0])
        
        # Build model
        model = DataFlixModel(
            n_users=n_users, n_items=n_items,
            k=k, d=d, n_heads=n_heads,
            dropout=dropout, path="A"
        )
        
        # Train with reduced epochs for HPO
        trainer = PathATrainer(
            model, lr=lr, weight_decay=reg,
            max_epochs=15,  # Fewer epochs for HPO
            patience=3,
            batch_size=4096
        )
        
        history = trainer.train(train_dataset, val_dataset)
        best_rmse = min(history["val_rmse"]) if history["val_rmse"] else float("inf")
        
        return best_rmse
    
    return objective


def run_hpo(train_dataset, val_dataset, n_users: int, n_items: int,
            n_trials: int = OPTUNA_N_TRIALS):
    """
    Run Optuna hyperparameter search.
    """
    objective = create_objective(train_dataset, val_dataset, n_users, n_items)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    print(f"Starting Optuna HPO with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "=" * 60)
    print("HPO Results")
    print("=" * 60)
    print(f"Best trial RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ]
    }
    with open(RESULTS_DIR / "hpo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return study.best_params
