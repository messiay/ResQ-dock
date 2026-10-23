import torch
import numpy as np
from typing import Optional, Union

def symmetry_aware_squared_loss(
    pred_coords: torch.Tensor,       # [B, N, 3] or [N, 3]
    target_coords: torch.Tensor,     # [B, N, 3] or [N, 3]
    automorphism_perms: torch.Tensor # [N_sym, N]
) -> torch.Tensor:
    """
    Computes minimum heavy-atom mean squared error across all graph automorphisms:
    L_sym = min_{pi} (1/N) sum_i || x_hat_i - x_target_{pi(i)} ||^2
    Returns tensor of shape [B]
    """
    if pred_coords.dim() == 2:
        pred_coords = pred_coords.unsqueeze(0)
    if target_coords.dim() == 2:
        target_coords = target_coords.unsqueeze(0)
        
    B, N, _ = pred_coords.shape
    device = pred_coords.device
    perms = automorphism_perms.to(device=device, dtype=torch.long)
    N_sym = perms.shape[0]
    
    # Target perms: target[:, perms, :] -> [B, N_sym, N, 3]
    # Gather target atoms according to each permutation pi
    # perms is [N_sym, N]
    perms_expanded = perms.unsqueeze(0).unsqueeze(-1).expand(B, N_sym, N, 3) # [B, N_sym, N, 3]
    target_expanded = target_coords.unsqueeze(1).expand(B, N_sym, N, 3)     # [B, N_sym, N, 3]
    permuted_targets = torch.gather(target_expanded, 2, perms_expanded)      # [B, N_sym, N, 3]
    
    # pred_coords: [B, 1, N, 3]
    pred_expanded = pred_coords.unsqueeze(1) # [B, 1, N, 3]
    
    # Squared differences: [B, N_sym, N, 3]
    diff_sq = (pred_expanded - permuted_targets) ** 2
    # Mean over atoms: [B, N_sym]
    mse_per_perm = torch.mean(torch.sum(diff_sq, dim=-1), dim=-1) # [B, N_sym]
    
    # Minimum over all permutations: [B]
    min_mse, _ = torch.min(mse_per_perm, dim=-1)
    return min_mse

def symmetry_aware_rmsd(
    pred_coords: torch.Tensor,       # [B, N, 3] or [N, 3]
    target_coords: torch.Tensor,     # [B, N, 3] or [N, 3]
    automorphism_perms: torch.Tensor,# [N_sym, N]
    eps: float = 1e-10
) -> torch.Tensor:
    """Computes symmetry-aware heavy-atom RMSD: sqrt(min_mse) in Angstroms."""
    min_mse = symmetry_aware_squared_loss(pred_coords, target_coords, automorphism_perms)
    return torch.sqrt(torch.clamp(min_mse, min=eps))

def compute_spyrmsd(
    pred_coords: torch.Tensor | np.ndarray,
    target_coords: torch.Tensor | np.ndarray,
    mol_or_atomicnums,
    adj_matrix: Optional[np.ndarray] = None
) -> float:
    """
    Computes graph-automorphism-corrected RMSD using the official spyrmsd library.
    Prevents artificial inflation of RMSD on symmetric rings (benzene, carboxylate, guanidinium).
    """
    import numpy as np
    import spyrmsd
    from spyrmsd import rmsd, molecule

    if isinstance(pred_coords, torch.Tensor):
        c_pred = pred_coords.detach().cpu().numpy().squeeze()
    else:
        c_pred = np.asarray(pred_coords)
        
    if isinstance(target_coords, torch.Tensor):
        c_target = target_coords.detach().cpu().numpy().squeeze()
    else:
        c_target = np.asarray(target_coords)

    if hasattr(mol_or_atomicnums, "GetAtoms"):
        # RDKit Mol
        spy_mol = molecule.Molecule.from_rdkit(mol_or_atomicnums)
        atomicnums = spy_mol.atomicnums
        adj = spy_mol.adjacency_matrix
    else:
        atomicnums = np.asarray(mol_or_atomicnums)
        adj = adj_matrix if adj_matrix is not None else np.ones((len(atomicnums), len(atomicnums)))

    val = rmsd.symmrmsd(
        c_target,
        c_pred,
        atomicnums,
        atomicnums,
        adj,
        adj,
        center=False,
        minimize=False
    )
    if isinstance(val, (list, np.ndarray)):
        return float(val[0])
    return float(val)

