import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CanonicalTransform:
    origin: torch.Tensor       # [3]
    rot_matrix: torch.Tensor   # [3, 3] SO(3) with det == +1.0

class PocketCanonicalizer:
    """
    SE(3)-Invariant Pocket Canonicalization using distance-weighted PCA with Chirality Lock.
    Guarantees det(V) = +1 (pure rotation, no reflection) and robust sign disambiguation.
    Computes PCA using double precision internally to prevent catastrophic loss of significance
    under large global translations (e.g. +/- 250 Å).
    """
    def __init__(self, sigma: float = 6.0, eps: float = 1e-8):
        self.sigma = sigma
        self.eps = eps

    def compute_canonical_frame(
        self,
        calpha_coords: torch.Tensor,
        whole_protein_centroid: Optional[torch.Tensor] = None,
    ) -> CanonicalTransform:
        """
        calpha_coords: [N_ca, 3]
        whole_protein_centroid: Optional [3]
        Returns CanonicalTransform(origin, rot_matrix)
        """
        device = calpha_coords.device
        orig_dtype = calpha_coords.dtype
        # Use float64 internally for numerical stability under large translations
        coords_d = calpha_coords.to(device=device, dtype=torch.float64)
        
        # 1. Pocket geometric centroid using two-pass shifted mean for stability
        c0 = coords_d[0]
        mu_pocket = c0 + torch.mean(coords_d - c0, dim=0) # [3]
        
        # 2. Distance-weighted covariance matrix
        diff = coords_d - mu_pocket # [N, 3]
        dist_sq = torch.sum(diff ** 2, dim=-1) # [N]
        weights = torch.exp(-dist_sq / (2.0 * (self.sigma ** 2))) # [N]
        weights = weights / (torch.sum(weights) + self.eps)
        
        weighted_diff = diff * torch.sqrt(weights.unsqueeze(-1)) # [N, 3]
        cov = torch.matmul(weighted_diff.T, weighted_diff) # [3, 3]
        
        # 3. Eigenvectors ordered by descending eigenvalue
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)
        v1 = evecs[:, idx[0]] # [3]
        v2 = evecs[:, idx[1]] # [3]
        
        # 4. Sign Disambiguation & Chirality Lock
        if whole_protein_centroid is not None:
            c_prot = whole_protein_centroid.to(device=device, dtype=torch.float64)
            d_prot = c_prot - mu_pocket
        else:
            d_prot = None
            
        # Third-order distance-weighted skewness vector as fallback
        skewness_dir = torch.sum(diff * (dist_sq.unsqueeze(-1) * weights.unsqueeze(-1)), dim=0)
        ref_dir = d_prot if (d_prot is not None and torch.norm(d_prot) > 1e-3) else skewness_dir
        if torch.norm(ref_dir) < 1e-6:
            ref_dir = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float64)

        if torch.dot(v1, ref_dir) < 0:
            v1 = -v1
        if torch.dot(v2, ref_dir) < 0:
            v2 = -v2
            
        # Guarantee det(V) = +1 via cross product
        v3 = torch.linalg.cross(v1, v2)
        v3 = v3 / (torch.norm(v3) + self.eps)
        
        # Re-orthogonalize v2 = v3 x v1
        v2 = torch.linalg.cross(v3, v1)
        v2 = v2 / (torch.norm(v2) + self.eps)
        v1 = v1 / (torch.norm(v1) + self.eps)
        
        V = torch.stack([v1, v2, v3], dim=1) # [3, 3]
        
        det = torch.linalg.det(V)
        if det < 0:
            v2 = -v2
            V = torch.stack([v1, v2, v3], dim=1)

        target_dtype = orig_dtype if orig_dtype in [torch.float32, torch.float64] else torch.float32
        return CanonicalTransform(
            origin=mu_pocket.to(dtype=target_dtype),
            rot_matrix=V.to(dtype=target_dtype)
        )

    def to_canonical(self, coords: torch.Tensor, transform: CanonicalTransform) -> torch.Tensor:
        """Transform global coordinates to canonical pocket frame: (x - mu) @ V"""
        orig_shape = coords.shape
        flat_coords = coords.view(-1, 3)
        origin = transform.origin.to(device=flat_coords.device, dtype=flat_coords.dtype)
        rot_mat = transform.rot_matrix.to(device=flat_coords.device, dtype=flat_coords.dtype)
        canon = torch.matmul(flat_coords - origin, rot_mat)
        return canon.view(orig_shape)

    def to_global(self, canon_coords: torch.Tensor, transform: CanonicalTransform) -> torch.Tensor:
        """Transform canonical coordinates back to global frame: x_canon @ V^T + mu"""
        orig_shape = canon_coords.shape
        flat_canon = canon_coords.view(-1, 3)
        origin = transform.origin.to(device=flat_canon.device, dtype=flat_canon.dtype)
        rot_mat = transform.rot_matrix.to(device=flat_canon.device, dtype=flat_canon.dtype)
        glob = torch.matmul(flat_canon, rot_mat.T) + origin
        return glob.view(orig_shape)
