from dataclasses import dataclass
from typing import Protocol, Tuple, Optional
import torch

@dataclass
class CanonicalComplexState:
    """SE(3)-canonicalized representation of pocket + ligand reference conformer."""
    complex_id: str
    pocket_node_feats: torch.Tensor       # [N_res, D_res] Precomputed ESM-2 + geometric features
    pocket_atom_coords: torch.Tensor      # [N_poc, 3] Canonical frame coords (float32)
    pocket_atom_charges: torch.Tensor     # [N_poc] Partial charges for FreeSolvE Poisson grid
    pocket_atom_vdw: torch.Tensor         # [N_poc] Van der Waals radii in Angstroms
    freesolve_fft_grid: Optional[torch.Tensor] # [G_x, G_y, G_z] Precomputed 1.0 Å Poisson potential Φ(r)
    
    # Canonical frame transformation (to map predicted local coords back to global PDB space)
    canonical_origin: torch.Tensor        # [3] Global pocket centroid mu_pocket
    canonical_rot_matrix: torch.Tensor    # [3, 3] SO(3) PCA matrix V with det(V) = +1
    
    # Ligand graph & kinematic tree
    ligand_atom_feats: torch.Tensor       # [N_lig, D_lig]
    ligand_ref_coords: torch.Tensor       # [N_lig, 3] Zero-centered RDKit ETKDG reference conformer
    ligand_atom_charges: torch.Tensor     # [N_lig] Gasteiger/MMFF partial charges
    ligand_atom_vdw: torch.Tensor         # [N_lig] Van der Waals radii
    rotatable_edges: torch.Tensor         # [M, 2] Directed bond atom indices (u_m, v_m)
    kinematic_masks: torch.Tensor         # [M, N_lig] Bool mask of downstream atoms rotated by bond m
    bond_tree_depths: torch.Tensor        # [M] Integer BFS depth from central root ring (1 = core)
    automorphism_perms: torch.Tensor      # [N_sym, N_lig] Graph symmetry permutations for spyrmsd
    
    # Ground-truth articulated parameters in canonical pocket frame (None at inference)
    gt_translation: Optional[torch.Tensor] = None # [3] T in [-12, +12]^3 Å
    gt_rotation_6d: Optional[torch.Tensor] = None # [6] Continuous 6D SO(3) representation
    gt_axis_angle: Optional[torch.Tensor] = None  # [3] omega in so(3) for FreeSolvE compatibility
    gt_torsions_sc: Optional[torch.Tensor] = None # [M, 2] (sin(theta_m), cos(theta_m))

@dataclass
class DiscretePoseTokens:
    """4-Scale hierarchical discrete representation (11 tokens total per pose)."""
    scale1_anchor: torch.Tensor           # [B, 1] in {0 .. 511}
    scale2_rigid: torch.Tensor            # [B, 2] s_2a in {0 .. 511}, s_2b in {0 .. 1023}
    scale3_core_torsions: torch.Tensor    # [B, 4] in {0 .. 1023} (or 1024 = RIGID_BYPASS if M==0)
    scale4_fine_torsions: torch.Tensor    # [B, 4] in {0 .. 1023} (or 1024 = RIGID_BYPASS if M==0)

@dataclass
class ArticulatedPoseParams:
    """Continuous decoded kinematic parameters on R^3 x SO(3) x T^M."""
    translation: torch.Tensor             # [B, 3]
    rotation_6d: torch.Tensor             # [B, 6]
    torsions_sc: torch.Tensor             # [B, M, 2] normalized (sin, cos) pairs

class PoseTokenizerProtocol(Protocol):
    def encode(self, state: CanonicalComplexState) -> DiscretePoseTokens: ...
    def decode_params(self, tokens: DiscretePoseTokens, state: CanonicalComplexState) -> ArticulatedPoseParams: ...
    def decode_coords(self, tokens: DiscretePoseTokens, state: CanonicalComplexState, up_to_scale: int = 4) -> torch.Tensor: ...

class PhysicsRelaxationProtocol(Protocol):
    def compute_physics_loss(self, state: CanonicalComplexState, pred_coords: torch.Tensor) -> torch.Tensor: ...
    def relax_pose(
        self,
        state: CanonicalComplexState,
        init_params: ArticulatedPoseParams,
        max_steps: int = 40,
    ) -> Tuple[torch.Tensor, ArticulatedPoseParams, torch.Tensor]:
        """Returns (relaxed_coords [B, N_lig, 3], relaxed_params, final_energy_kcal [B])."""
        ...
