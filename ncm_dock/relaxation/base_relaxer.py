import torch
from typing import Tuple, Optional
from ncm_dock.contracts import CanonicalComplexState, ArticulatedPoseParams, PhysicsRelaxationProtocol

class BaseRelaxer(PhysicsRelaxationProtocol):
    """Abstract base class for physics-based molecular pose relaxation."""
    def compute_physics_loss(self, state: CanonicalComplexState, pred_coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def relax_pose(
        self,
        state: CanonicalComplexState,
        init_params: ArticulatedPoseParams,
        max_steps: int = 40,
    ) -> Tuple[torch.Tensor, ArticulatedPoseParams, torch.Tensor]:
        raise NotImplementedError
