import torch
from typing import Tuple
from ncm_dock.contracts import CanonicalComplexState, ArticulatedPoseParams
from ncm_dock.relaxation.base_relaxer import BaseRelaxer
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics

class LBFGSVdWBMinimizer(BaseRelaxer):
    """Baseline L-BFGS torsion and rigid body Van der Waals minimizer for ablations."""
    def __init__(self, vdw_eps: float = 0.15, max_steps: int = 25):
        self.vdw_eps = vdw_eps
        self.max_steps = max_steps

    def compute_physics_loss(self, state: CanonicalComplexState, pred_coords: torch.Tensor) -> torch.Tensor:
        if pred_coords.dim() == 2:
            pred_coords = pred_coords.unsqueeze(0)
        poc_coords = state.pocket_atom_coords.unsqueeze(0)
        diff = pred_coords.unsqueeze(2) - poc_coords.unsqueeze(1)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-6)
        
        sig = (state.ligand_atom_vdw.unsqueeze(1) + state.pocket_atom_vdw.unsqueeze(0)).unsqueeze(0)
        ratio = sig / dist
        vdw = 4.0 * self.vdw_eps * (torch.clamp(ratio ** 12, max=1e3) - torch.clamp(ratio ** 6, max=1e3))
        return torch.mean(torch.sum(torch.clamp(vdw, min=-2.0, max=50.0), dim=(-1, -2)))

    def relax_pose(
        self,
        state: CanonicalComplexState,
        init_params: ArticulatedPoseParams,
        max_steps: int = 25,
    ) -> Tuple[torch.Tensor, ArticulatedPoseParams, torch.Tensor]:
        device = state.pocket_atom_coords.device
        T_param = init_params.translation.clone().detach().to(device).requires_grad_(True)
        r6d_param = init_params.rotation_6d.clone().detach().to(device).requires_grad_(True)
        
        M = state.rotatable_edges.shape[0]
        if M > 0 and init_params.torsions_sc is not None:
            init_s = init_params.torsions_sc[..., 0]
            init_c = init_params.torsions_sc[..., 1]
            init_theta = torch.atan2(init_s, init_c)
            theta_param = init_theta.clone().detach().to(device).requires_grad_(True)
            optimizer = torch.optim.LBFGS([T_param, r6d_param, theta_param], lr=0.1, max_iter=max_steps)
        else:
            theta_param = None
            optimizer = torch.optim.LBFGS([T_param, r6d_param], lr=0.1, max_iter=max_steps)
            
        def closure():
            optimizer.zero_grad()
            sc = torch.stack([torch.sin(theta_param), torch.cos(theta_param)], dim=-1) if theta_param is not None else None
            coords = ForwardKinematics.apply(
                ref_coords=state.ligand_ref_coords,
                rotatable_edges=state.rotatable_edges,
                kinematic_masks=state.kinematic_masks,
                translation=T_param,
                rotation_6d=r6d_param,
                torsions_sc=sc
            )
            loss = self.compute_physics_loss(state, coords)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass
            
        with torch.no_grad():
            sc = torch.stack([torch.sin(theta_param), torch.cos(theta_param)], dim=-1) if theta_param is not None else torch.zeros((init_params.translation.shape[0], 0, 2), device=device)
            relaxed_coords = ForwardKinematics.apply(
                ref_coords=state.ligand_ref_coords,
                rotatable_edges=state.rotatable_edges,
                kinematic_masks=state.kinematic_masks,
                translation=T_param,
                rotation_6d=r6d_param,
                torsions_sc=sc if M > 0 else None
            )
            energy = self.compute_physics_loss(state, relaxed_coords).unsqueeze(0)
            relaxed_params = ArticulatedPoseParams(
                translation=T_param.detach(),
                rotation_6d=r6d_param.detach(),
                torsions_sc=sc.detach()
            )
        return relaxed_coords, relaxed_params, energy
