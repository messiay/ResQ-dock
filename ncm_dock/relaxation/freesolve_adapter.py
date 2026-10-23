import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from ncm_dock.contracts import CanonicalComplexState, ArticulatedPoseParams, PhysicsRelaxationProtocol
from ncm_dock.relaxation.base_relaxer import BaseRelaxer
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics, rotation_6d_to_matrix, matrix_to_rotation_6d

class FreeSolvEAdapter(BaseRelaxer):
    """
    Differentiable continuum solvation and C1 logarithmic soft-core Van der Waals engine.
    Supports sub-10ms 3D FFT Poisson electrostatics and manifold articulated pose relaxation.
    """
    def __init__(
        self,
        grid_spacing: float = 1.0,
        epsilon_in: float = 4.0,
        epsilon_water: float = 78.5,
        vdw_e_cap: float = 25.0,
        vdw_s0: float = 10.0,
        vdw_eps_ij: float = 0.15,
        grid_dim: int = 32
    ):
        self.grid_spacing = grid_spacing
        self.epsilon_in = epsilon_in
        self.epsilon_water = epsilon_water
        self.vdw_e_cap = vdw_e_cap
        self.vdw_s0 = vdw_s0
        self.vdw_eps_ij = vdw_eps_ij
        self.grid_dim = grid_dim

    def compute_poisson_grid(
        self,
        pocket_coords: torch.Tensor,   # [N_poc, 3] in canonical frame
        pocket_charges: torch.Tensor,  # [N_poc]
    ) -> torch.Tensor:
        """
        Compute 3D FFT Poisson electrostatic potential Phi(r) on a uniform grid around pocket.
        Returns: [G, G, G] grid in float32.
        """
        device = pocket_coords.device
        G = self.grid_dim
        dx = self.grid_spacing
        half_box = (G * dx) / 2.0
        
        # Grid bounds: [-half_box, +half_box] in each dimension
        # Discretize charges onto grid via trilinear splatting
        grid_charges = torch.zeros((G, G, G), device=device, dtype=torch.float32)
        
        # Shift coords relative to grid center
        norm_coords = (pocket_coords + half_box) / dx # [N, 3] in grid cell units
        valid_mask = (norm_coords[:, 0] >= 0) & (norm_coords[:, 0] < G - 1) & \
                     (norm_coords[:, 1] >= 0) & (norm_coords[:, 1] < G - 1) & \
                     (norm_coords[:, 2] >= 0) & (norm_coords[:, 2] < G - 1)
                     
        nc = norm_coords[valid_mask]
        qc = pocket_charges[valid_mask]
        
        if nc.shape[0] > 0:
            x0 = torch.floor(nc[:, 0]).long()
            y0 = torch.floor(nc[:, 1]).long()
            z0 = torch.floor(nc[:, 2]).long()
            
            fx = nc[:, 0] - x0.float()
            fy = nc[:, 1] - y0.float()
            fz = nc[:, 2] - z0.float()
            
            # 8 corners splat
            for dx_i, wx in [(0, 1.0 - fx), (1, fx)]:
                for dy_i, wy in [(0, 1.0 - fy), (1, fy)]:
                    for dz_i, wz in [(0, 1.0 - fz), (1, fz)]:
                        grid_charges.index_put_(
                            (x0 + dx_i, y0 + dy_i, z0 + dz_i),
                            qc * wx * wy * wz,
                            accumulate=True
                        )
                        
        # Green's function kernel on periodic 3D grid: G(k) = 1 / (eps * (kx^2 + ky^2 + kz^2))
        kx = torch.fft.fftfreq(G, d=dx, device=device) * 2.0 * np.pi
        ky = torch.fft.fftfreq(G, d=dx, device=device) * 2.0 * np.pi
        kz = torch.fft.fftfreq(G, d=dx, device=device) * 2.0 * np.pi
        
        Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_sq = Kx**2 + Ky**2 + Kz**2
        k_sq[0, 0, 0] = 1.0 # Avoid div by zero at k=0
        
        # Distance-dependent screened electrostatic potential:
        # epsilon(k) smoothly transitions from epsilon_in = 4.0 (pocket contact / high k)
        # to epsilon_water = 78.5 (bulk solvent exterior / low k -> 0)
        k_screen = 0.3 # Screening frequency corresponding to ~3.3 Å Debye length
        eps_k = self.epsilon_water - (self.epsilon_water - self.epsilon_in) * (k_sq / (k_sq + k_screen**2))
        
        # Poisson Green's kernel in Fourier space: Phi(k) = 4*pi*rho(k) / (eps(k) * k^2)
        G_k = (4.0 * np.pi) / (eps_k * k_sq)
        G_k[0, 0, 0] = 0.0 # Neutral background at k=0
        
        # 3D FFT convolution
        rho_k = torch.fft.fftn(grid_charges)
        phi_k = rho_k * G_k
        phi_grid = torch.real(torch.fft.ifftn(phi_k))
        return phi_grid

    def interpolate_potential(
        self,
        grid: torch.Tensor,        # [G, G, G]
        coords: torch.Tensor,      # [B, N, 3] or [N, 3]
    ) -> torch.Tensor:
        """
        Differentiable trilinear grid interpolation using F.grid_sample.
        Returns: [B, N] potential values.
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        B, N, _ = coords.shape
        G = grid.shape[0]
        dx = self.grid_spacing
        half_box = (G * dx) / 2.0
        
        # Normalize coords to [-1, 1] for grid_sample: shape [B, 1, 1, N, 3]
        # grid_sample expects (x, y, z) in [-1, 1] corresponding to grid dimensions (D, H, W)
        norm_c = torch.clamp(coords / half_box, -1.0, 1.0)
        grid_input = norm_c.view(B, 1, 1, N, 3)
        
        # Grid shape for grid_sample: [B, C, D, H, W] = [B, 1, G, G, G]
        grid_5d = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, G, G, G)
        
        # Trilinear sampling
        sampled = F.grid_sample(grid_5d, grid_input, mode='bilinear', padding_mode='zeros', align_corners=True)
        return sampled.view(B, N)

    def soft_core_vdw(
        self,
        lig_coords: torch.Tensor,   # [B, N_lig, 3]
        lig_vdw: torch.Tensor,      # [N_lig]
        poc_coords: torch.Tensor,   # [N_poc, 3]
        poc_vdw: torch.Tensor,      # [N_poc]
    ) -> torch.Tensor:
        """
        C1-Continuous Logarithmic Soft-Core Van der Waals Potential.
        Prevents gradient explosion when r_ij -> 0.
        Returns: [B] energy in kcal/mol.
        """
        B, N_lig, _ = lig_coords.shape
        N_poc = poc_coords.shape[0]
        device = lig_coords.device
        
        # Pairwise distances: [B, N_lig, N_poc]
        # || lig_i - poc_j ||
        diff = lig_coords.unsqueeze(2) - poc_coords.unsqueeze(0).unsqueeze(0) # [B, N_lig, N_poc, 3]
        dist_sq = torch.sum(diff ** 2, dim=-1) # [B, N_lig, N_poc]
        dist = torch.sqrt(dist_sq + 1e-6)
        
        # Sigma_ij = sum of VDW radii
        sigma_ij = (lig_vdw.unsqueeze(1) + poc_vdw.unsqueeze(0)).unsqueeze(0) # [1, N_lig, N_poc]
        
        # Standard Lennard-Jones: 4 * eps * [ (sigma/r)^12 - (sigma/r)^6 ]
        ratio = sigma_ij / dist
        ratio_6 = torch.clamp(ratio ** 6, max=1e4)
        ratio_12 = ratio_6 ** 2
        e0 = 4.0 * self.vdw_eps_ij * (ratio_12 - ratio_6) # [B, N_lig, N_poc]
        
        # C1 logarithmic soft cap
        E_cap = self.vdw_e_cap
        s0 = self.vdw_s0
        
        is_high = e0 > E_cap
        e_soft = torch.where(
            is_high,
            E_cap + s0 * torch.log(1.0 + torch.clamp((e0 - E_cap) / s0, min=0.0)),
            e0
        )
        
        # Sum over pairs: [B]
        vdw_energy = torch.sum(e_soft, dim=(-1, -2))
        return vdw_energy

    def compute_physics_loss(
        self,
        state: CanonicalComplexState,
        pred_coords: torch.Tensor, # [B, N_lig, 3] or [N_lig, 3]
    ) -> torch.Tensor:
        """
        Evaluates composite biophysical loss: Soft-core VdW + Poisson electrostatics.
        Returns scalar loss tensor [1] with full autograd tracking.
        """
        if pred_coords.dim() == 2:
            pred_coords = pred_coords.unsqueeze(0)
        B = pred_coords.shape[0]
        device = pred_coords.device
        
        # 1. Soft-core VdW clash penalty with pocket atoms
        poc_coords = state.pocket_atom_coords.to(device=device, dtype=torch.float32)
        poc_vdw = state.pocket_atom_vdw.to(device=device, dtype=torch.float32)
        lig_vdw = state.ligand_atom_vdw.to(device=device, dtype=torch.float32)
        
        vdw_energy = self.soft_core_vdw(pred_coords, lig_vdw, poc_coords, poc_vdw) # [B]
        
        # 2. Electrostatic potential from Poisson grid
        if state.freesolve_fft_grid is not None:
            phi_grid = state.freesolve_fft_grid.to(device=device, dtype=torch.float32)
        else:
            poc_chg = state.pocket_atom_charges.to(device=device, dtype=torch.float32)
            phi_grid = self.compute_poisson_grid(poc_coords, poc_chg)
            
        phi_sampled = self.interpolate_potential(phi_grid, pred_coords) # [B, N_lig]
        lig_chg = state.ligand_atom_charges.to(device=device, dtype=torch.float32).unsqueeze(0) # [1, N_lig]
        elec_energy = torch.sum(phi_sampled * lig_chg, dim=-1) # [B]
        
        total_energy = torch.mean(vdw_energy + 0.1 * elec_energy)
        return total_energy

    def relax_pose(
        self,
        state: CanonicalComplexState,
        init_params: ArticulatedPoseParams,
        max_steps: int = 40,
        lr: float = 0.05
    ) -> Tuple[torch.Tensor, ArticulatedPoseParams, torch.Tensor]:
        """
        Refines pose on the exact articulated manifold (T, r6d, theta) using Adam optimizer.
        Returns: (relaxed_coords [B, N, 3], relaxed_params, final_energies [B])
        """
        device = state.pocket_atom_coords.device
        T_param = init_params.translation.clone().detach().to(device).requires_grad_(True)
        r6d_param = init_params.rotation_6d.clone().detach().to(device).requires_grad_(True)
        
        params_to_opt = [T_param, r6d_param]
        M = state.rotatable_edges.shape[0]
        if M > 0 and init_params.torsions_sc is not None:
            # Optimize angles theta directly in R^M to guarantee continuous exploration
            init_s = init_params.torsions_sc[..., 0]
            init_c = init_params.torsions_sc[..., 1]
            init_theta = torch.atan2(init_s, init_c)
            theta_param = init_theta.clone().detach().to(device).requires_grad_(True)
            params_to_opt.append(theta_param)
        else:
            theta_param = None
            
        optimizer = torch.optim.Adam(params_to_opt, lr=lr)
        
        # Evaluate initial baseline energy
        with torch.no_grad():
            init_coords = ForwardKinematics.apply(
                ref_coords=state.ligand_ref_coords,
                rotatable_edges=state.rotatable_edges,
                kinematic_masks=state.kinematic_masks,
                translation=init_params.translation,
                rotation_6d=init_params.rotation_6d,
                torsions_sc=init_params.torsions_sc if M > 0 else None
            )
            best_loss = self.compute_physics_loss(state, init_coords).item()
            best_T = init_params.translation.clone()
            best_r6d = init_params.rotation_6d.clone()
            best_theta = theta_param.clone() if theta_param is not None else None

        for step in range(max_steps):
            optimizer.zero_grad()
            if theta_param is not None:
                sc = torch.stack([torch.sin(theta_param), torch.cos(theta_param)], dim=-1)
            else:
                sc = None
                
            pred_coords = ForwardKinematics.apply(
                ref_coords=state.ligand_ref_coords,
                rotatable_edges=state.rotatable_edges,
                kinematic_masks=state.kinematic_masks,
                translation=T_param,
                rotation_6d=r6d_param,
                torsions_sc=sc
            )
            
            loss = self.compute_physics_loss(state, pred_coords)
            l_val = loss.item()
            if l_val < best_loss:
                best_loss = l_val
                best_T = T_param.detach().clone()
                best_r6d = r6d_param.detach().clone()
                if theta_param is not None:
                    best_theta = theta_param.detach().clone()
                    
            loss.backward()
            optimizer.step()
            
        # Final evaluation using best energy parameters
        with torch.no_grad():
            if best_theta is not None:
                final_sc = torch.stack([torch.sin(best_theta), torch.cos(best_theta)], dim=-1)
            else:
                final_sc = torch.zeros((init_params.translation.shape[0], 0, 2), device=device)
                
            relaxed_coords = ForwardKinematics.apply(
                ref_coords=state.ligand_ref_coords,
                rotatable_edges=state.rotatable_edges,
                kinematic_masks=state.kinematic_masks,
                translation=best_T,
                rotation_6d=best_r6d,
                torsions_sc=final_sc if M > 0 else None
            )
            
            relaxed_params = ArticulatedPoseParams(
                translation=best_T,
                rotation_6d=best_r6d,
                torsions_sc=final_sc
            )
            
            # Compute final energy per batch element
            poc_coords = state.pocket_atom_coords.to(device=device, dtype=torch.float32)
            poc_vdw = state.pocket_atom_vdw.to(device=device, dtype=torch.float32)
            lig_vdw = state.ligand_atom_vdw.to(device=device, dtype=torch.float32)
            vdw_energy = self.soft_core_vdw(relaxed_coords, lig_vdw, poc_coords, poc_vdw)
            
        return relaxed_coords, relaxed_params, vdw_energy
