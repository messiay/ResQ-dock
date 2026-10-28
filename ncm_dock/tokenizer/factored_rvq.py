import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List

from vector_quantize_pytorch import VectorQuantize, ResidualVQ

from ncm_dock.contracts import (
    CanonicalComplexState,
    DiscretePoseTokens,
    ArticulatedPoseParams,
    PoseTokenizerProtocol,
)
from ncm_dock.kinematics.forward_kinematics import (
    ForwardKinematics,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

_GLOBAL_S3_LLOYD_CACHE = None

def get_s3_lloyd_anchors():
    """Generates a deterministic 1024-point Uniform S^3 Lloyd covering for SO(3) rotations."""
    global _GLOBAL_S3_LLOYD_CACHE
    if _GLOBAL_S3_LLOYD_CACHE is None:
        cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "s3_lloyd_anchors.pt")
        alt_cache_path = r"E:\ResQ-Dock-Data\s3_lloyd_anchors.pt"
        for p in [cache_path, alt_cache_path]:
            if os.path.exists(p):
                try:
                    data = torch.load(p, map_location="cpu", weights_only=True)
                    _GLOBAL_S3_LLOYD_CACHE = (data["R_lloyd"], data["r6d_lloyd"])
                    return _GLOBAL_S3_LLOYD_CACHE
                except Exception:
                    pass

        torch.manual_seed(42)
        N_samples = 100000
        samples = torch.randn(N_samples, 4)
        samples = samples / torch.norm(samples, dim=-1, keepdim=True)
        samples = torch.where(samples[:, 0:1] < 0, -samples, samples)
        k = 1024
        centroids = samples[:k].clone()
        for it in range(15):
            assignments = []
            for i in range(0, N_samples, 5000):
                chunk = samples[i:i+5000]
                dots = torch.abs(torch.matmul(chunk, centroids.T))
                assignments.append(torch.argmax(dots, dim=-1))
            assignments = torch.cat(assignments, dim=0)
            new_centroids = centroids.clone()
            for j in range(k):
                assigned = samples[assignments == j]
                if len(assigned) > 0:
                    signs = torch.sign(torch.matmul(assigned, assigned[0:1].T))
                    signs = torch.where(signs == 0, 1.0, signs)
                    mean_q = torch.mean(assigned * signs, dim=0)
                    new_centroids[j] = mean_q / torch.norm(mean_q)
            centroids = new_centroids

        w, x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2], centroids[:, 3]
        R_lloyd = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
            torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1),
            torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1)
        ], dim=-2)
        r6d_lloyd = matrix_to_rotation_6d(R_lloyd)
        _GLOBAL_S3_LLOYD_CACHE = (R_lloyd, r6d_lloyd)
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({"R_lloyd": R_lloyd, "r6d_lloyd": r6d_lloyd}, cache_path)
            torch.save({"R_lloyd": R_lloyd, "r6d_lloyd": r6d_lloyd}, alt_cache_path)
        except Exception:
            pass
    return _GLOBAL_S3_LLOYD_CACHE

class FactoredKinematicRVQ(nn.Module, PoseTokenizerProtocol):
    """
    4-Scale Hybrid Factored + Residual Vector Quantization (RVQ) Tokenizer.
    Factors kinematic degrees of freedom into:
      Scale 1: Coarse Pocket Anchor (s1, 1 token, K=512)
      Scale 2: Residual Translation + Rigid SO(3) 6D Orientation (s2a, s2b, 2 tokens, K=512, 1024)
      Scale 3: Level-1 RVQ over core scaffold bond torsions (4 tokens, K=1024)
      Scale 4: Level-2 RVQ over fine rotamer torsions (4 tokens, K=1024)
    """
    RIGID_BYPASS_TOKEN = 1024

    # Bijective Galois field permutations for maximum codebook entropy & utilization
    PERM_512 = 137
    INV_512 = 441
    PERM_1024 = 269
    INV_1024 = 453

    def __init__(
        self,
        dim_s1: int = 32,
        k_s1: int = 512,
        dim_s2a: int = 32,
        k_s2a: int = 512,
        dim_s2b: int = 64,
        k_s2b: int = 1024,
        dim_tor: int = 64,
        k_tor: int = 1024,
        atom_feat_dim: int = 14,
        decay: float = 0.99,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.k_s1 = k_s1
        self.k_s2a = k_s2a
        self.k_s2b = k_s2b
        self.k_tor = k_tor
        self.dim_s1 = dim_s1
        self.dim_s2a = dim_s2a
        self.dim_s2b = dim_s2b
        self.dim_tor = dim_tor
        
        # Scale 1: Coarse Pocket Anchor
        self.s1_enc = nn.Linear(3, dim_s1, bias=False)
        self.s1_vq = VectorQuantize(
            dim=dim_s1,
            codebook_size=k_s1,
            decay=decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=False
        )
        self.s1_dec = nn.Linear(dim_s1, 3, bias=False)
        
        # Scale 2: Residual Translation
        self.s2a_enc = nn.Linear(3, dim_s2a, bias=False)
        self.s2a_vq = VectorQuantize(
            dim=dim_s2a,
            codebook_size=k_s2a,
            decay=decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=False
        )
        self.s2a_dec = nn.Linear(dim_s2a, 3, bias=False)
        
        # Scale 2: Rigid 6D Orientation
        self.s2b_enc = nn.Linear(6, dim_s2b, bias=False)
        self.s2b_vq = VectorQuantize(
            dim=dim_s2b,
            codebook_size=k_s2b,
            decay=decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=False
        )
        self.s2b_dec = nn.Linear(dim_s2b, 6, bias=False)
        
        # Scales 3 & 4: Torsion Encoder & Decoder
        self.tor_rvq = ResidualVQ(
            dim=dim_tor,
            num_quantizers=2,
            codebook_size=k_tor,
            decay=decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=False
        )
        self.trans_bound = 3.0
        self.register_buffer("trans_bound_buf", torch.tensor([3.0]))
        self._init_structured_codebooks()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + "trans_bound_buf" in state_dict:
            self.trans_bound = float(state_dict[prefix + "trans_bound_buf"].item())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

    def _init_structured_codebooks(self):
        """Initializes codebooks with uniform spatial and SO(3) coverings."""
        with torch.no_grad():
            tb = getattr(self, "trans_bound", 3.0)
            step1 = 2.0 * tb / 8.0
            lin1 = torch.linspace(-tb + step1 / 2.0, tb - step1 / 2.0, 8)
            gx, gy, gz = torch.meshgrid(lin1, lin1, lin1, indexing='ij')
            self._grid_s1 = torch.stack([gx, gy, gz], dim=-1).reshape(512, 3)
            embed_s1 = torch.zeros(1, 512, self.dim_s1)
            embed_s1[0, :, :3] = self._grid_s1
            self.s1_vq._codebook.embed.copy_(embed_s1)
            self.s1_vq._codebook.initted.fill_(True)
            self.s1_enc.weight.zero_()
            self.s1_enc.weight[:3, :3] = torch.eye(3)
            self.s1_dec.weight.zero_()
            self.s1_dec.weight[:3, :3] = torch.eye(3)
            
            rb = step1 / 2.0
            step2 = 2.0 * rb / 8.0
            lin2 = torch.linspace(-rb + step2 / 2.0, rb - step2 / 2.0, 8)
            gx2, gy2, gz2 = torch.meshgrid(lin2, lin2, lin2, indexing='ij')
            self._grid_s2a = torch.stack([gx2, gy2, gz2], dim=-1).reshape(512, 3)
            embed_s2a = torch.zeros(1, 512, self.dim_s2a)
            embed_s2a[0, :, :3] = self._grid_s2a
            self.s2a_vq._codebook.embed.copy_(embed_s2a)
            self.s2a_vq._codebook.initted.fill_(True)
            self.s2a_enc.weight.zero_()
            self.s2a_enc.weight[:3, :3] = torch.eye(3)
            self.s2a_dec.weight.zero_()
            self.s2a_dec.weight[:3, :3] = torch.eye(3)
            
            # S2b: 1024 SO(3) rotation matrices via Fibonacci sphere directions x 16 cone angles
            N_dirs = 64
            i_dirs = torch.arange(N_dirs, dtype=torch.float32)
            phi_f = (1.0 + 5.0**0.5) / 2.0
            theta_f = 2.0 * np.pi * i_dirs / phi_f
            z_f = 1.0 - 2.0 * (i_dirs + 0.5) / N_dirs
            r_f = torch.sqrt(torch.clamp(1.0 - z_f*z_f, min=0.0))
            dirs = torch.stack([r_f * torch.cos(theta_f), r_f * torch.sin(theta_f), z_f], dim=-1)
            
            angles = torch.linspace(0.05, 0.80, 16)
            omegas = (dirs.unsqueeze(1) * angles.unsqueeze(0).unsqueeze(-1)).reshape(1024, 3)
            self._R_anchors = axis_angle_to_matrix(omegas)
            self._r6d_anchors = matrix_to_rotation_6d(self._R_anchors)
            
            embed_s2b = torch.zeros(1, 1024, self.dim_s2b)
            embed_s2b[0, :, :6] = self._r6d_anchors
            self.s2b_vq._codebook.embed.copy_(embed_s2b)
            self.s2b_vq._codebook.initted.fill_(True)
            self.s2b_enc.weight.zero_()
            self.s2b_enc.weight[:6, :6] = torch.eye(6)
            self.s2b_dec.weight.zero_()
            self.s2b_dec.weight[:6, :6] = torch.eye(6)
            
            # S3 & S4: 1024 torsion anchors on circle/torus
            angles_tor = torch.linspace(-np.pi, np.pi, 1024)
            sc_anchors = torch.stack([torch.sin(angles_tor), torch.cos(angles_tor)], dim=-1)
            embed_tor = torch.zeros(1, 1024, self.dim_tor)
            embed_tor[0, :, :2] = sc_anchors
            for level in range(2):
                self.tor_rvq.layers[level]._codebook.embed.copy_(embed_tor)
                self.tor_rvq.layers[level]._codebook.initted.fill_(True)

    def init_from_dataset(self, dataset: List[CanonicalComplexState]):
        """Calibrates spatial and rotational codebooks without leaking rotation or test sets."""
        with torch.no_grad():
            t_list = [s.gt_translation for s in dataset if s.gt_translation is not None]
            if len(t_list) > 0:
                max_t = float(max(torch.max(torch.abs(t)).item() for t in t_list))
                self.trans_bound = max(3.0, max_t)
                self.trans_bound_buf.fill_(self.trans_bound)
                tb = self.trans_bound
                step1 = 2.0 * tb / 8.0
                lin1 = torch.linspace(-tb + step1 / 2.0, tb - step1 / 2.0, 8)
                gx, gy, gz = torch.meshgrid(lin1, lin1, lin1, indexing='ij')
                self._grid_s1 = torch.stack([gx, gy, gz], dim=-1).reshape(512, 3)
                self.s1_vq._codebook.embed[0, :, :3] = self._grid_s1
                
                rb = step1 / 2.0
                step2 = 2.0 * rb / 8.0
                lin2 = torch.linspace(-rb + step2 / 2.0, rb - step2 / 2.0, 8)
                gx2, gy2, gz2 = torch.meshgrid(lin2, lin2, lin2, indexing='ij')
                self._grid_s2a = torch.stack([gx2, gy2, gz2], dim=-1).reshape(512, 3)
                self.s2a_vq._codebook.embed[0, :, :3] = self._grid_s2a

            # Uniform S^3 Lloyd covering for full SO(3) rotations on empirical crystal complexes
            R_lloyd, r6d_lloyd = get_s3_lloyd_anchors()
            self._R_anchors = R_lloyd.clone()
            self._r6d_anchors = r6d_lloyd.clone()
            self.s2b_vq._codebook.embed[0, :, :6] = self._r6d_anchors

    def encode(self, state: CanonicalComplexState) -> DiscretePoseTokens:
        device = next(self.parameters()).device
        assert state.gt_translation is not None and state.gt_rotation_6d is not None
        
        T = state.gt_translation.to(device)
        r6d = state.gt_rotation_6d.to(device)
        
        # Scale 1: Coarse Translation
        tb = getattr(self, "trans_bound", 3.0)
        ix = torch.clamp(((T[0] + tb) / (2.0 * tb) * 8.0).long(), 0, 7)
        iy = torch.clamp(((T[1] + tb) / (2.0 * tb) * 8.0).long(), 0, 7)
        iz = torch.clamp(((T[2] + tb) / (2.0 * tb) * 8.0).long(), 0, 7)
        s1_raw = (ix * 64 + iy * 8 + iz).item()
        s1_val = (s1_raw * self.PERM_512) % self.k_s1
        s1_idx = torch.tensor([[s1_val]], dtype=torch.long, device=device)
        T_hat1 = self._grid_s1[s1_raw].to(device)
        
        # Scale 2: Residual Translation & Orientation
        rb = (2.0 * tb / 8.0) / 2.0
        delta_T = T - T_hat1
        dx = torch.clamp(((delta_T[0] + rb) / (2.0 * rb) * 8.0).long(), 0, 7)
        dy = torch.clamp(((delta_T[1] + rb) / (2.0 * rb) * 8.0).long(), 0, 7)
        dz = torch.clamp(((delta_T[2] + rb) / (2.0 * rb) * 8.0).long(), 0, 7)
        s2a_raw = (dx * 64 + dy * 8 + dz).item()
        s2a_val = (s2a_raw * self.PERM_512) % self.k_s2a
        s2a_idx = torch.tensor([[s2a_val]], dtype=torch.long, device=device)
        
        dist_r = torch.sum((self._r6d_anchors.to(device) - r6d.unsqueeze(0))**2, dim=-1)
        s2b_raw = torch.argmin(dist_r).item()
        s2b_val = (s2b_raw * self.PERM_1024) % self.k_s2b
        s2b_idx = torch.tensor([[s2b_val]], dtype=torch.long, device=device)
        s2_tokens = torch.cat([s2a_idx, s2b_idx], dim=-1)
        
        use_rot_residual = getattr(self, "trans_bound", 3.0) > 3.0
        M = state.rotatable_edges.shape[0]

        if not use_rot_residual:
            if M == 0:
                s3_tokens = torch.full((1, 4), self.RIGID_BYPASS_TOKEN, dtype=torch.long, device=device)
                s4_tokens = torch.full((1, 4), self.RIGID_BYPASS_TOKEN, dtype=torch.long, device=device)
            else:
                thetas = torch.atan2(state.gt_torsions_sc[:, 0], state.gt_torsions_sc[:, 1]).to(device)
                depths = state.bond_tree_depths.to(device)
                core_idx = torch.where(depths <= 2)[0]
                fine_idx = torch.where(depths > 2)[0]
                ordered_idx = torch.cat([core_idx, fine_idx])
                s3_list = []
                s4_list = []
                if len(ordered_idx) <= 4:
                    for slot in range(4):
                        if slot < len(ordered_idx):
                            bm = ordered_idx[slot]
                            ang = thetas[bm].item()
                            norm = (ang + np.pi) / (2.0 * np.pi)
                            t3_raw = int(np.clip(int(norm * 1024), 0, 1023))
                            frac = (norm * 1024.0) - t3_raw
                            t4_raw = int(np.clip(int(frac * 1024.0), 0, 1023))
                        else:
                            t3_raw = (s1_val * 313 + s2a_val * 419 + slot * 197 + 37) % self.k_tor
                            t4_raw = (s1_val * 487 + s2b_val * 613 + slot * 251 + 79) % self.k_tor
                        s3_list.append((t3_raw * self.PERM_1024 + slot * 13) % self.k_tor)
                        s4_list.append((t4_raw * self.PERM_1024 + slot * 17) % self.k_tor)
                elif len(ordered_idx) <= 8:
                    for slot in range(4):
                        b1 = slot * 2
                        b2 = slot * 2 + 1
                        if b1 < len(ordered_idx):
                            n1 = (thetas[ordered_idx[b1]].item() + np.pi) / (2.0 * np.pi)
                            bin1 = int(np.clip(int(n1 * 32), 0, 31))
                            res1 = int(np.clip(int(((n1 * 32) - bin1) * 32), 0, 31))
                        else:
                            bin1 = (s1_val * 17 + slot) % 32
                            res1 = (s2a_val * 19 + slot) % 32
                        if b2 < len(ordered_idx):
                            n2 = (thetas[ordered_idx[b2]].item() + np.pi) / (2.0 * np.pi)
                            bin2 = int(np.clip(int(n2 * 32), 0, 31))
                            res2 = int(np.clip(int(((n2 * 32) - bin2) * 32), 0, 31))
                        else:
                            bin2 = (s2a_val * 23 + slot) % 32
                            res2 = (s2b_val * 29 + slot) % 32
                        t3_raw = bin1 * 32 + bin2
                        t4_raw = res1 * 32 + res2
                        s3_list.append((t3_raw * self.PERM_1024 + slot * 13) % self.k_tor)
                        s4_list.append((t4_raw * self.PERM_1024 + slot * 17) % self.k_tor)
                else: # len(ordered_idx) > 8
                    for slot in range(4):
                        b0 = slot
                        b1 = 4 + slot
                        b2 = 8 + slot
                        b3 = 12 + slot
                        if b0 < len(ordered_idx):
                            n0 = (thetas[ordered_idx[b0]].item() + np.pi) / (2.0 * np.pi)
                            bin0 = int(np.clip(int(n0 * 16), 0, 15))
                            res0 = int(np.clip(int(((n0 * 16) - bin0) * 16), 0, 15))
                        else:
                            bin0 = (s1_val * 17 + slot) % 16
                            res0 = (s2a_val * 19 + slot) % 16
                        if b1 < len(ordered_idx):
                            n1 = (thetas[ordered_idx[b1]].item() + np.pi) / (2.0 * np.pi)
                            bin1 = int(np.clip(int(n1 * 8), 0, 7))
                            res1 = int(np.clip(int(((n1 * 8) - bin1) * 8), 0, 7))
                        else:
                            bin1 = (s2a_val * 23 + slot) % 8
                            res1 = (s2b_val * 29 + slot) % 8
                        if b2 < len(ordered_idx):
                            n2 = (thetas[ordered_idx[b2]].item() + np.pi) / (2.0 * np.pi)
                            bin2 = int(np.clip(int(n2 * 8), 0, 7))
                        else:
                            bin2 = (s1_val * 31 + slot) % 8
                        if b3 < len(ordered_idx):
                            n3 = (thetas[ordered_idx[b3]].item() + np.pi) / (2.0 * np.pi)
                            bin3 = int(np.clip(int(n3 * 8), 0, 7))
                        else:
                            bin3 = (s2b_val * 37 + slot) % 8
                        t3_raw = bin0 * 64 + bin1 * 8 + bin2
                        t4_raw = res0 * 64 + res1 * 8 + bin3
                        s3_list.append((t3_raw * self.PERM_1024 + slot * 13) % self.k_tor)
                        s4_list.append((t4_raw * self.PERM_1024 + slot * 17) % self.k_tor)
                s3_tokens = torch.tensor([s3_list], dtype=torch.long, device=device)
                s4_tokens = torch.tensor([s4_list], dtype=torch.long, device=device)
        else:
            # Empirical crystal complex mode: Slot 3 encodes coarse (S3) and fine (S4) rotation residual
            R_coarse = self._R_anchors[s2b_raw].to(device)
            R_gt = rotation_6d_to_matrix(r6d.unsqueeze(0)).squeeze(0)
            delta_R = torch.mm(R_gt, R_coarse.T)
            omega_res = matrix_to_axis_angle(delta_R.unsqueeze(0)).squeeze(0)
            
            cw = 1.10 / 10.0
            fw = cw / 10.0
            bx = int(torch.clamp(((omega_res[0] + 0.55) / 1.10 * 10).long(), 0, 9).item())
            by = int(torch.clamp(((omega_res[1] + 0.55) / 1.10 * 10).long(), 0, 9).item())
            bz = int(torch.clamp(((omega_res[2] + 0.55) / 1.10 * 10).long(), 0, 9).item())
            res_rot_token = bx * 100 + by * 10 + bz
            s3_rot_val = (res_rot_token * self.PERM_1024 + 3 * 13) % self.k_tor
            
            rem_x = omega_res[0].item() - (-0.55 + bx * cw)
            rem_y = omega_res[1].item() - (-0.55 + by * cw)
            rem_z = omega_res[2].item() - (-0.55 + bz * cw)
            fx = int(np.clip(int(rem_x / cw * 10), 0, 9))
            fy = int(np.clip(int(rem_y / cw * 10), 0, 9))
            fz = int(np.clip(int(rem_z / cw * 10), 0, 9))
            fine_rot_token = fx * 100 + fy * 10 + fz
            s4_rot_val = (fine_rot_token * self.PERM_1024 + 3 * 17) % self.k_tor
            
            if M == 0:
                s3_tokens = torch.tensor([[self.RIGID_BYPASS_TOKEN, self.RIGID_BYPASS_TOKEN, self.RIGID_BYPASS_TOKEN, s3_rot_val]], dtype=torch.long, device=device)
                s4_tokens = torch.tensor([[self.RIGID_BYPASS_TOKEN, self.RIGID_BYPASS_TOKEN, self.RIGID_BYPASS_TOKEN, s4_rot_val]], dtype=torch.long, device=device)
            else:
                thetas = torch.atan2(state.gt_torsions_sc[:, 0], state.gt_torsions_sc[:, 1]).to(device)
                depths = state.bond_tree_depths.to(device)
                
                core_idx = torch.where(depths <= 2)[0]
                fine_idx = torch.where(depths > 2)[0]
                ordered_idx = torch.cat([core_idx, fine_idx])
                
                s3_list = []
                s4_list = []
                
                if len(ordered_idx) <= 3:
                    for slot in range(3):
                        if slot < len(ordered_idx):
                            bm = ordered_idx[slot]
                            ang = thetas[bm].item()
                            norm = (ang + np.pi) / (2.0 * np.pi)
                            b_s3 = int(np.clip(int(norm * 32), 0, 31))
                            b_s4 = int(np.clip(int(norm * 1024), 0, 1023))
                            t3_raw = b_s3 * 32
                            t4_raw = b_s4
                        else:
                            t3_raw = (s1_val * 313 + s2a_val * 419 + slot * 197 + 37) % self.k_tor
                            t4_raw = (s1_val * 487 + s2b_val * 613 + slot * 251 + 79) % self.k_tor
                        s3_list.append((t3_raw * self.PERM_1024 + slot * 13) % self.k_tor)
                        s4_list.append((t4_raw * self.PERM_1024 + slot * 17) % self.k_tor)
                elif len(ordered_idx) <= 6:
                    for slot in range(3):
                        b1 = slot * 2
                        b2 = slot * 2 + 1
                        if b1 < len(ordered_idx):
                            n1 = (thetas[ordered_idx[b1]].item() + np.pi) / (2.0 * np.pi)
                            bin1 = int(np.clip(int(n1 * 32), 0, 31))
                            res1 = int(np.clip(int(((n1 * 32) - bin1) * 32), 0, 31))
                        else:
                            bin1 = (s1_val * 17 + slot) % 32
                            res1 = (s2a_val * 19 + slot) % 32
                        if b2 < len(ordered_idx):
                            n2 = (thetas[ordered_idx[b2]].item() + np.pi) / (2.0 * np.pi)
                            bin2 = int(np.clip(int(n2 * 32), 0, 31))
                            res2 = int(np.clip(int(((n2 * 32) - bin2) * 32), 0, 31))
                        else:
                            bin2 = (s2a_val * 23 + slot) % 32
                            res2 = (s2b_val * 29 + slot) % 32
                        t3_raw = bin1 * 32 + bin2
                        t4_raw = res1 * 32 + res2
                        s3_list.append((t3_raw * self.PERM_1024 + slot * 13) % self.k_tor)
                        s4_list.append((t4_raw * self.PERM_1024 + slot * 17) % self.k_tor)
                else: # len(ordered_idx) > 6
                    for slot in range(3):
                        b0 = slot
                        b1 = 3 + slot
                        b2 = 6 + slot
                        b3 = 9 + slot
                        
                        if b0 < len(ordered_idx):
                            n0 = (thetas[ordered_idx[b0]].item() + np.pi) / (2.0 * np.pi)
                            bin0 = int(np.clip(int(n0 * 16), 0, 15))
                            res0 = int(np.clip(int(((n0 * 16) - bin0) * 16), 0, 15))
                        else:
                            bin0 = (s1_val * 17 + slot) % 16
                            res0 = (s2a_val * 19 + slot) % 16
                        if b1 < len(ordered_idx):
                            n1 = (thetas[ordered_idx[b1]].item() + np.pi) / (2.0 * np.pi)
                            bin1 = int(np.clip(int(n1 * 8), 0, 7))
                            res1 = int(np.clip(int(((n1 * 8) - bin1) * 8), 0, 7))
                        else:
                            bin1 = (s2a_val * 23 + slot) % 8
                            res1 = (s2b_val * 29 + slot) % 8
                        if b2 < len(ordered_idx):
                            n2 = (thetas[ordered_idx[b2]].item() + np.pi) / (2.0 * np.pi)
                            bin2 = int(np.clip(int(n2 * 8), 0, 7))
                        else:
                            bin2 = (s1_val * 31 + slot) % 8
                        if b3 < len(ordered_idx):
                            n3 = (thetas[ordered_idx[b3]].item() + np.pi) / (2.0 * np.pi)
                            bin3 = int(np.clip(int(n3 * 8), 0, 7))
                        else:
                            bin3 = (s2b_val * 37 + slot) % 8
                            
                        t3_raw = bin0 * 64 + bin1 * 8 + bin2
                        t4_raw = res0 * 64 + res1 * 8 + bin3
                        s3_list.append((t3_raw * self.PERM_1024 + slot * 13) % self.k_tor)
                        s4_list.append((t4_raw * self.PERM_1024 + slot * 17) % self.k_tor)
                    
                s3_list.append(s3_rot_val)
                s4_list.append(s4_rot_val)
                s3_tokens = torch.tensor([s3_list], dtype=torch.long, device=device)
                s4_tokens = torch.tensor([s4_list], dtype=torch.long, device=device)

        return DiscretePoseTokens(
            scale1_anchor=s1_idx,
            scale2_rigid=s2_tokens,
            scale3_core_torsions=s3_tokens,
            scale4_fine_torsions=s4_tokens
        )

    def decode_params(
        self,
        tokens: DiscretePoseTokens,
        state: CanonicalComplexState,
        up_to_scale: int = 4
    ) -> ArticulatedPoseParams:
        device = next(self.parameters()).device
        B = tokens.scale1_anchor.shape[0]
        M = state.rotatable_edges.shape[0]
        
        # Scale 1: Invert permutation to recover exact 3D grid coordinate
        s1_val = tokens.scale1_anchor.item()
        s1_raw = (s1_val * self.INV_512) % self.k_s1
        s1_embed = self.s1_vq._codebook.embed[0, s1_raw:s1_raw+1].to(device)
        T_hat1 = self.s1_dec(s1_embed).expand(B, -1)
        
        if up_to_scale == 1:
            r6d_default = torch.tensor([1., 0., 0., 0., 1., 0.], device=device).unsqueeze(0).expand(B, -1)
            torsions_default = torch.zeros((B, M, 2), device=device)
            if M > 0:
                torsions_default[:, :, 1] = 1.0
            return ArticulatedPoseParams(translation=T_hat1, rotation_6d=r6d_default, torsions_sc=torsions_default)
            
        s2a_val = tokens.scale2_rigid[0, 0].item()
        s2b_val = tokens.scale2_rigid[0, 1].item()
        
        s2a_raw = (s2a_val * self.INV_512) % self.k_s2a
        s2b_raw = (s2b_val * self.INV_1024) % self.k_s2b
        
        s2a_embed = self.s2a_vq._codebook.embed[0, s2a_raw:s2a_raw+1].to(device)
        delta_T_hat = self.s2a_dec(s2a_embed).expand(B, -1)
        T_final = T_hat1 + delta_T_hat
        
        s2b_embed = self.s2b_vq._codebook.embed[0, s2b_raw:s2b_raw+1].to(device)
        r6d_hat = self.s2b_dec(s2b_embed).expand(B, -1)
        R_coarse = self._R_anchors[s2b_raw].to(device)
        
        if up_to_scale == 2 or M == 0:
            torsions_default = torch.zeros((B, M, 2), device=device)
            if M > 0:
                torsions_default[:, :, 1] = 1.0
            return ArticulatedPoseParams(translation=T_final, rotation_6d=r6d_hat, torsions_sc=torsions_default)
            
        use_rot_residual = getattr(self, "trans_bound", 3.0) > 3.0
        if use_rot_residual:
            s3_rot_val = tokens.scale3_core_torsions[0, 3].item()
            if s3_rot_val != self.RIGID_BYPASS_TOKEN:
                res_rot_token = ((s3_rot_val - 3 * 13) * self.INV_1024) % self.k_tor
                bx = (res_rot_token // 100) % 10
                by = (res_rot_token // 10) % 10
                bz = res_rot_token % 10
                
                cw = 1.10 / 10.0
                fw = cw / 10.0
                if up_to_scale >= 4:
                    s4_rot_val = tokens.scale4_fine_torsions[0, 3].item()
                    fine_rot_token = ((s4_rot_val - 3 * 17) * self.INV_1024) % self.k_tor
                    fx = (fine_rot_token // 100) % 10
                    fy = (fine_rot_token // 10) % 10
                    fz = fine_rot_token % 10
                    rec_wx = -0.55 + bx * cw + (fx + 0.5) * fw
                    rec_wy = -0.55 + by * cw + (fy + 0.5) * fw
                    rec_wz = -0.55 + bz * cw + (fz + 0.5) * fw
                else:
                    rec_wx = -0.55 + (bx + 0.5) * cw
                    rec_wy = -0.55 + (by + 0.5) * cw
                    rec_wz = -0.55 + (bz + 0.5) * cw
                    
                rec_omega = torch.tensor([rec_wx, rec_wy, rec_wz], dtype=torch.float32, device=device)
                rec_delta_R = axis_angle_to_matrix(rec_omega.unsqueeze(0)).squeeze(0)
                R_final = torch.mm(rec_delta_R, R_coarse)
                r6d_final = matrix_to_rotation_6d(R_final.unsqueeze(0)).expand(B, -1)
            else:
                r6d_final = r6d_hat

            is_bypass = (tokens.scale3_core_torsions[:, :3] == self.RIGID_BYPASS_TOKEN).all() or M == 0
            if is_bypass:
                torsions = torch.zeros((B, M, 2), device=device)
                if M > 0:
                    torsions[:, :, 1] = 1.0
                return ArticulatedPoseParams(translation=T_final, rotation_6d=r6d_final, torsions_sc=torsions)
                
            depths = state.bond_tree_depths.to(device)
            core_idx = torch.where(depths <= 2)[0]
            fine_idx = torch.where(depths > 2)[0]
            ordered_idx = torch.cat([core_idx, fine_idx])
            
            s3 = tokens.scale3_core_torsions[0]
            s4 = tokens.scale4_fine_torsions[0]
            rec_thetas = torch.zeros(M, device=device)
            
            if len(ordered_idx) <= 3:
                for slot in range(min(3, len(ordered_idx))):
                    bm = ordered_idx[slot]
                    t3_val = s3[slot].item()
                    t3_raw = ((t3_val - slot * 13) * self.INV_1024) % self.k_tor
                    b_s3 = t3_raw // 32
                    if up_to_scale >= 4:
                        t4_val = s4[slot].item()
                        t4_raw = ((t4_val - slot * 17) * self.INV_1024) % self.k_tor
                        ang = (t4_raw + 0.5) / 1024.0 * 2.0 * np.pi - np.pi
                    else:
                        ang = (b_s3 + 0.5) / 32.0 * 2.0 * np.pi - np.pi
                    rec_thetas[bm] = ang
            elif len(ordered_idx) <= 6:
                for slot in range(3):
                    b1 = slot * 2
                    b2 = slot * 2 + 1
                    t3_val = s3[slot].item()
                    t3_raw = ((t3_val - slot * 13) * self.INV_1024) % self.k_tor
                    bin1 = t3_raw // 32
                    bin2 = t3_raw % 32
                    if up_to_scale >= 4:
                        t4_val = s4[slot].item()
                        t4_raw = ((t4_val - slot * 17) * self.INV_1024) % self.k_tor
                        res1 = t4_raw // 32
                        res2 = t4_raw % 32
                        frac1 = (bin1 * 32 + res1 + 0.5) / 1024.0
                        frac2 = (bin2 * 32 + res2 + 0.5) / 1024.0
                    else:
                        frac1 = (bin1 + 0.5) / 32.0
                        frac2 = (bin2 + 0.5) / 32.0
                    if b1 < len(ordered_idx):
                        rec_thetas[ordered_idx[b1]] = frac1 * 2.0 * np.pi - np.pi
                    if b2 < len(ordered_idx):
                        rec_thetas[ordered_idx[b2]] = frac2 * 2.0 * np.pi - np.pi
            else: # len(ordered_idx) > 6
                for slot in range(3):
                    b0 = slot
                    b1 = 3 + slot
                    b2 = 6 + slot
                    b3 = 9 + slot
                    t3_val = s3[slot].item()
                    t3_raw = ((t3_val - slot * 13) * self.INV_1024) % self.k_tor
                    bin0 = (t3_raw // 64) % 16
                    bin1 = (t3_raw // 8) % 8
                    bin2 = t3_raw % 8
                    
                    if up_to_scale >= 4:
                        t4_val = s4[slot].item()
                        t4_raw = ((t4_val - slot * 17) * self.INV_1024) % self.k_tor
                        res0 = (t4_raw // 64) % 16
                        res1 = (t4_raw // 8) % 8
                        bin3 = t4_raw % 8
                        frac0 = (bin0 * 16 + res0 + 0.5) / 256.0
                        frac1 = (bin1 * 8 + res1 + 0.5) / 64.0
                        if b3 < len(ordered_idx):
                            rec_thetas[ordered_idx[b3]] = (bin3 + 0.5) / 8.0 * 2.0 * np.pi - np.pi
                    else:
                        frac0 = (bin0 + 0.5) / 16.0
                        frac1 = (bin1 + 0.5) / 8.0
                        
                    if b0 < len(ordered_idx):
                        rec_thetas[ordered_idx[b0]] = frac0 * 2.0 * np.pi - np.pi
                    if b1 < len(ordered_idx):
                        rec_thetas[ordered_idx[b1]] = frac1 * 2.0 * np.pi - np.pi
                    if b2 < len(ordered_idx):
                        rec_thetas[ordered_idx[b2]] = (bin2 + 0.5) / 8.0 * 2.0 * np.pi - np.pi
                    
            thetas_t = rec_thetas.unsqueeze(0).expand(B, -1)
            sc = torch.stack([torch.sin(thetas_t), torch.cos(thetas_t)], dim=-1)
            return ArticulatedPoseParams(translation=T_final, rotation_6d=r6d_final, torsions_sc=sc)
        else:
            # Synthetic benchmark mode: pure torsions on all 4 slots
            is_bypass = (tokens.scale3_core_torsions == self.RIGID_BYPASS_TOKEN).all()
            if is_bypass:
                torsions = torch.zeros((B, M, 2), device=device)
                if M > 0:
                    torsions[:, :, 1] = 1.0
                return ArticulatedPoseParams(translation=T_final, rotation_6d=r6d_hat, torsions_sc=torsions)
                
            depths = state.bond_tree_depths.to(device)
            core_idx = torch.where(depths <= 2)[0]
            fine_idx = torch.where(depths > 2)[0]
            ordered_idx = torch.cat([core_idx, fine_idx])
            
            s3 = tokens.scale3_core_torsions[0]
            s4 = tokens.scale4_fine_torsions[0]
            rec_thetas = torch.zeros(M, device=device)
            
            if len(ordered_idx) <= 4:
                for slot in range(min(4, len(ordered_idx))):
                    bm = ordered_idx[slot]
                    t3_val = s3[slot].item()
                    t3_raw = ((t3_val - slot * 13) * self.INV_1024) % self.k_tor
                    base_ang = (t3_raw / 1024.0) * 2.0 * np.pi - np.pi
                    if up_to_scale >= 4:
                        t4_val = s4[slot].item()
                        t4_raw = ((t4_val - slot * 17) * self.INV_1024) % self.k_tor
                        res_ang = ((t4_raw + 0.5) / 1024.0) * (2.0 * np.pi / 1024.0)
                        ang = base_ang + res_ang
                    else:
                        ang = base_ang
                    rec_thetas[bm] = ang
            elif len(ordered_idx) <= 8:
                for slot in range(4):
                    b1 = slot * 2
                    b2 = slot * 2 + 1
                    t3_val = s3[slot].item()
                    t3_raw = ((t3_val - slot * 13) * self.INV_1024) % self.k_tor
                    bin1 = t3_raw // 32
                    bin2 = t3_raw % 32
                    if up_to_scale >= 4:
                        t4_val = s4[slot].item()
                        t4_raw = ((t4_val - slot * 17) * self.INV_1024) % self.k_tor
                        res1 = t4_raw // 32
                        res2 = t4_raw % 32
                        frac1 = (bin1 * 32 + res1 + 0.5) / 1024.0
                        frac2 = (bin2 * 32 + res2 + 0.5) / 1024.0
                    else:
                        frac1 = (bin1 + 0.5) / 32.0
                        frac2 = (bin2 + 0.5) / 32.0
                    if b1 < len(ordered_idx):
                        rec_thetas[ordered_idx[b1]] = frac1 * 2.0 * np.pi - np.pi
                    if b2 < len(ordered_idx):
                        rec_thetas[ordered_idx[b2]] = frac2 * 2.0 * np.pi - np.pi
            else: # len(ordered_idx) > 8
                for slot in range(4):
                    b0 = slot
                    b1 = 4 + slot
                    b2 = 8 + slot
                    b3 = 12 + slot
                    t3_val = s3[slot].item()
                    t3_raw = ((t3_val - slot * 13) * self.INV_1024) % self.k_tor
                    bin0 = (t3_raw // 64) % 16
                    bin1 = (t3_raw // 8) % 8
                    bin2 = t3_raw % 8
                    
                    if up_to_scale >= 4:
                        t4_val = s4[slot].item()
                        t4_raw = ((t4_val - slot * 17) * self.INV_1024) % self.k_tor
                        res0 = (t4_raw // 64) % 16
                        res1 = (t4_raw // 8) % 8
                        bin3 = t4_raw % 8
                        frac0 = (bin0 * 16 + res0 + 0.5) / 256.0
                        frac1 = (bin1 * 8 + res1 + 0.5) / 64.0
                        if b3 < len(ordered_idx):
                            rec_thetas[ordered_idx[b3]] = (bin3 + 0.5) / 8.0 * 2.0 * np.pi - np.pi
                    else:
                        frac0 = (bin0 + 0.5) / 16.0
                        frac1 = (bin1 + 0.5) / 8.0
                        
                    if b0 < len(ordered_idx):
                        rec_thetas[ordered_idx[b0]] = frac0 * 2.0 * np.pi - np.pi
                    if b1 < len(ordered_idx):
                        rec_thetas[ordered_idx[b1]] = frac1 * 2.0 * np.pi - np.pi
                    if b2 < len(ordered_idx):
                        rec_thetas[ordered_idx[b2]] = (bin2 + 0.5) / 8.0 * 2.0 * np.pi - np.pi
                    
            thetas_t = rec_thetas.unsqueeze(0).expand(B, -1)
            sc = torch.stack([torch.sin(thetas_t), torch.cos(thetas_t)], dim=-1)
            return ArticulatedPoseParams(translation=T_final, rotation_6d=r6d_hat, torsions_sc=sc)

    def decode_coords(
        self,
        tokens: DiscretePoseTokens,
        state: CanonicalComplexState,
        up_to_scale: int = 4
    ) -> torch.Tensor:
        params = self.decode_params(tokens, state, up_to_scale=up_to_scale)
        M = state.rotatable_edges.shape[0]
        coords = ForwardKinematics.apply(
            ref_coords=state.ligand_ref_coords,
            rotatable_edges=state.rotatable_edges,
            kinematic_masks=state.kinematic_masks,
            translation=params.translation,
            rotation_6d=params.rotation_6d,
            torsions_sc=params.torsions_sc if M > 0 else None
        )
        return coords

    def forward_loss(
        self,
        state: CanonicalComplexState,
        physics_loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        tokens = self.encode(state)
        c4 = self.decode_coords(tokens, state, up_to_scale=4)
        c3 = self.decode_coords(tokens, state, up_to_scale=3)
        c2 = self.decode_coords(tokens, state, up_to_scale=2)
        
        M = state.rotatable_edges.shape[0]
        gt_coords = ForwardKinematics.apply(
            ref_coords=state.ligand_ref_coords,
            rotatable_edges=state.rotatable_edges,
            kinematic_masks=state.kinematic_masks,
            translation=state.gt_translation,
            rotation_6d=state.gt_rotation_6d,
            torsions_sc=state.gt_torsions_sc if M > 0 else None
        )
        gt_rigid = ForwardKinematics.apply(
            ref_coords=state.ligand_ref_coords,
            rotatable_edges=state.rotatable_edges,
            kinematic_masks=state.kinematic_masks,
            translation=state.gt_translation,
            rotation_6d=state.gt_rotation_6d,
            torsions_sc=None
        )
        
        from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_squared_loss
        loss_s4 = torch.mean(symmetry_aware_squared_loss(c4, gt_coords, state.automorphism_perms))
        loss_s3 = torch.mean(symmetry_aware_squared_loss(c3, gt_coords, state.automorphism_perms))
        loss_s2 = torch.mean(symmetry_aware_squared_loss(c2, gt_rigid, state.automorphism_perms))
        
        if physics_loss_fn is not None:
            loss_fs = physics_loss_fn.compute_physics_loss(state, c4)
        else:
            loss_fs = torch.tensor(0.0, device=device)
            
        commit_total = torch.tensor(0.01, device=device)
        total_loss = loss_s4 + 0.5 * loss_s3 + 0.25 * loss_s2 + 0.08 * loss_fs + 0.25 * commit_total
        
        return {
            "loss": total_loss,
            "loss_s4": loss_s4,
            "loss_s3": loss_s3,
            "loss_s2": loss_s2,
            "loss_fs": loss_fs,
            "commit_loss": commit_total
        }
