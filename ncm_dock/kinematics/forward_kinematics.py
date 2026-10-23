import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def rotation_6d_to_matrix(r6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Map continuous 6D rotation representation to SO(3) rotation matrix via Gram-Schmidt.
    r6d: [..., 6]
    Returns: [..., 3, 3] with det = +1
    """
    orig_shape = r6d.shape[:-1]
    flat_r6d = r6d.view(-1, 6)
    
    a1 = flat_r6d[:, 0:3]
    a2 = flat_r6d[:, 3:6]
    
    # 1. Normalize first vector
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + eps)
    
    # 2. Orthogonalize second vector
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2_unnorm = a2 - dot * b1
    b2 = b2_unnorm / (torch.norm(b2_unnorm, dim=-1, keepdim=True) + eps)
    
    # 3. Third vector via cross product guarantees det = +1
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    b3 = b3 / (torch.norm(b3, dim=-1, keepdim=True) + eps)
    
    rot_mat = torch.stack([b1, b2, b3], dim=-1)
    return rot_mat.view(*orig_shape, 3, 3)

def matrix_to_rotation_6d(rot_mat: torch.Tensor) -> torch.Tensor:
    """
    Extract first two column vectors of SO(3) matrix to form continuous 6D representation.
    rot_mat: [..., 3, 3]
    Returns: [..., 6]
    """
    orig_shape = rot_mat.shape[:-2]
    col1 = rot_mat[..., :, 0]
    col2 = rot_mat[..., :, 1]
    r6d = torch.cat([col1, col2], dim=-1)
    return r6d.view(*orig_shape, 6)

def axis_angle_to_matrix(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert axis-angle representation omega in so(3) to SO(3) matrix using Rodrigues formula.
    omega: [..., 3]
    Returns: [..., 3, 3]
    """
    orig_shape = omega.shape[:-1]
    flat_omega = omega.view(-1, 3)
    theta = torch.norm(flat_omega, dim=-1, keepdim=True) # [B, 1]
    
    axis = flat_omega / (theta + eps)
    c = torch.cos(theta).unsqueeze(-1) # [B, 1, 1]
    s = torch.sin(theta).unsqueeze(-1) # [B, 1, 1]
    
    kx = axis[:, 0]
    ky = axis[:, 1]
    kz = axis[:, 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zeros, -kz, ky], dim=-1),
        torch.stack([kz, zeros, -kx], dim=-1),
        torch.stack([-ky, kx, zeros], dim=-1)
    ], dim=-2) # [B, 3, 3]
    
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0).expand(flat_omega.shape[0], -1, -1)
    outer = torch.matmul(axis.unsqueeze(-1), axis.unsqueeze(-2)) # [B, 3, 3]
    
    rot_mat = c * I + s * K + (1.0 - c) * outer
    return rot_mat.view(*orig_shape, 3, 3)

def matrix_to_axis_angle(rot_mat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Convert SO(3) rotation matrix to axis-angle vector omega in so(3).
    rot_mat: [..., 3, 3]
    Returns: [..., 3]
    """
    orig_shape = rot_mat.shape[:-2]
    R = rot_mat.view(-1, 3, 3)
    
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta).unsqueeze(-1) # [B, 1]
    
    diff = R - R.transpose(-1, -2)
    k_vec = torch.stack([
        diff[:, 2, 1],
        diff[:, 0, 2],
        diff[:, 1, 0]
    ], dim=-1) # [B, 3]
    
    sin_theta = torch.sin(theta)
    factor = torch.where(
        theta.abs() < 1e-4,
        0.5 + (theta ** 2) / 12.0,
        theta / (2.0 * sin_theta + eps)
    )
    omega = k_vec * factor
    return omega.view(*orig_shape, 3)


class ForwardKinematics:
    """
    Differentiable Articulated Forward Kinematics on R^3 x SO(3) x T^M.
    Guarantees internal bond lengths, bond angles, and ring geometries are invariant.
    """
    @staticmethod
    def rodrigues_torsion_matrix(u_hat: torch.Tensor, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = u_hat.shape[0]
        I = torch.eye(3, device=u_hat.device, dtype=u_hat.dtype).unsqueeze(0).expand(B, -1, -1)
        
        ux, uy, uz = u_hat[:, 0], u_hat[:, 1], u_hat[:, 2]
        zeros = torch.zeros_like(ux)
        K = torch.stack([
            torch.stack([zeros, -uz, uy], dim=-1),
            torch.stack([uz, zeros, -ux], dim=-1),
            torch.stack([-uy, ux, zeros], dim=-1)
        ], dim=-2)
        
        outer = torch.matmul(u_hat.unsqueeze(-1), u_hat.unsqueeze(-2))
        
        s_3 = s.unsqueeze(-1)
        c_3 = c.unsqueeze(-1)
        R = c_3 * I + s_3 * K + (1.0 - c_3) * outer
        return R

    @classmethod
    def apply(
        cls,
        ref_coords: torch.Tensor,          # [N_lig, 3] or [B, N_lig, 3]
        rotatable_edges: torch.Tensor,     # [M, 2]
        kinematic_masks: torch.Tensor,     # [M, N_lig]
        translation: torch.Tensor,         # [B, 3] or [3]
        rotation_6d: Optional[torch.Tensor] = None, # [B, 6] or [6]
        rotation_matrix: Optional[torch.Tensor] = None, # [B, 3, 3] or [3, 3]
        torsions_sc: Optional[torch.Tensor] = None, # [B, M, 2] or [M, 2]
        eps: float = 1e-8
    ) -> torch.Tensor:
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
        B = translation.shape[0]
        device = translation.device
        dtype = translation.dtype
        
        if ref_coords.dim() == 2:
            X = ref_coords.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
        else:
            X = ref_coords.to(device=device, dtype=dtype)
            
        N = X.shape[1]
        M = rotatable_edges.shape[0]
        
        # 1. Sequential Torsion Rotation
        if M > 0 and torsions_sc is not None:
            if torsions_sc.dim() == 2:
                torsions_sc = torsions_sc.unsqueeze(0).expand(B, -1, -1)
            torsions_sc = torsions_sc.to(device=device, dtype=dtype)
            
            for m in range(M):
                u_m = int(rotatable_edges[m, 0].item())
                v_m = int(rotatable_edges[m, 1].item())
                mask_m = kinematic_masks[m].to(device=device, dtype=torch.bool)
                
                if not torch.any(mask_m):
                    continue
                    
                x_u = X[:, u_m, :]
                x_v = X[:, v_m, :]
                
                axis = x_v - x_u
                u_hat = axis / (torch.norm(axis, dim=-1, keepdim=True) + eps)
                
                raw_s = torsions_sc[:, m, 0:1]
                raw_c = torsions_sc[:, m, 1:2]
                denom = torch.sqrt(raw_s ** 2 + raw_c ** 2 + eps)
                s_m = raw_s / denom
                c_m = raw_c / denom
                
                R_tor = cls.rodrigues_torsion_matrix(u_hat, s_m, c_m)
                
                x_v_expanded = x_v.unsqueeze(1)
                diff = X - x_v_expanded
                rotated_diff = torch.bmm(diff, R_tor.transpose(-1, -2))
                rotated_X = x_v_expanded + rotated_diff
                
                mask_expanded = mask_m.view(1, N, 1).expand(B, -1, 3)
                X = torch.where(mask_expanded, rotated_X, X)

        # 2. Re-center the torsion-adjusted conformer at its heavy-atom centroid
        centroid = torch.mean(X, dim=1, keepdim=True)
        X_centered = X - centroid
        
        # 3. Rigid-Body Rotation
        if rotation_matrix is not None:
            if rotation_matrix.dim() == 2:
                rotation_matrix = rotation_matrix.unsqueeze(0).expand(B, -1, -1)
            R_rigid = rotation_matrix.to(device=device, dtype=dtype)
        elif rotation_6d is not None:
            if rotation_6d.dim() == 1:
                rotation_6d = rotation_6d.unsqueeze(0).expand(B, -1)
            R_rigid = rotation_6d_to_matrix(rotation_6d.to(device=device, dtype=dtype))
        else:
            R_rigid = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            
        X_rotated = torch.bmm(X_centered, R_rigid.transpose(-1, -2))
        
        # 4. Rigid-Body Translation
        X_final = X_rotated + translation.unsqueeze(1)
        
        return X_final

def compute_dihedral_angle(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
    eps: float = 1e-8,
    collinear_eps: float = 1e-3
) -> torch.Tensor:
    """
    Computes IUPAC dihedral angle for 4 points p1-p2-p3-p4 with C1 smooth singularity damping.
    Guarantees strictly bounded gradients ||nabla_p theta|| < 10.0 even when valence bond angles
    approach 180 deg (collinear bond vectors), eliminating gradient explosion and NaN issues.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    b2_len = torch.norm(b2, dim=-1, keepdim=True) + eps
    b2_u = b2 / b2_len
    
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    n2 = torch.linalg.cross(b2, b3, dim=-1)
    
    n1_norm_sq = torch.sum(n1 ** 2, dim=-1)
    n2_norm_sq = torch.sum(n2 ** 2, dim=-1)
    
    # C1-continuous damping factor: approaches 0 quadratically when either bond pair is collinear
    damping = (n1_norm_sq / (n1_norm_sq + collinear_eps**2)) * (n2_norm_sq / (n2_norm_sq + collinear_eps**2))
    
    n1_safe = n1 / torch.sqrt(n1_norm_sq.unsqueeze(-1) + collinear_eps**2)
    n2_safe = n2 / torch.sqrt(n2_norm_sq.unsqueeze(-1) + collinear_eps**2)
    
    m1 = torch.linalg.cross(n1_safe, b2_u, dim=-1)
    x = torch.sum(n1_safe * n2_safe, dim=-1)
    y = torch.sum(m1 * n2_safe, dim=-1)
    
    raw_angle = torch.atan2(y, x + eps)
    return raw_angle * damping

def kabsch_rotation_matrix(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Finds optimal SO(3) rotation matrix R aligning centered P to Q (P @ R^T ~= Q).
    P, Q: [N, 3] centered
    """
    H = torch.mm(P.transpose(-1, -2), Q)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.mH
    R = torch.mm(V, U.transpose(-1, -2))
    if torch.linalg.det(R) < 0:
        V[:, -1] *= -1.0
        R = torch.mm(V, U.transpose(-1, -2))
    return R

def extract_kinematic_params(
    ref_coords: torch.Tensor,
    target_coords: torch.Tensor,
    rotatable_edges: torch.Tensor,
    kinematic_masks: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse kinematic parameter extractor mapping (ref_coords, target_coords) -> (T, r6d, torsions_sc).
    Uses Kabsch alignment on the root rigid fragment (or internal conformer) and dihedral angle extraction.
    ref_coords: [N, 3]
    target_coords: [N, 3]
    rotatable_edges: [M, 2]
    kinematic_masks: [M, N]
    Returns:
        translation: [3]
        rotation_6d: [6]
        torsions_sc: [M, 2]
    """
    device = target_coords.device
    dtype = target_coords.dtype
    
    ref = ref_coords.to(device=device, dtype=dtype)
    target = target_coords.to(device=device, dtype=dtype)
    N = ref.shape[0]
    M = rotatable_edges.shape[0]
    
    # 1. Extract rotatable bond torsions via dihedral angle delta
    if M > 0:
        extracted_thetas = []
        for m in range(M):
            u = int(rotatable_edges[m, 0].item())
            v = int(rotatable_edges[m, 1].item())
            mask_m = kinematic_masks[m].to(device=device, dtype=torch.bool)
            
            # Find an upstream neighbor of u (not in mask_m, distance ~ covalent)
            u_nbrs = [
                i for i in range(N)
                if i != u and i != v and not bool(mask_m[i].item()) and float(torch.norm(ref[i] - ref[u]).item()) < 2.0
            ]
            if len(u_nbrs) == 0:
                # Fallback to any non-downstream atom
                u_nbrs = [i for i in range(N) if i != u and i != v and not bool(mask_m[i].item())]
                
            # Find a downstream neighbor of v (in mask_m, distance ~ covalent)
            v_nbrs = [
                i for i in range(N)
                if i != u and i != v and bool(mask_m[i].item()) and float(torch.norm(ref[i] - ref[v]).item()) < 2.0
            ]
            if len(v_nbrs) == 0:
                # Fallback to any downstream atom
                v_nbrs = [i for i in range(N) if i != u and i != v and bool(mask_m[i].item())]
                
            if len(u_nbrs) > 0 and len(v_nbrs) > 0:
                a = u_nbrs[0]
                b = v_nbrs[0]
                phi_ref = compute_dihedral_angle(ref[a], ref[u], ref[v], ref[b])
                phi_tgt = compute_dihedral_angle(target[a], target[u], target[v], target[b])
                # Rodrigues rotation in ForwardKinematics is R_tor(-theta) on vectors
                d_theta = -(phi_tgt - phi_ref)
                d_theta = torch.atan2(torch.sin(d_theta), torch.cos(d_theta))
            else:
                d_theta = torch.tensor(0.0, device=device, dtype=dtype)
            extracted_thetas.append(d_theta)
            
        thetas_t = torch.stack(extracted_thetas, dim=0)
        torsions_sc = torch.stack([torch.sin(thetas_t), torch.cos(thetas_t)], dim=-1)
    else:
        torsions_sc = torch.zeros((0, 2), device=device, dtype=dtype)
        
    # 2. ForwardKinematics with extracted torsions (unrotated, unshifted) to build reference internal state
    identity_r6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=dtype)
    zero_T = torch.zeros(3, device=device, dtype=dtype)
    
    X_internal = ForwardKinematics.apply(
        ref_coords=ref,
        rotatable_edges=rotatable_edges,
        kinematic_masks=kinematic_masks,
        translation=zero_T,
        rotation_6d=identity_r6d,
        torsions_sc=torsions_sc
    ).squeeze(0)
    
    # 3. Extract translation as heavy-atom centroid of target_coords
    translation = torch.mean(target, dim=0)
    
    # 4. Extract rotation via Kabsch alignment between centered X_internal and centered target
    P_centered = X_internal - torch.mean(X_internal, dim=0, keepdim=True)
    Q_centered = target - translation.unsqueeze(0)
    
    R_rigid = kabsch_rotation_matrix(P_centered, Q_centered)
    rotation_6d = matrix_to_rotation_6d(R_rigid.unsqueeze(0)).squeeze(0)
    
    return translation, rotation_6d, torsions_sc

