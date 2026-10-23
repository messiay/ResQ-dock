from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem

ROTATABLE_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

@dataclass
class LigandKinematicGraph:
    atom_features: torch.Tensor       # [N, D]
    ref_coords: torch.Tensor          # [N, 3] Zero-centered RDKit conformer
    atom_charges: torch.Tensor        # [N]
    atom_vdw: torch.Tensor            # [N]
    rotatable_edges: torch.Tensor     # [M, 2] (u_m, v_m)
    kinematic_masks: torch.Tensor     # [M, N] Bool mask
    bond_tree_depths: torch.Tensor    # [M] LongTensor
    automorphism_perms: torch.Tensor  # [N_sym, N] LongTensor
    root_atom_idx: int
    num_rotatable_bonds: int

class KinematicTreeBuilder:
    """Constructs BFS rotatable bond tree and symmetry automorphisms from RDKit Mol."""
    
    VDW_RADII = {
        1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98
    }

    def __init__(self):
        self.rot_smarts = ROTATABLE_SMARTS

    def build_kinematic_tree(self, mol: Chem.Mol, conf_id: int = -1) -> LigandKinematicGraph:
        """Analyze mol and build kinematic tree with zero in-place side effects."""
        mol = Chem.Mol(mol) # Deep copy
        Chem.SanitizeMol(mol)
        
        # Heavy atom count
        heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        N = len(heavy_atoms)
        if N == 0:
            raise ValueError("Molecule has zero heavy atoms.")
            
        # Ensure only heavy atoms are indexed
        mol_heavy = Chem.RemoveHs(mol)
        N = mol_heavy.GetNumAtoms()
        
        # Ensure conformer exists
        if mol_heavy.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_heavy, randomSeed=42)
        conf = mol_heavy.GetConformer(conf_id)
        raw_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(N)], dtype=np.float32)
        
        # Compute Gasteiger partial charges
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol_heavy)
        charges = []
        vdw = []
        atom_feats = []
        for i in range(N):
            atom = mol_heavy.GetAtomWithIdx(i)
            # Partial charge
            try:
                chg = float(atom.GetProp("_GasteigerCharge"))
                if np.isnan(chg) or np.isinf(chg):
                    chg = 0.0
            except Exception:
                chg = 0.0
            charges.append(chg)
            
            # VDW radius
            anum = atom.GetAtomicNum()
            vdw.append(self.VDW_RADII.get(anum, 1.70))
            
            # Atom features: atomic number one-hot (up to 53), degree, hybridization, aromaticity
            feats = [
                float(anum == 6), float(anum == 7), float(anum == 8),
                float(anum == 9), float(anum == 15), float(anum == 16),
                float(anum == 17), float(anum == 35), float(anum not in [6,7,8,9,15,16,17,35]),
                float(atom.GetDegree()) / 4.0,
                float(atom.GetIsAromatic()),
                float(atom.GetFormalCharge()),
                chg,
                self.VDW_RADII.get(anum, 1.70) / 2.0
            ]
            atom_feats.append(feats)
            
        atom_feats_t = torch.tensor(atom_feats, dtype=torch.float32)
        charges_t = torch.tensor(charges, dtype=torch.float32)
        vdw_t = torch.tensor(vdw, dtype=torch.float32)
        
        # 1. Rotatable bonds matching SMARTS
        matches = mol_heavy.GetSubstructMatches(self.rot_smarts)
        rot_bonds = []
        seen_bonds = set()
        for u, v in matches:
            bond = mol_heavy.GetBondBetweenAtoms(u, v)
            if bond is not None and not bond.IsInRing():
                b_idx = tuple(sorted((u, v)))
                if b_idx not in seen_bonds:
                    seen_bonds.add(b_idx)
                    rot_bonds.append((u, v))
                    
        # 2. Find root rigid fragment
        # Distance matrix in graph
        adj = Chem.GetDistanceMatrix(mol_heavy)
        sum_dist = np.sum(adj, axis=1) # [N]
        # Morgan ranks as tie-breaker
        ranks = list(Chem.CanonicalRankAtoms(mol_heavy, breakTies=True))
        
        # Atom with minimum sum of shortest-path graph distances
        best_atom = 0
        best_val = (sum_dist[0], ranks[0])
        for i in range(1, N):
            val = (sum_dist[i], ranks[i])
            if val < best_val:
                best_val = val
                best_atom = i
                
        root_atom = best_atom
        
        # 3. BFS traversal to orient rotatable bonds away from root
        # Build adjacency graph
        neighbors: Dict[int, List[int]] = {i: [] for i in range(N)}
        for bond in mol_heavy.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            neighbors[u].append(v)
            neighbors[v].append(u)
            
        # BFS distances from root
        dist_from_root = {root_atom: 0}
        parent = {root_atom: -1}
        queue = [root_atom]
        while queue:
            curr = queue.pop(0)
            for nbr in neighbors[curr]:
                if nbr not in dist_from_root:
                    dist_from_root[nbr] = dist_from_root[curr] + 1
                    parent[nbr] = curr
                    queue.append(nbr)
                    
        # Orient bonds: u_m -> v_m where dist_from_root[u_m] < dist_from_root[v_m]
        oriented_bonds = []
        bond_depths = []
        kinematic_masks = []
        
        for u, v in rot_bonds:
            if dist_from_root[u] > dist_from_root[v]:
                u, v = v, u # Orient away from root
            oriented_bonds.append((u, v))
            
            # BFS depth of the bond: 1 + number of rotatable bonds on path from root to u
            # or BFS depth in terms of rotatable bond hierarchy
            # Count rotatable bonds encountered on path from root to u:
            path_depth = 1
            curr_node = u
            while curr_node != root_atom and curr_node != -1:
                p = parent[curr_node]
                if tuple(sorted((curr_node, p))) in seen_bonds:
                    path_depth += 1
                curr_node = p
            bond_depths.append(path_depth)
            
            # Compute downstream atoms when edge (u, v) is cut
            # Subtree rooted at v excluding u and v
            downstream = set()
            sub_q = [v]
            sub_visited = {u, v}
            while sub_q:
                curr = sub_q.pop(0)
                for nbr in neighbors[curr]:
                    if nbr not in sub_visited:
                        sub_visited.add(nbr)
                        downstream.add(nbr)
                        sub_q.append(nbr)
            
            mask = np.zeros(N, dtype=bool)
            for d in downstream:
                mask[d] = True
            kinematic_masks.append(mask)

        # Sort bonds by ascending tree depth for correct forward kinematics order
        M = len(oriented_bonds)
        if M > 0:
            sort_order = np.argsort(bond_depths)
            oriented_bonds = [oriented_bonds[i] for i in sort_order]
            bond_depths = [bond_depths[i] for i in sort_order]
            kinematic_masks = [kinematic_masks[i] for i in sort_order]
            rot_edges_t = torch.tensor(oriented_bonds, dtype=torch.long)
            kin_masks_t = torch.tensor(np.array(kinematic_masks), dtype=torch.bool)
            depths_t = torch.tensor(bond_depths, dtype=torch.long)
        else:
            rot_edges_t = torch.zeros((0, 2), dtype=torch.long)
            kin_masks_t = torch.zeros((0, N), dtype=torch.bool)
            depths_t = torch.zeros(0, dtype=torch.long)
            
        # 5. Graph Automorphisms
        auto_matches = mol_heavy.GetSubstructMatches(mol_heavy, uniquify=False)
        if len(auto_matches) == 0:
            auto_matches = [tuple(range(N))]
        perms_t = torch.tensor(auto_matches, dtype=torch.long)
        
        # Zero-center reference conformer
        ref_coords_centered = raw_coords - np.mean(raw_coords, axis=0, keepdims=True)
        ref_coords_t = torch.tensor(ref_coords_centered, dtype=torch.float32)
        
        return LigandKinematicGraph(
            atom_features=atom_feats_t,
            ref_coords=ref_coords_t,
            atom_charges=charges_t,
            atom_vdw=vdw_t,
            rotatable_edges=rot_edges_t,
            kinematic_masks=kin_masks_t,
            bond_tree_depths=depths_t,
            automorphism_perms=perms_t,
            root_atom_idx=root_atom,
            num_rotatable_bonds=M
        )
