"""
Blind Complex Builder for Whole-Protein Molecular Docking.

Takes arbitrary whole-protein PDBs and ligand structures (SDF, MOL2, SMILES),
detects top-K candidate binding pockets using FastCavityDetector, and builds
SE(3)-canonicalized CanonicalComplexState instances ready for generative docking.
"""

import os
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.contracts import CanonicalComplexState
from ncm_dock.featurizer.pocket_canonicalizer import PocketCanonicalizer
from ncm_dock.featurizer.torsion_tree import KinematicTreeBuilder
from ncm_dock.featurizer.cavity_detector import FastCavityDetector, CandidateCavity
from ncm_dock.featurizer.pdb_ingestor import STANDARD_AMINO_ACID_VDW, STANDARD_BACKBONE_CHARGES


class BlindComplexBuilder:
    """
    Builds CanonicalComplexState instances for blind docking across candidate pockets.
    """
    def __init__(
        self,
        cavity_detector: Optional[FastCavityDetector] = None,
        pocket_crop_radius: float = 10.0,
        max_pocket_atoms: int = 400,
        max_pocket_residues: int = 64
    ):
        self.detector = cavity_detector or FastCavityDetector()
        self.canonicalizer = PocketCanonicalizer()
        self.tree_builder = KinematicTreeBuilder()
        self.pocket_crop_radius = pocket_crop_radius
        self.max_pocket_atoms = max_pocket_atoms
        self.max_pocket_residues = max_pocket_residues

    def build_blind_states(
        self,
        pdb_path: str,
        ligand_input: str,
        complex_id: str = "blind_complex",
        top_k: int = 3,
        device: torch.device = torch.device("cpu")
    ) -> List[Tuple[CandidateCavity, CanonicalComplexState]]:
        """
        Detects top-K cavities on the whole protein and builds a CanonicalComplexState for each.
        
        ligand_input: Path to SDF/MOL2 file or SMILES string.
        Returns: List of (CandidateCavity, CanonicalComplexState)
        """
        # 1. Parse Ligand (either from SDF/MOL2 file or SMILES string)
        mol = None
        if os.path.exists(ligand_input):
            if ligand_input.endswith((".sdf", ".sd")):
                suppl = Chem.SDMolSupplier(ligand_input, removeHs=True, sanitize=False)
                for m in suppl:
                    if m is not None:
                        mol = m
                        break
            elif ligand_input.endswith(".mol2"):
                mol = Chem.MolFromMol2File(ligand_input, removeHs=True, sanitize=False)
            elif ligand_input.endswith(".pdb"):
                mol = Chem.MolFromPDBFile(ligand_input, removeHs=True, sanitize=False)
        else:
            # Treat as SMILES string
            mol = Chem.MolFromSmiles(ligand_input)

        if mol is None:
            raise ValueError(f"Failed to read valid ligand from input: {ligand_input}")

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        mol_heavy = Chem.RemoveHs(mol)
        if mol_heavy.GetNumAtoms() == 0:
            raise ValueError(f"Ligand has 0 heavy atoms.")

        # Build kinematic tree and generate fresh ETKDG conformer
        tree_state = self.tree_builder.build_kinematic_tree(mol_heavy, generate_etkdg=True)

        # 2. Parse Whole Receptor PDB
        raw_atoms: List[torch.Tensor] = []
        atom_names: List[str] = []
        res_names: List[str] = []
        res_keys: List[str] = []
        vdw_list: List[float] = []
        chg_list: List[float] = []
        calpha_dict: Dict[str, torch.Tensor] = {}

        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    a_name = line[12:16].strip()
                    r_name = line[17:20].strip()
                    chain_id = line[21].strip() if len(line) > 21 else "A"
                    r_seq = line[22:26].strip() if len(line) > 25 else "1"
                    r_key = f"{chain_id}_{r_seq}_{r_name}"

                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue

                    elem = line[76:78].strip() if len(line) >= 78 else ""
                    if not elem:
                        elem = a_name[0] if a_name else "C"
                    elem = elem.upper()
                    if elem == "H":
                        continue

                    pos = torch.tensor([x, y, z], dtype=torch.float32)
                    raw_atoms.append(pos)
                    atom_names.append(a_name)
                    res_names.append(r_name)
                    res_keys.append(r_key)

                    vdw = STANDARD_AMINO_ACID_VDW.get(elem, 1.70)
                    chg = STANDARD_BACKBONE_CHARGES.get(a_name, 0.0)
                    if r_name in ["ASP", "GLU"] and a_name in ["OD1", "OD2", "OE1", "OE2"]:
                        chg = -0.5
                    elif r_name in ["LYS"] and a_name in ["NZ"]:
                        chg = 1.0
                    elif r_name in ["ARG"] and a_name in ["NH1", "NH2"]:
                        chg = 0.5
                    vdw_list.append(vdw)
                    chg_list.append(chg)

                    if a_name == "CA":
                        calpha_dict[r_key] = pos

        if not raw_atoms:
            raise ValueError(f"No heavy atoms found in receptor PDB: {pdb_path}")

        prot_coords = torch.stack(raw_atoms, dim=0) # [N_prot, 3]
        whole_prot_centroid = torch.mean(prot_coords, dim=0)

        # 3. Detect Top-K Candidate Cavities on Whole Protein
        cavities = self.detector.detect_cavities(
            prot_coords=prot_coords,
            res_names=res_names,
            res_keys=res_keys,
            top_k=top_k,
            device=device
        )

        # 4. Build a CanonicalComplexState for each candidate cavity
        results = []
        for cav in cavities:
            c_center = cav.centroid.to(device)

            # Crop atoms within pocket_crop_radius
            dists = torch.norm(prot_coords.to(device) - c_center, dim=-1)
            within_crop = dists <= self.pocket_crop_radius
            crop_idx = torch.where(within_crop)[0]

            if len(crop_idx) == 0:
                # If too small, expand to 12 Å
                dists_all = torch.norm(prot_coords.to(device) - c_center, dim=-1)
                crop_idx = torch.argsort(dists_all)[:self.max_pocket_atoms]

            # Sort by distance to cavity center and cap
            sub_dists = dists[crop_idx]
            sort_order = torch.argsort(sub_dists)[:self.max_pocket_atoms]
            final_atom_idx = crop_idx[sort_order]

            poc_pos = prot_coords[final_atom_idx.cpu()]
            poc_chg = torch.tensor([chg_list[i.item()] for i in final_atom_idx], dtype=torch.float32)
            poc_vdw = torch.tensor([vdw_list[i.item()] for i in final_atom_idx], dtype=torch.float32)

            # Extract C-alpha coords for PCA frame
            poc_res_keys = list(dict.fromkeys([res_keys[i.item()] for i in final_atom_idx]))[:self.max_pocket_residues]
            poc_ca_list = [calpha_dict[k] for k in poc_res_keys if k in calpha_dict]
            if len(poc_ca_list) >= 3:
                poc_ca = torch.stack(poc_ca_list, dim=0)
            else:
                poc_ca = poc_pos[:min(self.max_pocket_residues, len(poc_pos))]

            # Compute Canonical Transform
            trans = self.canonicalizer.compute_canonical_frame(
                calpha_coords=poc_ca,
                whole_protein_centroid=whole_prot_centroid
            )
            V = trans.rot_matrix.to(dtype=torch.float32) # [3, 3]
            mu_poc = trans.origin.to(dtype=torch.float32) # [3]

            # Transform pocket coordinates into canonical frame: (X - mu) @ V
            poc_pos_canon = torch.matmul(poc_pos - mu_poc, V)

            # Residue node features
            D_res = 32
            N_res = len(poc_res_keys)
            pocket_node_feats = torch.zeros((N_res, D_res), dtype=torch.float32)
            for ri, rk in enumerate(poc_res_keys):
                r_type = rk.split("_")[-1]
                aa_hash = hash(r_type) % D_res
                pocket_node_feats[ri, aa_hash] = 1.0

            state = CanonicalComplexState(
                complex_id=f"{complex_id}_cavity_{cav.cavity_id}",
                pocket_node_feats=pocket_node_feats,
                pocket_atom_coords=poc_pos_canon,
                pocket_atom_charges=poc_chg,
                pocket_atom_vdw=poc_vdw,
                freesolve_fft_grid=None,
                canonical_origin=mu_poc,
                canonical_rot_matrix=V,
                ligand_atom_feats=tree_state.atom_features,
                ligand_ref_coords=tree_state.ref_coords,
                ligand_atom_charges=tree_state.atom_charges,
                ligand_atom_vdw=tree_state.atom_vdw,
                rotatable_edges=tree_state.rotatable_edges,
                kinematic_masks=tree_state.kinematic_masks,
                bond_tree_depths=tree_state.bond_tree_depths,
                automorphism_perms=tree_state.automorphism_perms,
                gt_translation=None,
                gt_rotation_6d=None,
                gt_axis_angle=None,
                gt_torsions_sc=None
            )
            results.append((cav, state))

        return results
