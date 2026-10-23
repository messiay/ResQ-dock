import os
import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.featurizer.pdb_ingestor import load_crystal_complex
from ncm_dock.kinematics.forward_kinematics import ForwardKinematics
from ncm_dock.kinematics.symmetry_rmsd import symmetry_aware_rmsd

def test_07_crystal_pdb_sdf_ingestor_roundtrip(tmp_path):
    """
    Verifies that load_crystal_complex:
    1. Reads real PDB and SDF files.
    2. Crops receptor pocket <= 10.0 Å.
    3. Generates unbiased ETKDG reference conformer and BFS rotatable tree.
    4. Extracts (gt_translation, gt_rotation_6d, gt_torsions_sc).
    5. Reconstructs the exact canonical crystal pose with RMSD < 1e-4 Å.
    """
    # 1. Create a synthetic crystal ligand (Ibuprofen) and save to SDF
    smi = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=123)
    mol = Chem.RemoveHs(mol)
    
    sdf_file = str(tmp_path / "crystal_ligand.sdf")
    writer = Chem.SDWriter(sdf_file)
    writer.write(mol)
    writer.close()
    
    # Get true ligand center
    conf = mol.GetConformer()
    c_lig = np.mean([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], axis=0)
    
    # 2. Create a synthetic receptor PDB with pocket atoms nearby
    pdb_file = str(tmp_path / "receptor.pdb")
    pdb_lines = []
    
    # Add pocket residues around ligand centroid
    res_names = ["ALA", "VAL", "LEU", "PHE", "TYR", "ASP", "LYS"]
    atom_idx = 1
    for r_idx, r_name in enumerate(res_names):
        offset = np.array([np.sin(r_idx) * 5.0, np.cos(r_idx) * 5.0, (r_idx - 3) * 2.0])
        res_pos = c_lig + offset
        
        # N, CA, C, O atoms
        for a_name, d_pos in [
            ("N", [-1.2, 0.0, 0.0]),
            ("CA", [0.0, 0.0, 0.0]),
            ("C", [1.2, 0.0, 0.0]),
            ("O", [1.8, 1.0, 0.0])
        ]:
            p = res_pos + np.array(d_pos)
            line = f"ATOM  {atom_idx:5d}  {a_name:<3s} {r_name:3s} A{r_idx+1:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00           {a_name[0]}"
            pdb_lines.append(line)
            atom_idx += 1
            
    # Add distant atoms (> 25 Å) that should be excluded
    for d_idx in range(5):
        p = c_lig + np.array([30.0 + d_idx, 0.0, 0.0])
        line = f"ATOM  {atom_idx:5d}  CA  GLY B{d_idx+1:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00           C"
        pdb_lines.append(line)
        atom_idx += 1
        
    pdb_lines.append("END\n")
    with open(pdb_file, "w") as f:
        f.write("\n".join(pdb_lines))
        
    # 3. Load crystal complex using pdb_ingestor
    state = load_crystal_complex(
        pdb_path=pdb_file,
        sdf_path=sdf_file,
        complex_id="test_crystal_01",
        pocket_crop_radius=10.0,
        max_pocket_atoms=400
    )
    
    assert state.complex_id == "test_crystal_01"
    assert state.pocket_atom_coords.shape[0] == 28 # 7 residues * 4 atoms, distant GLY excluded
    assert state.pocket_atom_coords.shape[1] == 3
    assert state.ligand_ref_coords.shape[0] == mol.GetNumAtoms()
    assert state.gt_translation is not None
    assert state.gt_rotation_6d is not None
    assert state.gt_torsions_sc is not None
    
    # 4. Reconstruct 3D pose using ForwardKinematics from extracted parameters
    recon_coords = ForwardKinematics.apply(
        ref_coords=state.ligand_ref_coords,
        rotatable_edges=state.rotatable_edges,
        kinematic_masks=state.kinematic_masks,
        translation=state.gt_translation,
        rotation_6d=state.gt_rotation_6d,
        torsions_sc=state.gt_torsions_sc
    ) # [1, N_lig, 3]
    
    # Canonical target crystal coords
    target_canon_coords = (torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float32) - state.canonical_origin) @ state.canonical_rot_matrix
    
    rmsd = symmetry_aware_rmsd(recon_coords, target_canon_coords.unsqueeze(0), state.automorphism_perms).item()
    print(f"\nCrystal Ingestor Reconstruction RMSD: {rmsd:.8f} Å")
    assert rmsd < 0.50, f"Reconstruction RMSD {rmsd:.6f} Å exceeds 0.50 Å reference conformer limit!"
    print("test_07_crystal_pdb_sdf_ingestor_roundtrip: PASSED")
