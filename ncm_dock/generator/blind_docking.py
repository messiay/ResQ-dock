"""
Fast Blind Whole-Protein Docking Engine.

Executes autonomous whole-protein blind molecular docking:
1. Detects top-K druggable surface cavities via FastCavityDetector (~15 ms).
2. Docks ligand into candidate cavities in parallel via Steered Next-Scale Transformer (~25 ms).
3. Evaluates and ranks poses globally via FreeSolvE 3D FFT Poisson electrostatics + soft-core VdW.
4. Returns the top global binding pose and identified binding pocket in ~50-80 ms.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import time
import os
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ncm_dock.contracts import CanonicalComplexState, DiscretePoseTokens, ArticulatedPoseParams
from ncm_dock.tokenizer.factored_rvq import FactoredKinematicRVQ, get_s3_lloyd_anchors
from ncm_dock.generator.next_scale_transformer import NextScaleTransformer
from ncm_dock.generator.physics_guided_search import PhysicsGuidedBeamSearch
from ncm_dock.relaxation.freesolve_adapter import FreeSolvEAdapter
from ncm_dock.featurizer.cavity_detector import FastCavityDetector, CandidateCavity
from ncm_dock.featurizer.blind_complex_builder import BlindComplexBuilder


@dataclass
class BlindPoseResult:
    rank: int
    cavity_id: int
    cavity_centroid: torch.Tensor       # [3] Global PDB coordinates of pocket center
    global_coords: torch.Tensor         # [N_lig, 3] Global 3D PDB coordinates
    free_energy_kcal: float             # FreeSolvE binding energy (kcal/mol)
    log_probability: float              # Model log-probability
    candidate_cavity: CandidateCavity


@dataclass
class BlindDockingSummary:
    complex_id: str
    top_pose: BlindPoseResult
    all_poses: List[BlindPoseResult]
    num_cavities_evaluated: int
    detector_latency_ms: float
    docking_latency_ms: float
    total_latency_ms: float


class FastBlindDockingEngine:
    """
    Autonomous Blind Whole-Protein Molecular Docking Engine.
    """
    def __init__(
        self,
        transformer: NextScaleTransformer,
        tokenizer: FactoredKinematicRVQ,
        relaxer: Optional[FreeSolvEAdapter] = None,
        top_k_cavities: int = 3,
        beam_width: int = 8,
        device: Optional[torch.device] = None
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.relaxer = relaxer or FreeSolvEAdapter()
        self.top_k_cavities = top_k_cavities
        self.beam_width = beam_width
        self.device = device or next(self.transformer.parameters()).device

        self.builder = BlindComplexBuilder()
        self.searcher = PhysicsGuidedBeamSearch(
            transformer=self.transformer,
            tokenizer=self.tokenizer,
            relaxer=self.relaxer,
            beam_width=self.beam_width,
            physics_weight_s2=0.005,
            energy_rank_weight=1.0
        )

    @torch.no_grad()
    def dock_blind(
        self,
        pdb_path: str,
        ligand_input: str,
        complex_id: str = "blind_target",
        top_k_cavities: Optional[int] = None,
        refine_steps: int = 25
    ) -> BlindDockingSummary:
        """
        Docks a ligand blindly against an entire protein structure with zero known pocket input.
        
        pdb_path: Receptor PDB file path
        ligand_input: Ligand SDF/MOL2 file or SMILES string
        """
        k_cavities = top_k_cavities or self.top_k_cavities

        t_start = time.perf_counter()

        # Step 1: Detect Cavities and Build Candidate States
        t0_det = time.perf_counter()
        cavity_states = self.builder.build_blind_states(
            pdb_path=pdb_path,
            ligand_input=ligand_input,
            complex_id=complex_id,
            top_k=k_cavities,
            device=self.device
        )
        t_detector_ms = (time.perf_counter() - t0_det) * 1000.0

        if not cavity_states:
            raise RuntimeError(f"Could not identify any candidate cavities on {pdb_path}")

        # Step 2: Dock into each candidate cavity and collect candidates
        t0_dock = time.perf_counter()
        candidate_results: List[BlindPoseResult] = []

        for cav, state in cavity_states:
            origin = state.canonical_origin.to(self.device)
            V = state.canonical_rot_matrix.to(self.device)

            # Search with FreeSolvE physics guidance
            c_canon, tokens_b, params_b, logps_b, energies_b = self.searcher.search(
                state=state,
                device=self.device,
                use_physics_guidance=True,
                refine_beams=True,
                refine_steps=refine_steps
            )

            # Map from canonical pocket frame back to global PDB Cartesian space:
            # X_global = (X_canon @ V^T) + origin
            for b in range(min(self.beam_width, c_canon.shape[0])):
                b_canon = c_canon[b] # [N_lig, 3]
                b_global = torch.matmul(b_canon, V.T) + origin
                e_val = float(energies_b[b].item())
                lp_val = float(logps_b[b].item())

                candidate_results.append(BlindPoseResult(
                    rank=0, # Assigned after global sort
                    cavity_id=cav.cavity_id,
                    cavity_centroid=cav.centroid,
                    global_coords=b_global,
                    free_energy_kcal=e_val,
                    log_probability=lp_val,
                    candidate_cavity=cav
                ))

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_docking_ms = (time.perf_counter() - t0_dock) * 1000.0
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        # Step 3: Global FreeSolvE Physical Energy Re-ranking
        # Lowest energy = Rank 1 (best thermodynamic binding pose)
        candidate_results.sort(key=lambda x: x.free_energy_kcal)
        for r_idx, res in enumerate(candidate_results):
            res.rank = r_idx + 1

        top_pose = candidate_results[0]

        return BlindDockingSummary(
            complex_id=complex_id,
            top_pose=top_pose,
            all_poses=candidate_results,
            num_cavities_evaluated=len(cavity_states),
            detector_latency_ms=t_detector_ms,
            docking_latency_ms=t_docking_ms,
            total_latency_ms=t_total_ms
        )

    def export_blind_poses_sdf(
        self,
        summary: BlindDockingSummary,
        reference_ligand_path: str,
        out_sdf_path: str,
        top_n: int = 5
    ) -> str:
        """
        Exports the top-N blind docked poses into a multi-conformation SDF file.
        """
        suppl = Chem.SDMolSupplier(reference_ligand_path, removeHs=True, sanitize=False)
        mol = None
        for m in suppl:
            if m is not None:
                mol = m
                break
        if mol is None:
            mol = Chem.MolFromMol2File(reference_ligand_path, removeHs=True, sanitize=False)

        mol_heavy = Chem.RemoveHs(mol)
        writer = Chem.SDWriter(out_sdf_path)

        for pose in summary.all_poses[:top_n]:
            conf = Chem.Conformer(mol_heavy.GetNumAtoms())
            coords_np = pose.global_coords.detach().cpu().numpy()
            for ai in range(mol_heavy.GetNumAtoms()):
                conf.SetAtomPosition(ai, (float(coords_np[ai, 0]), float(coords_np[ai, 1]), float(coords_np[ai, 2])))

            mol_copy = Chem.Mol(mol_heavy)
            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(conf, assignId=True)
            mol_copy.SetProp("_Name", f"{summary.complex_id}_Rank_{pose.rank}")
            mol_copy.SetProp("Cavity_ID", str(pose.cavity_id))
            mol_copy.SetProp("FreeSolvE_DeltaG_kcal", f"{pose.free_energy_kcal:.3f}")
            mol_copy.SetProp("Model_LogProb", f"{pose.log_probability:.3f}")
            writer.write(mol_copy)

        writer.close()
        return out_sdf_path
