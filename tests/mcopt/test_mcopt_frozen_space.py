import time
import numpy as np

from forte2 import (
    ASET,
    ASET2,
    AVAS,
    CI,
    CISolver,
    DSRG_MRPT2,
    MCOptimizer,
    RHF,
    State,
    System,
)
from forte2.helpers.comparisons import approx_abs
from forte2.mcopt.orbital_optimizer import OrbOptimizer


ETHYLDIAZENE_XYZ = """
N   1.25000000   0.00000000   0.00000000
N   0.00000000   0.00000000   0.00000000
C  -0.58616939  -1.34808585   0.00000000
C  -2.11805410  -1.36658654   0.00000000
H   1.54423960   0.99335756   0.00000000
H  -0.23309926  -1.88351979   0.88346326
H  -0.23309926  -1.88351979  -0.88346326
H  -2.48797723  -2.39359337   0.00000000
H  -2.49593261  -0.85886526   0.88918678
H  -2.49593261  -0.85886526  -0.88918678
"""


def test_frozen_space_mcopt_matches_full_matrix_oracle_and_preserves_frozen_mos():
    """
    Verify that frozen-space MCOptimizer (caching F_frz_ao and restricting
    OrbOptimizer to mo_space.corr) reproduces the exact energy, active-space
    integrals, and orbital gradient of the un-optimized full-MO OrbOptimizer,
    and leaves frozen_core and frozen_virt columns bit-identical.
    """
    system = System(
        xyz=ETHYLDIAZENE_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=False,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    avas = AVAS(subspace=["N1-2(2px)", "N1-2(2pz)"], cas=(4, 4))(rhf).run()
    aset = ASET(
        fragment=["N1-2", "H1"],
        frozen_core_orbitals=4,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        cutoff=0.5,
    )(avas).run()

    C_before = aset.mos.C[0].copy()
    frz_c = aset.mo_space.frozen_core_indices
    frz_v = aset.mo_space.frozen_virtual_indices
    assert len(frz_c) > 4 and len(frz_v) > 10

    # Optimized Fragment-CASSCF (W2: RHF -> AVAS -> ASET -> MCOptimizer)
    # Use final_orbitals="original" so post-hoc semicanonicalization does not
    # rotate frozen_core (1s + O_B) internally; also test default semicanonical below.
    mc_frag = MCOptimizer(
        CISolver(State(system=system, multiplicity=1, ms=0.0)),
        e_tol=1e-10,
        g_tol=1e-8,
        maxiter=60,
        final_orbitals="original",
    )(aset).run()
    assert mc_frag.converged

    # Frozen columns must remain bit-identical
    assert np.allclose(mc_frag.mos.C[0][:, frz_c], C_before[:, frz_c], atol=1e-14)
    assert np.allclose(mc_frag.mos.C[0][:, frz_v], C_before[:, frz_v], atol=1e-14)

    # Default semicanonical final_orbitals must reach identical energy and preserve frozen subspaces
    mc_frag_sc = MCOptimizer(
        CISolver(State(system=system, multiplicity=1, ms=0.0)),
        e_tol=1e-10,
        g_tol=1e-8,
        maxiter=60,
    )(aset).run()
    assert mc_frag_sc.E == approx_abs(mc_frag.E, 1e-11)
    S = system.ints_overlap()
    for blk in (frz_c, frz_v):
        ovlp = C_before[:, blk].T @ S @ mc_frag_sc.mos.C[0][:, blk]
        assert np.allclose(ovlp @ ovlp.T, np.eye(len(blk)), atol=1e-12)

    # Cross-check Ecore, Fcore[actv, actv], and orbgrad against un-optimized full-MO OrbOptimizer
    mo_space = mc_frag.mo_space
    C_full = mc_frag.mos.C[0][:, mo_space.orig_to_contig].copy()
    full_opt = OrbOptimizer(
        C_full,
        (mo_space.docc, mo_space.actv, mo_space.uocc),
        system.fock_builder,
        system.ints_hcore(),
        system.nuclear_repulsion,
        mc_frag.nrr,
    )
    g1 = mc_frag.make_average_rdm(1)
    g2 = mc_frag.make_average_rdm(2)
    full_opt.set_rdms(g1, g2)
    full_opt._compute_Fcore()
    full_opt.get_eri_gaaa()
    e_full_oracle = full_opt._compute_reference_energy()
    grad_full_oracle = full_opt.gradient(np.zeros(full_opt.nrot))

    assert mc_frag.E == approx_abs(e_full_oracle, 1e-11)
    # Exact element-wise match between optimized corr-only gradient and full-MO oracle gradient
    mc_frag.orb_opt.set_rdms(g1, g2)
    grad_corr = mc_frag.orb_opt.gradient(np.zeros(mc_frag.orb_opt.nrot))
    assert np.allclose(grad_full_oracle, grad_corr, atol=1e-11)
    assert np.sqrt(np.mean(grad_full_oracle**2)) < 1e-7

    # Also verify downstream ASET(2) + DSRG-MRPT2 on Fragment-CASSCF (W2 -> ASET(2))
    a2 = ASET2(flow_param=0.5)(mc_frag).run()
    ci2 = CI(CISolver(State(system=a2.system, multiplicity=1, ms=0.0)))(a2)
    pt2_a2 = DSRG_MRPT2(flow_param=0.5)(ci2).run()
    assert np.isfinite(pt2_a2.E_dsrg)
