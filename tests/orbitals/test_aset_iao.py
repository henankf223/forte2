import numpy as np
import pytest

from forte2 import (
    ASET,
    ASET2,
    AVAS,
    CI,
    CISolver,
    DSRG_MRPT2,
    DSRG_MRPT3,
    MCOptimizer,
    RHF,
    State,
    System,
)
from forte2.helpers.comparisons import approx_abs


PROPENE_XYZ = """
C       -2.2314881720      2.3523969887      0.1565319638
C       -1.1287322054      1.6651786288     -0.1651010551
H       -3.2159664855      1.9109197306      0.0351701750
H       -2.1807424354      3.3645292222      0.5457999612
H       -1.2085033449      0.7043108616     -0.5330598833
C        0.2601218384      2.1970946692     -0.0290628762
H        0.7545456004      2.2023392001     -1.0052240245
H        0.8387453665      1.5599644558      0.6466877402
H        0.2749376338      3.2174213526      0.3670138598
"""

# Ethane (C1-2, H1-6) + spectator methane (C3, H7-10) separated by 4.5 A (k = 0 cuts)
ETHANE_METHANE_XYZ = """
C  0.000000  0.000000  0.000000
C  0.000000  0.000000  1.540000
H  1.031271  0.000000 -0.365225
H -0.515636  0.893107 -0.365225
H -0.515636 -0.893107 -0.365225
H  0.515636  0.893107  1.905225
H -1.031271  0.000000  1.905225
H  0.515636 -0.893107  1.905225
C  4.500000  0.000000  0.770000
H  5.131621  0.631621  1.401621
H  5.131621 -0.631621  0.138379
H  3.868379  0.631621  0.138379
H  3.868379 -0.631621  1.401621
"""

# All-trans ethyldiazene H-N1=N2-C1H2-C2H3 (k = 1 cut on N2-C1 for ["N1-2", "H1"])
ETHYLDIAZENE_XYZ = """
N  1.250000  0.000000  0.000000
N  0.000000  0.000000  0.000000
C -0.586161 -1.348096  0.000000
C -2.117912 -1.321353  0.000000
H  1.544242  0.993357  0.000000
H -0.251200 -1.905173  0.882580
H -0.251200 -1.905173 -0.882580
H -2.517737 -2.337224  0.000000
H -2.495932 -0.805860  0.886066
H -2.495932 -0.805860 -0.886066
"""

# Perfluoroethyldiazene H-N1=N2-C1F2-C2F3 (polar N2-C1F2 cut)
PERFLUORO_ETHYLDIAZENE_XYZ = """
N  1.250000  0.000000  0.000000
N  0.000000  0.000000  0.000000
C -0.586161 -1.348096  0.000000
C -2.117912 -1.321353  0.000000
H  1.544242  0.993357  0.000000
F -0.175000 -2.028000  1.080000
F -0.175000 -2.028000 -1.080000
F -2.610000 -2.565000  0.000000
F -2.580000 -0.680000  1.085000
F -2.580000 -0.680000 -1.085000
"""

# tert-Butyl hydroperoxide (CH3)3C1-O1-O2-H1 (k = 1 cut on C1-O1, or k = 3 cuts on C1-C2,3,4)
TBHP_XYZ = """
O  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.450000
H  0.942000  0.000000  1.680000
C -1.340000  0.000000 -0.520000
C -1.340000  1.250000 -1.400000
C -1.340000 -1.250000 -1.400000
C -2.500000  0.000000  0.470000
H -2.290000  1.330000 -1.940000
H -0.530000  1.210000 -2.140000
H -1.220000  2.160000 -0.800000
H -2.290000 -1.330000 -1.940000
H -0.530000 -1.210000 -2.140000
H -1.220000 -2.160000 -0.800000
H -3.450000  0.000000 -0.070000
H -2.470000  0.890000  1.100000
H -2.470000 -0.890000  1.100000
"""


@pytest.mark.parametrize("iao_include_active", [True, False])
def test_iao_partition_of_unity(iao_include_active):
    """P^{A,IAO} + P^{B,IAO} must equal I on the core space to machine precision."""
    system = System(
        xyz=PROPENE_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=8,
            active_orbitals=2,
        ),
        e_tol=1e-10,
        g_tol=1e-7,
    )(rhf).run()

    aset_a = ASET(
        fragment=["C1-2", "H1-3"],
        occ_projector="iao",
        iao_include_active=iao_include_active,
    )(mc).run()
    aset_b = ASET(
        fragment=["C3", "H4-6"],
        occ_projector="iao",
        iao_include_active=iao_include_active,
    )(mc).run()

    C_core = mc.mos.C[0][:, mc.mo_space.core_indices]
    P_sum = C_core.T @ (aset_a.P_ao_occ + aset_b.P_ao_occ) @ C_core
    np.testing.assert_allclose(P_sum, np.eye(C_core.shape[1]), atol=1e-11)

    # All eigenvalues strictly inside [0, 1]
    lo_a = aset_a.partition["lo_vals"]
    assert np.all(lo_a >= -1e-12) and np.all(lo_a <= 1.0 + 1e-12)
    # Complement eigenvalues satisfy lambda_A + lambda_B = 1
    lo_b = aset_b.partition["lo_vals"]
    np.testing.assert_allclose(np.sort(lo_a) + np.sort(lo_b)[::-1], 1.0, atol=1e-11)


def test_iao_bond_bath_exact_severed_bond_counts():
    """
    Verify IAO + bond_bath captures exactly k severed covalent bond orbitals for:
      - k = 0: ethane + methane (adds 0 extra occupied orbitals)
      - k = 1: ethyldiazene -N2H (adds 1 N-C sigma bond)
      - k = 3: tert-butyl hydroperoxide C1-O1-O2-H1 (adds 3 C-CH3 sigma bonds)
    """
    # k = 0 (noncovalent)
    sys0 = System(
        xyz=ETHANE_METHANE_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )
    rhf0 = RHF(charge=0, e_tol=1e-12)(sys0).run()
    a0_t = ASET(
        fragment=["C1-2", "H1-6"],
        frozen_core_orbitals=3,
        occ_projector="iao",
        cutoff=0.5,
    )(rhf0).run()
    a0_bb = ASET(
        fragment=["C1-2", "H1-6"],
        frozen_core_orbitals=3,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
    )(rhf0).run()
    lo0 = a0_bb.partition["lo_vals"]
    assert int(np.sum((lo0 > 0.15) & (lo0 < 0.85))) == 0
    assert len(a0_bb.partition["index_A_occ"]) == len(a0_t.partition["index_A_occ"]) == 7

    # k = 1 (single covalent cut N2-C1)
    sys1 = System(
        xyz=ETHYLDIAZENE_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )
    rhf1 = RHF(charge=0, e_tol=1e-12)(sys1).run()
    a1_bb = ASET(
        fragment=["N1-2", "H1"],
        frozen_core_orbitals=4,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
    )(rhf1).run()
    lo1 = a1_bb.partition["lo_vals"]
    # Exactly 1 severed bond orbital in (0.15, 0.85), retained in C_A
    assert int(np.sum((lo1 > 0.15) & (lo1 < 0.85))) == 1
    assert len(a1_bb.partition["index_A_occ"]) == int(np.sum(lo1 > 0.85)) + 1

    # k = 3 (three simultaneous C-C cuts in tert-butyl hydroperoxide)
    sys3 = System(
        xyz=TBHP_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )
    rhf3 = RHF(charge=0, e_tol=1e-12)(sys3).run()
    a3_bb = ASET(
        fragment=["O1-2", "H1", "C1"],
        frozen_core_orbitals=6,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
    )(rhf3).run()
    lo3 = a3_bb.partition["lo_vals"]
    # Exactly 3 severed C-CH3 bond orbitals in (0.15, 0.85), all retained in C_A
    assert int(np.sum((lo3 > 0.15) & (lo3 < 0.85))) == 3
    assert len(a3_bb.partition["index_A_occ"]) == int(np.sum(lo3 > 0.85)) + 3


def test_iao_polar_bond_log_gap_rule():
    """
    On strongly polar N-CF2 in perfluoroethyldiazene, the severed N-C bond
    orbital is kept cleanly by occ_cutoff_method="bond_bath".
    """
    system = System(
        xyz=PERFLUORO_ETHYLDIAZENE_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    aset_bb = ASET(
        fragment=["N1-2", "H1"],
        frozen_core_orbitals=9,  # 2 N + 2 C + 5 F 1s
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
    )(rhf).run()

    lo = np.sort(aset_bb.partition["lo_vals"])[::-1]
    n_bulk = int(np.sum(lo > 0.85))
    n_bond = int(np.sum((lo >= 0.15) & (lo <= 0.85)))
    assert n_bond == 1
    assert len(aset_bb.partition["index_A_occ"]) == n_bulk + 1
    assert aset_bb.bond_bath_gap > 1.5


def test_iao_diffuse_basis_stability():
    """
    In aug-cc-pVDZ vs cc-pVDZ on propene, IAO occupied eigenvalues remain
    virtually basis-independent while raw-AO eigenvalues bleed into environment B.
    """
    iao_evals = {}
    ao_evals = {}
    for basis in ("cc-pvdz", "aug-cc-pvdz"):
        system = System(
            xyz=PROPENE_XYZ,
            basis_set=basis,
            auxiliary_basis_set="cc-pvtz-jkfit",
        )
        rhf = RHF(charge=0, e_tol=1e-12)(system).run()
        a_iao = ASET(
            fragment=["C1-2", "H1-3"],
            frozen_core_orbitals=3,
            occ_projector="iao",
            occ_cutoff_method="bond_bath",
        )(rhf).run()
        a_ao = ASET(
            fragment=["C1-2", "H1-3"],
            frozen_core_orbitals=3,
            occ_projector="ao",
            cutoff=0.5,
        )(rhf).run()
        iao_evals[basis] = np.sort(a_iao.partition["lo_vals"])[::-1]
        ao_evals[basis] = np.sort(a_ao.partition["lo_vals"])[::-1]

    # IAO occupied spectrum is nearly identical between cc-pVDZ and aug-cc-pVDZ
    max_diff_iao = float(np.max(np.abs(iao_evals["cc-pvdz"] - iao_evals["aug-cc-pvdz"])))
    max_diff_ao = float(np.max(np.abs(ao_evals["cc-pvdz"] - ao_evals["aug-cc-pvdz"])))
    assert max_diff_iao < 0.03
    assert max_diff_iao < max_diff_ao


def test_iao_symmetry_and_aset2_integration():
    """
    Verify IAO + bond_bath preserves point-group symmetry (C_s in ethyldiazene)
    and integrates cleanly with ASET2 -> CI / DSRG-MRPT2 / DSRG-MRPT3.
    """
    system = System(
        xyz=ETHYLDIAZENE_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
        symmetry=True,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    avas = AVAS(subspace=["N1-2(2px)", "N1-2(2pz)"], cas=(4, 4))(rhf).run()
    mc = MCOptimizer(
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            frozen_core_orbitals=avas.mo_space.core_indices[:4],
            core_orbitals=avas.mo_space.core_indices[4:],
            active_orbitals=avas.mo_space.active_indices,
        ),
        e_tol=1e-10,
        g_tol=1e-8,
    )(avas).run()

    aset = ASET(
        fragment=["N1-2", "H1"],
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        cutoff=0.5,
    )(mc).run()
    # Symmetry preserved: all orbitals retain valid Cs irrep labels
    assert len(aset.mos.irrep_indices[0]) == aset.nmo

    # s = 0 identity with ASET(2) -> CI / DSRG_MRPT2 / DSRG_MRPT3
    ci0 = CI(CISolver(State(system=system, multiplicity=1, ms=0.0)))(aset)
    pt2_0 = DSRG_MRPT2(flow_param=0.5)(ci0).run()
    pt3_0 = DSRG_MRPT3(flow_param=0.5)(ci0).run()

    a2_zero = ASET2(flow_param=0.0)(aset).run()
    ci2_0 = CI(CISolver(State(system=a2_zero.system, multiplicity=1, ms=0.0)))(a2_zero)
    pt2_z = DSRG_MRPT2(flow_param=0.5)(ci2_0).run()
    pt3_z = DSRG_MRPT3(flow_param=0.5)(ci2_0).run()

    assert ci2_0.E == approx_abs(ci0.E, 1e-11)
    # Standard DF DSRG_MRPT2 uses renormalize_3index on off-diagonal active 1-RDM,
    # while ASET2's dense path evaluates exact Vtilde * S2 (~2e-7 Eh difference).
    assert pt2_z.E_dsrg == approx_abs(pt2_0.E_dsrg, 1e-6)
    assert pt3_z.E_dsrg == approx_abs(pt3_0.E_dsrg, 1e-9)

    # Nonzero flow_param = 0.5 runs cleanly and improves absolute energy over ASET(0)
    pt2_full = DSRG_MRPT2(flow_param=0.5)(mc).run()
    a2 = ASET2(flow_param=0.5)(aset).run()
    ci2 = CI(CISolver(State(system=a2.system, multiplicity=1, ms=0.0)))(a2)
    pt2_a2 = DSRG_MRPT2(flow_param=0.5)(ci2).run()
    pt3_a2 = DSRG_MRPT3(flow_param=0.5)(ci2).run()
    assert abs(pt2_a2.E_dsrg - pt2_full.E_dsrg) < abs(pt2_0.E_dsrg - pt2_full.E_dsrg)
    assert np.isfinite(pt3_a2.E_dsrg)
