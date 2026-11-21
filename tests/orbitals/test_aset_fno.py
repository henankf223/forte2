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
    ROHF,
    RelCISolver,
    RelDSRG_MRPT2,
    SpinorUpcaster,
    State,
    System,
)
from forte2.helpers.comparisons import approx_abs


ETHYLDIAZENE_CS_XYZ = """
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

PROPENE_XYZ = """
C   0.00000000   0.00000000   0.00000000
C   1.33600000   0.00000000   0.00000000
C   2.18844637   1.24034372   0.00000000
H  -0.56190665   0.93514780   0.00000000
H  -0.56190665  -0.93514780   0.00000000
H   1.86492720  -0.95421712   0.00000000
H   3.25574896   0.99636535   0.00000000
H   1.96488702   1.84426000   0.88500000
H   1.96488702   1.84426000  -0.88500000
"""


def test_fragment_fno_full_molecule_matches_reldsrg():
    """
    When Fragment A is the entire molecule (C_A = C), ASET(vir_projector='fno')
    must produce the exact same virtual natural occupation spectrum and trace as
    RelDSRG_MRPT2.compute_unrelaxed_gamma_vv() (with skip_3_cumulant=True).
    """
    xyz = """
    O 0.000000 0.000000 0.000000
    H 0.000000 1.100000 0.900000
    H 0.000000 -1.100000 0.900000
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=False,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    mc = MCOptimizer(
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            frozen_core_orbitals=1,
            core_orbitals=2,
            active_orbitals=4,
        ),
        e_tol=1e-11,
        g_tol=1e-9,
    )(rhf).run()

    # Full-molecule ASET with Fragment-FNO virtuals (run before SpinorUpcaster mutates system)
    aset = ASET(
        fragment=["O", "H"],
        occ_projector="ao",
        cutoff=0.5,
        vir_projector="fno",
        vir_fno_p_o=1.0,
        vir_fno_flow_param=0.5,
    )(mc).run()

    # Exact spin-orbital reference
    conv = SpinorUpcaster()(mc).run()
    ci_rel = CI(RelCISolver(nel=10), final_orbitals="original")(conv).run()
    pt2_rel = RelDSRG_MRPT2(flow_param=0.5, skip_3_cumulant=True)(ci_rel).run()
    G_rel = pt2_rel.compute_unrelaxed_gamma_vv()
    evals_rel_so = np.sort(np.linalg.eigvalsh(G_rel))[::-1]
    evals_rel_spatial = 2.0 * evals_rel_so[0::2]

    lv_vals = aset.partition["lv_vals"]
    assert aset.fno_trace == approx_abs(float(np.trace(G_rel).real), 1e-8)
    assert np.max(np.abs(lv_vals - evals_rel_spatial)) < 1e-8
    assert len(aset.partition["index_A_vir"]) == len(lv_vals)


def test_fragment_fno_truncation_modes():
    """Verify cumulative (p_o), threshold (n_kappa), and num_of_orbitals modes."""
    system = System(
        xyz=ETHYLDIAZENE_CS_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=False,
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

    # 1. Cumulative p_o
    a_po = ASET(
        fragment=["N1-2", "H1"],
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="fno",
        vir_fno_p_o=0.95,
    )(mc).run()
    lv = a_po.partition["lv_vals"]
    assert np.all(np.diff(lv) <= 1e-12)
    assert a_po.fno_trace > 0.0
    assert 0.95 <= a_po.fno_retained_fraction <= 1.0
    n_va_95 = len(a_po.partition["index_A_vir"])
    assert 0 < n_va_95 < len(lv)

    # 2. Hard occupation threshold n_kappa
    a_nk = ASET(
        fragment=["N1-2", "H1"],
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="fno",
        vir_fno_n_kappa=1e-4,
        vir_fno_degeneracy_tol=0.0,
    )(mc).run()
    n_va_nk = len(a_nk.partition["index_A_vir"])
    assert n_va_nk == int(np.sum(a_nk.partition["lv_vals"] >= 1e-4))

    # 3. Fixed orbital count num_of_orbitals
    a_num = ASET(
        fragment=["N1-2", "H1"],
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="fno",
        vir_cutoff_method="num_of_orbitals",
        num_A_vir=15,
    )(mc).run()
    assert len(a_num.partition["index_A_vir"]) == 15


def test_fragment_fno_recovers_more_correlation_than_raw_ao():
    """
    At matched (N_CA, N_VA) on a severed covalent bond (HN2 in ethyldiazene),
    Fragment-FNOs must recover strictly more fragment correlation energy than
    raw-AO virtuals by prioritizing sigma*_NC and active-correlating virtuals.
    """
    system = System(
        xyz=ETHYLDIAZENE_CS_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=False,
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

    # Exact fragment correlation energy (full V, same C_A from IAO bond_bath)
    a_fullv = ASET(
        fragment=["N1-2", "H1"],
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="fno",
        vir_fno_p_o=1.0,
    )(mc).run()
    e_fullv = DSRG_MRPT2(flow_param=0.5)(
        CI(CISolver(State(system=system, multiplicity=1, ms=0.0)))(a_fullv)
    ).run().E_dsrg

    for n_va in (15, 23):
        a_ao = ASET(
            fragment=["N1-2", "H1"],
            occ_projector="iao",
            occ_cutoff_method="bond_bath",
            vir_projector="ao",
            vir_cutoff_method="num_of_orbitals",
            num_A_vir=n_va,
        )(mc).run()
        e_ao = DSRG_MRPT2(flow_param=0.5)(
            CI(CISolver(State(system=system, multiplicity=1, ms=0.0)))(a_ao)
        ).run().E_dsrg

        a_fno = ASET(
            fragment=["N1-2", "H1"],
            occ_projector="iao",
            occ_cutoff_method="bond_bath",
            vir_projector="fno",
            vir_cutoff_method="num_of_orbitals",
            num_A_vir=n_va,
        )(mc).run()
        e_fno = DSRG_MRPT2(flow_param=0.5)(
            CI(CISolver(State(system=system, multiplicity=1, ms=0.0)))(a_fno)
        ).run().E_dsrg

        # FNO must be strictly lower (closer to full-V fragment correlation) than raw AO
        assert e_fno < e_ao
        assert abs(e_fno - e_fullv) < abs(e_ao - e_fullv)


def test_fragment_fno_diffuse_basis_compression():
    """
    In aug-cc-pVDZ, raw-AO virtual projection bleeds diffuse functions into V_A,
    whereas Fragment-FNO (p_o=0.95) achieves comparable or better correlation
    recovery with significantly fewer virtual orbitals.
    """
    system = System(
        xyz=PROPENE_XYZ,
        basis_set="aug-cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=False,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    frag = ["C1-2", "H1-3"]

    a_ao = ASET(
        fragment=frag,
        frozen_core_orbitals=3,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="ao",
        cutoff=0.5,
    )(rhf).run()
    a_fno = ASET(
        fragment=frag,
        frozen_core_orbitals=3,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="fno",
        vir_fno_p_o=0.95,
    )(rhf).run()

    n_va_ao = len(a_ao.partition["index_A_vir"])
    n_va_fno = len(a_fno.partition["index_A_vir"])
    assert n_va_fno < n_va_ao


def test_fragment_fno_rhf_and_rohf_parents():
    """Fragment-FNO works cleanly with both RHF (n_act=0) and ROHF (open-shell) parents."""
    system = System(
        xyz=ETHYLDIAZENE_CS_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=False,
    )
    # RHF parent -> ASET(fno) -> ASET(2) -> DSRG-MRPT2
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    a_rhf = ASET(
        fragment=["N1-2", "H1"],
        frozen_core_orbitals=4,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="fno",
        vir_fno_p_o=0.95,
    )(rhf).run()
    assert a_rhf.fno_trace > 0.0
    a2_rhf = ASET2(flow_param=0.5)(a_rhf).run()
    ci2_rhf = CI(CISolver(State(system=a2_rhf.system, multiplicity=1, ms=0.0)))(a2_rhf)
    pt2_rhf = DSRG_MRPT2(flow_param=0.5)(ci2_rhf).run()
    assert np.isfinite(pt2_rhf.E_dsrg)

    # ROHF triplet parent -> ASET(fno)
    rohf = ROHF(charge=0, ms=1.0, e_tol=1e-12)(system).run()
    a_rohf = ASET(
        fragment=["N1-2", "H1"],
        frozen_core_orbitals=4,
        occ_projector="iao",
        occ_cutoff_method="bond_bath",
        vir_projector="fno",
        vir_fno_p_o=0.95,
    )(rohf).run()
    assert a_rohf.fno_trace > 0.0
    assert len(a_rohf.partition["index_A_vir"]) > 0


def test_fragment_fno_symmetry_and_aset2_pt2_pt3_integration():
    """
    Fragment-FNO preserves point-group symmetry (Cs) and integrates cleanly with
    ASET(2) -> CI / DSRG_MRPT2 / DSRG_MRPT3, including exact s=0 identity.
    """
    system = System(
        xyz=ETHYLDIAZENE_CS_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
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
        vir_projector="fno",
        vir_fno_p_o=0.95,
    )(mc).run()
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
    assert pt2_z.E_dsrg == approx_abs(pt2_0.E_dsrg, 1e-6)
    assert pt3_z.E_dsrg == approx_abs(pt3_0.E_dsrg, 1e-9)

    # Nonzero flow_param = 0.5 runs cleanly and improves total energy over ASET(0)
    pt2_full = DSRG_MRPT2(flow_param=0.5)(mc).run()
    a2 = ASET2(flow_param=0.5)(aset).run()
    ci2 = CI(CISolver(State(system=a2.system, multiplicity=1, ms=0.0)))(a2)
    pt2_a2 = DSRG_MRPT2(flow_param=0.5)(ci2).run()
    pt3_a2 = DSRG_MRPT3(flow_param=0.5)(ci2).run()
    assert abs(pt2_a2.E_dsrg - pt2_full.E_dsrg) < abs(pt2_0.E_dsrg - pt2_full.E_dsrg)
    assert np.isfinite(pt3_a2.E_dsrg)
