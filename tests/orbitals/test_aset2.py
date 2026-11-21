import numpy as np
import pytest

from forte2 import (
    ASET,
    ASET2,
    CI,
    CISolver,
    DSRG_MRPT2,
    DSRG_MRPT3,
    MCOptimizer,
    RHF,
    State,
    System,
)
from forte2.dsrg.dsrg_common import _DSRGHelper
from forte2.dsrg.utils import (
    compute_t1_block,
    compute_t2_block,
    degno_active_sf,
    renormalize_V_block,
)
from forte2.helpers.comparisons import approx, approx_abs


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

HNNF_XYZ = """
N       -1.1226987119      2.0137160725     -0.0992218410
N       -0.1519067161      1.2402226172     -0.0345618482
H        0.7253474870      1.7181546089     -0.2678695726
F       -2.2714806355      1.3880717623      0.2062454513
"""


def _naive_dense_giant_active_aset2(aset, mc, flow_param=0.5, include_env_corr=True):
    """
    Reference ASET(2) implementation using the naive "giant active space"
    construction (Paper 2 Sec. 2.7 & legacy_forte) backed by _DSRGHelper with
    store_large=True and dense 4-index integrals over the full correlated space.
    Used strictly as an independent mathematical oracle on small molecules.
    """
    emb = aset.emb_space
    semican = aset.semican
    C_semi = semican.C_semican
    fock_full = semican.fock_semican
    eps_full = semican.eps_semican
    Uactv = semican.Uactv

    g1_orig = mc.ci_solver.make_average_rdm(1)
    l2_orig = mc.ci_solver.make_average_cumulant(2)
    g1_a = np.einsum("ip,ij,jq->pq", Uactv, g1_orig, Uactv.conj(), optimize=True)
    l2_a = np.einsum(
        "ip,jq,ijkl,kr,ls->pqrs",
        Uactv,
        Uactv,
        l2_orig,
        Uactv.conj(),
        Uactv.conj(),
        optimize=True,
    )

    sl_ob = emb.B_core
    sl_ca_full = emb.A_core
    sl_aa_full = emb.actv
    sl_va_full = emb.A_virt
    sl_vb = emb.B_virt
    sl_corr = slice(sl_ob.start, sl_vb.stop)

    nc = sl_ob.stop - sl_ob.start
    nc_a = sl_ca_full.stop - sl_ca_full.start
    na_a = sl_aa_full.stop - sl_aa_full.start
    nv_a = sl_va_full.stop - sl_va_full.start
    na = nc_a + na_a + nv_a
    nv = sl_vb.stop - sl_vb.start
    sl_aa = slice(nc_a, nc_a + na_a)

    # Dense padded gamma1, eta1, lambda2 over all N_A fragment orbitals
    g1_pad = np.zeros((na, na))
    g1_pad[:nc_a, :nc_a] = 2.0 * np.eye(nc_a)
    g1_pad[sl_aa, sl_aa] = g1_a
    e1_pad = 2.0 * np.eye(na) - g1_pad
    l2_pad = np.zeros((na, na, na, na))
    l2_pad[sl_aa, sl_aa, sl_aa, sl_aa] = l2_a
    cumulants = {"gamma1": g1_pad, "eta1": e1_pad, "lambda2": l2_pad, "lambda3": None}

    fock = fock_full[sl_corr, sl_corr].copy()
    eps = eps_full[sl_corr].copy()
    c = slice(0, nc)
    a = slice(nc, nc + na)
    v = slice(nc + na, nc + na + nv)
    h = slice(0, nc + na)
    p = slice(nc, nc + na + nv)

    class _DummyDSRG:
        pass

    dummy = _DummyDSRG()
    dummy.hc = c
    dummy.ha = a
    dummy.pa = slice(0, na)
    dummy.pv = slice(na, na + nv)
    helper = _DSRGHelper(dummy)

    # Full dense 4-index ERI on correlated space
    C_corr = C_semi[:, sl_corr]
    V_full = aset.system.fock_builder.two_electron_integrals_block(C_corr)
    sp = {"c": c, "a": a, "v": v}
    V = {}
    for key in ("vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "aaaa", "vvcc", "vvac", "vacc"):
        V[key] = np.ascontiguousarray(V_full[sp[key[0]], sp[key[1]], sp[key[2]], sp[key[3]]])

    eps_d = {"c": eps[c], "a": eps[a], "v": eps[v]}
    T2, S2 = {}, {}
    for key in ("aavv", "ccaa", "caav", "acav", "aava", "caaa", "ccvv", "acvv", "ccva"):
        vkey = key[2:] + key[:2]
        T2[key] = V[vkey].transpose(2, 3, 0, 1).copy()
        compute_t2_block(T2[key], *(eps_d[k] for k in key), flow_param)
    if not include_env_corr:
        T2["ccvv"][:] = 0.0

    S2["aavv"] = 2 * T2["aavv"] - T2["aavv"].swapaxes(2, 3)
    S2["ccaa"] = 2 * T2["ccaa"] - T2["ccaa"].swapaxes(2, 3)
    S2["caav"] = 2 * T2["caav"] - T2["acav"].swapaxes(0, 1)
    S2["acav"] = 2 * T2["acav"] - T2["caav"].swapaxes(0, 1)
    S2["aava"] = 2 * T2["aava"] - T2["aava"].swapaxes(0, 1)
    S2["caaa"] = 2 * T2["caaa"] - T2["caaa"].swapaxes(2, 3)
    S2["ccvv"] = 2 * T2["ccvv"] - T2["ccvv"].swapaxes(2, 3)
    S2["acvv"] = 2 * T2["acvv"] - T2["acvv"].swapaxes(2, 3)
    S2["ccva"] = 2 * T2["ccva"] - T2["ccva"].swapaxes(0, 1)

    faa = np.diag(eps[a])
    T1 = fock[h, p].copy()
    hc, ha, pa, pv = dummy.hc, dummy.ha, dummy.pa, dummy.pv
    T1[hc, pa] += 0.5 * np.einsum("ivaw,wu,uv->ia", S2["caaa"], faa, g1_pad, optimize=True)
    T1[hc, pv] += 0.5 * np.einsum("vmwe,wu,uv->me", S2["acav"], faa, g1_pad, optimize=True)
    T1[ha, pv] += 0.5 * np.einsum("ivaw,wu,uv->ia", S2["aava"], faa, g1_pad, optimize=True)
    T1[hc, pa] -= 0.5 * np.einsum("iwau,vw,uv->ia", S2["caaa"], faa, g1_pad, optimize=True)
    T1[hc, pv] -= 0.5 * np.einsum("wmue,vw,uv->me", S2["acav"], faa, g1_pad, optimize=True)
    T1[ha, pv] -= 0.5 * np.einsum("iwau,vw,uv->ia", S2["aava"], faa, g1_pad, optimize=True)
    compute_t1_block(T1, eps[h], eps[p], flow_param)
    T1[ha, pa] = 0.0
    if not include_env_corr:
        T1[hc, pv] = 0.0

    F_tilde = fock[p, h].copy()
    F_tilde[pa, hc] += 0.5 * np.einsum("ivaw,wu,uv->ai", S2["caaa"], faa, g1_pad, optimize=True)
    F_tilde[pa, hc] -= 0.5 * np.einsum("iwau,vw,uv->ai", S2["caaa"], faa, g1_pad, optimize=True)
    F_tilde[pv, hc] += 0.5 * np.einsum("vmwe,wu,uv->em", S2["acav"], faa, g1_pad, optimize=True)
    F_tilde[pv, hc] -= 0.5 * np.einsum("wmue,vw,uv->em", S2["acav"], faa, g1_pad, optimize=True)
    F_tilde[pv, ha] += 0.5 * np.einsum("ivaw,wu,uv->ai", S2["aava"], faa, g1_pad, optimize=True)
    F_tilde[pv, ha] -= 0.5 * np.einsum("iwau,vw,uv->ai", S2["aava"], faa, g1_pad, optimize=True)
    delta_ph = -eps[p][:, None] + eps[h][None, :]
    F_tilde *= np.exp(-flow_param * delta_ph**2)
    F_tilde += fock[p, h]

    for key in ("vvaa", "aacc", "avca", "avac", "vaaa", "aaca", "vvcc", "vvac", "vacc"):
        renormalize_V_block(V[key], *(eps_d[k] for k in key), flow_param)

    dE = helper.evaluate_H_T_C0(
        T1, {"T2": T2, "S2": S2}, F_tilde, V, cumulants, store_large=True
    )
    C1 = helper.H_T_C1_active(T1, T2, S2, F_tilde, V, g1_pad, e1_pad, l2_pad, store_large=True)
    C2 = helper.H_T_C2_active(T1, T2, S2, F_tilde, V, g1_pad, e1_pad)

    hbar1_no = fock[a, a].copy() + 0.5 * (C1 + C1.T)
    hbar2_no = V["aaaa"].copy() + 0.5 * (C2 + C2.transpose(2, 3, 0, 1))
    E_scalar, hbar1 = degno_active_sf(mc.E + dE, hbar1_no, hbar2_no, g1_pad, l2_pad)
    return E_scalar, hbar1, hbar2_no, mc.E + dE


def test_aset2_zero_flow_reduces_to_aset0():
    """At flow_param=0.0, ASET(2)-[CI/MRPT2/MRPT3] must exactly equal ASET(0)."""
    system = System(
        xyz=PROPENE_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(State(nel=24, multiplicity=1, ms=0.0), core_orbitals=11, active_orbitals=2),
        e_tol=1e-10,
        g_tol=1e-10,
    )(rhf)
    aset = ASET(fragment=["C1-2", "H1-3"], cutoff_method="threshold", cutoff=0.5)(mc)

    # ASET(0) path
    ci0 = CI(CISolver(State(system=system, multiplicity=1, ms=0.0)))(aset)
    pt2_0 = DSRG_MRPT2(flow_param=0.5)(ci0)
    pt3_0 = DSRG_MRPT3(flow_param=0.5)(ci0)
    pt2_0.run()
    pt3_0.run()

    # ASET(2) with flow_param=0.0
    aset2 = ASET2(flow_param=0.0)(aset)
    ci2 = CI(CISolver(State(system=None, nel=16, multiplicity=1, ms=0.0)))
    # Resolve nel_frag dynamically after aset2.run()
    aset2.run()
    assert aset2.delta_E_AB == approx_abs(0.0, 1e-13)
    assert aset2.delta_E_B == approx_abs(0.0, 1e-13)
    assert aset2.E_ref == approx_abs(mc.E, 1e-12)

    ci2 = CI(CISolver(State(system=aset2.system, multiplicity=1, ms=0.0)))(aset2)
    pt2_2 = DSRG_MRPT2(flow_param=0.5)(ci2)
    pt3_2 = DSRG_MRPT3(flow_param=0.5)(ci2)
    pt2_2.run()
    pt3_2.run()

    assert ci2.E == approx_abs(ci0.E, 1e-11)
    assert pt2_2.E_dsrg == approx_abs(pt2_0.E_dsrg, 1e-11)
    assert pt3_2.E_dsrg == approx_abs(pt3_0.E_dsrg, 1e-11)


def test_aset2_all_active_reduces_to_pr_dsrg_mrpt2():
    """
    When C_A = empty and V_A = empty (all fragment orbitals are active),
    ASET(2)-[CASCI] must match pr-DSRG-MRPT2 (relax_reference='once', skip_3_cumulant=True).
    """
    system = System(
        xyz=HNNF_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    mc.run()

    # Run ASET + ASET(2) BEFORE pt2_ref relaxes mc.ci_solver in-place!
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_A_occ=0,
        num_A_vir=0,
    )(mc)
    aset2 = ASET2(flow_param=0.5, skip_3_cumulant=True)(aset)
    aset2.run()
    ci2 = CI(CISolver(State(system=aset2.system, multiplicity=1, ms=0.0)))(aset2)
    ci2.run()

    # Reference pr-DSRG-MRPT2
    pt2_ref = DSRG_MRPT2(flow_param=0.5, relax_reference="once", skip_3_cumulant=True)(mc)
    pt2_ref.run()

    # Unrelaxed energy matches DSRG-MRPT2 fixed-reference energy (tiny ~3e-7 difference
    # solely from the renormalize_3index approximation for off-diag g1 in DF MRPT2)
    assert aset2.E_ref == approx_abs(mc.E, 1e-12)
    assert aset2.E_aset2_ref == approx_abs(pt2_ref.relax_energies[0, 0], 1e-6)
    # Relaxed CASCI on Gbar matches partially-relaxed DSRG-MRPT2
    assert ci2.E == approx_abs(pt2_ref.relax_energies[0, 1], 1e-5)

    # Also verify exact lambda_3 contribution (Paper 2 Eq. 35) matches full DSRG-MRPT2
    mc2 = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    mc2.run()
    aset_l3 = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_A_occ=0,
        num_A_vir=0,
    )(mc2)
    aset2_l3 = ASET2(flow_param=0.5, skip_3_cumulant=False)(aset_l3).run()
    pt2_l3 = DSRG_MRPT2(flow_param=0.5, skip_3_cumulant=False)(mc2).run()
    assert aset2_l3.E_aset2_ref == approx_abs(pt2_l3.E_dsrg, 1e-6)
    # The lambda_3 shift itself must match to < 1e-11!
    dl3_aset2 = aset2_l3.E_aset2_ref - aset2.E_aset2_ref
    dl3_pt2 = pt2_l3.E_dsrg - pt2_ref.relax_energies[0, 0]
    assert dl3_aset2 == approx_abs(dl3_pt2, 1e-11)


@pytest.mark.parametrize("include_env_corr", [True, False])
def test_aset2_matches_naive_giant_active_oracle(include_env_corr):
    """
    Element-by-element verification of E_scalar, hbar1, hbar2, and E_aset2_ref
    against the naive giant-active-space DSRG-MRPT2 oracle on HNNF (STO-3G).
    """
    system = System(
        xyz=HNNF_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_A_occ=5,
        num_A_vir=1,
    )(mc)
    aset2 = ASET2(flow_param=0.5, include_env_corr=include_env_corr)(aset)
    aset2.run()

    E_sc_ref, h1_ref, h2_ref, E_tot_ref = _naive_dense_giant_active_aset2(
        aset, mc, flow_param=0.5, include_env_corr=include_env_corr
    )

    assert aset2.E_aset2_ref == approx_abs(E_tot_ref, 1e-11)
    assert aset2.E_scalar == approx_abs(E_sc_ref, 1e-11)
    np.testing.assert_allclose(aset2.hbar1, h1_ref, atol=1e-11)
    np.testing.assert_allclose(aset2.hbar2, h2_ref, atol=1e-11)


def test_aset2_full_molecule_matches_unembedded():
    """When fragment A = all atoms (O_B=empty, V_B=empty), ASET(2)-[MR] == full MR."""
    system = System(
        xyz=HNNF_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    pt2_full = DSRG_MRPT2(flow_param=0.5)(mc)
    pt3_full = DSRG_MRPT3(flow_param=0.5)(mc)
    pt2_full.run()
    pt3_full.run()

    aset = ASET(fragment=["N", "H", "F"], cutoff_method="threshold", cutoff=0.01)(mc)
    aset2 = ASET2(flow_param=0.5)(aset)
    aset2.run()
    ci2 = CI(CISolver(State(system=aset2.system, multiplicity=1, ms=0.0)))(aset2)
    pt2_emb = DSRG_MRPT2(flow_param=0.5)(ci2)
    pt3_emb = DSRG_MRPT3(flow_param=0.5)(ci2)
    pt2_emb.run()
    pt3_emb.run()

    assert aset2.delta_E_AB == approx_abs(0.0, 1e-13)
    assert aset2.delta_E_B == approx_abs(0.0, 1e-13)
    assert ci2.E == approx_abs(mc.E, 1e-11)
    # Note: pt2_emb uses exact dense Vtilde*S2 for cavv/ccav, whereas pt2_full's DF
    # path uses the renormalize_3index shortcut (differs by ~3e-7 for off-diag g1).
    assert pt2_emb.E_dsrg == approx_abs(pt2_full.E_dsrg, 1e-6)
    assert pt3_emb.E_dsrg == approx_abs(pt3_full.E_dsrg, 1e-9)


def test_aset_and_aset2_with_rhf_parent():
    """Verify ASET(0) and ASET(2) work cleanly starting from an RHF reference."""
    system = System(
        xyz=PROPENE_XYZ,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    aset = ASET(fragment=["C1-2", "H1-3"], cutoff_method="threshold", cutoff=0.5)(rhf)
    aset.run()
    assert aset.mo_space.nactv == 0

    # Define a (2e, 2o) active space inside the RHF-partitioned fragment A:
    # highest occupied in C_A and lowest virtual in V_A
    c_a = aset.partition["index_A_occ"]
    v_a = aset.partition["index_A_vir"]
    ci_ref = CI(
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            nroots=2,
            frozen_core_orbitals=aset.mo_space.frozen_core_indices,
            core_orbitals=c_a[:-1],
            active_orbitals=[c_a[-1], v_a[0]],
            frozen_virtual_orbitals=aset.mo_space.frozen_virtual_indices,
        )
    )(aset)
    ci_ref.run()

    # Run state-specific ASET(2) for S0 (root=0) and S1 (root=1)
    energies = []
    for root in (0, 1):
        aset2 = ASET2(flow_param=0.5, root=root)(ci_ref)
        aset2.run()
        ci_frag = CI(
            CISolver(State(system=aset2.system, multiplicity=1, ms=0.0), nroots=2)
        )(aset2)
        pt2 = DSRG_MRPT2(flow_param=0.5)(ci_frag)
        pt2.run()
        energies.append(pt2.E_dsrg)
    assert len(energies) == 2
    assert np.isfinite(energies[0]) and np.isfinite(energies[1])
