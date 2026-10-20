"""Defensive validation for the RBD-AIM reproduction. Every build/crop/mutation
step should pass through these asserts so a numbering / residue / chain mistake
fails LOUDLY instead of silently corrupting a point.

All functions raise AssertionError with a descriptive message on violation.
"""
from Bio import Align

# canonical SARS-CoV-2 RBD residue -> 1-letter (Wuhan), for numbering sanity
CANON_RBD = {333: "T", 417: "K", 440: "N", 446: "G", 452: "L", 478: "T",
             484: "E", 493: "Q", 501: "N", 505: "Y", 519: "H"}
AA3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
          'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
          'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


def _ident(a, b):
    al = Align.PairwiseAligner(); al.mode = "global"
    al.open_gap_score = -10; al.extend_gap_score = -0.5
    aln = al.align(a, b)[0]
    n = m = 0
    for x, y in zip(aln[0], aln[1]):
        if x != "-" and y != "-":
            n += 1; m += (x == y)
    return m / n if n else 0.0


def assert_rbd_region(seq, wuhan, tag, min_len=180, max_len=210, min_id=0.85):
    """RBD crop is the right length and is genuinely an RBD (high identity to Wuhan)."""
    assert min_len <= len(seq) <= max_len, \
        f"[{tag}] RBD crop length {len(seq)} outside [{min_len},{max_len}] -- wrong chain/crop?"
    idv = _ident(seq, wuhan)
    assert idv >= min_id, \
        f"[{tag}] RBD crop identity to Wuhan {idv:.2f} < {min_id} -- not the RBD (wrong chain)?"
    return idv


def assert_canonical_numbering(num2aa, tag, min_present=8):
    """Canonical SARS-CoV-2 numbering is in use: the well-known RBD positions
    (417, 484, 501, 505, ...) are PRESENT in the crop. We check presence + valid
    residue, NOT WT identity -- variant crystals (omicron etc.) legitimately carry
    mutated residues there, so requiring WT would wrongly reject them."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    present = sum(1 for num in CANON_RBD if num2aa.get(num) in valid)
    assert present >= min_present, \
        f"[{tag}] only {present}/{len(CANON_RBD)} canonical RBD positions present -- numbering is off / wrong region"
    # also: most positions should be WT OR a single plausible substitution (sanity, not strict)
    return present


def assert_unique_residues(resnums, tag):
    """No duplicate residue numbers within a chain (insertion codes must be resolved)."""
    dup = [r for r in set(resnums) if resnums.count(r) > 1] if isinstance(resnums, list) else []
    assert not dup, f"[{tag}] duplicate residue numbers {dup[:5]} -- insertion codes not collapsed"


def assert_chain_is_antibody(seq, tag, variable_span_fn):
    """Chain is a real antibody variable domain (ANARCII recognises it) and is V-domain sized."""
    sp = variable_span_fn(seq)
    assert sp is not None, f"[{tag}] ANARCII did not recognise an antibody variable domain"
    vlen = sp[1] - sp[0] + 1
    assert 95 <= vlen <= 135, f"[{tag}] VH/VL span {vlen} outside [95,135] -- wrong chain?"
    return sp


def assert_built_matches_target(built_seq, target_seq, tag, min_id=0.995):
    """After mutate/thread, the RBD sequence equals the intended strain (modulo crystal gaps)."""
    idv = _ident(built_seq, target_seq)
    assert idv >= min_id, \
        f"[{tag}] built RBD identity to target strain {idv:.3f} < {min_id} -- mutation/threading wrong"
    return idv


def assert_wt_mt_strains(wt_seq, mt_seq, wuhan, strain, tag):
    """Hard guarantee the two endpoints are exactly what they should be:
      * WT RBD sequence == Wuhan,
      * MT RBD sequence == the strain,
      * WT and MT differ at EXACTLY the strain's mutation positions (no more, no less).
    Catches construct contamination, mis-mutation, indel/numbering errors."""
    assert len(wt_seq) == len(wuhan), f"[{tag}] WT len {len(wt_seq)} != Wuhan {len(wuhan)}"
    assert wt_seq == wuhan, \
        f"[{tag}] WT RBD != Wuhan ({sum(a!=b for a,b in zip(wt_seq,wuhan))} mismatches) -- WT not canonicalized"
    assert mt_seq == strain, \
        f"[{tag}] MT RBD != strain ({sum(a!=b for a,b in zip(mt_seq,strain))} mismatches) -- wrong mutation"
    wt_mt_diff = {i for i in range(len(wt_seq)) if wt_seq[i] != mt_seq[i]}
    strain_diff = {i for i in range(len(wuhan)) if wuhan[i] != strain[i]}
    assert wt_mt_diff == strain_diff, \
        f"[{tag}] WT->MT changes {sorted(wt_mt_diff)} != strain mutations {sorted(strain_diff)} -- contamination"
    return len(strain_diff)


def assert_interface(labels, group_a, group_b, tag):
    """Both interface sides are present in the residue labels."""
    la = sum(1 for x in labels if x in group_a)
    lb = sum(1 for x in labels if x in group_b)
    assert la > 0 and lb > 0, f"[{tag}] empty interface side: |A|={la} |B|={lb}"
    return la, lb


def assert_distinct_chains(chains, tag):
    """RBD chain id and antibody chain ids are distinct (no collisions after relabel)."""
    assert len(set(chains)) == len(chains), f"[{tag}] chain id collision in {chains}"


def assert_dlogkd(kd_variant, kd_wt, dlogkd, tag, tol=1e-3):
    """A relative affinity value is correct: ΔlogKD == log10(K_D_variant / K_D_WT),
    with both K_D positive and finite. Catches sign/ratio/inverted-reference errors."""
    import math
    assert kd_wt is not None and kd_wt == kd_wt and kd_wt > 0, f"[{tag}] invalid WT K_D = {kd_wt}"
    assert kd_variant is not None and kd_variant == kd_variant and kd_variant > 0, \
        f"[{tag}] invalid variant K_D = {kd_variant}"
    expect = math.log10(kd_variant / kd_wt)
    assert abs(expect - dlogkd) < tol, \
        f"[{tag}] ΔlogKD {dlogkd:.4f} != log10({kd_variant}/{kd_wt}) = {expect:.4f}"
    return expect
