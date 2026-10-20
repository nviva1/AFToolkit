# 510A5: input X-ray crystal vs the 3 predicted complexes (r5, 6 cycles), RBD-aligned,
# with the 16 mutated/masked RBD positions shown as sticks.
# Chains: A = RBD, E = heavy, D = light.  Reference = omicron BA.1 (best K_D).
# Run:  pymol view_510A5.pml      (or headless: pymol -cq view_510A5.pml)

load x_ray_pdbs/510A5_7WS5_EDA.pdb,                 XRAY_input
load pred_pdbs_r5/restored/510A5__REF_omicron_BA1_B.1.1.529.pdb, PRED_omicron_ref
load pred_pdbs_r5/restored/510A5__WT.pdb,           PRED_WT
load pred_pdbs_r5/restored/510A5__delta.pdb,        PRED_delta

bg_color white
hide everything
show cartoon
set cartoon_transparency, 0.1

# align everything onto the input crystal RBD (chain A)
align PRED_omicron_ref and chain A, XRAY_input and chain A
align PRED_WT and chain A,          XRAY_input and chain A
align PRED_delta and chain A,       XRAY_input and chain A

# colour per object: input grey, predictions distinct
color grey70,   XRAY_input
color forest,   PRED_omicron_ref
color marine,   PRED_WT
color magenta,  PRED_delta

# the 16 mutated / masked RBD positions (chain A)
select muts, chain A and resi 339+371+373+375+417+440+446+452+477+478+484+493+496+498+501+505
show sticks, muts and not name N+C+O
set stick_radius, 0.25, muts
color yellow, muts and elem C and XRAY_input
util.cnc muts

set label_size, 14
label XRAY_input and muts and name CA, "%s%s" % (resn, resi)

# de-emphasise the antibody so the RBD + mutations read clearly
set cartoon_transparency, 0.6, chain E or chain D
orient chain A
zoom muts, 6
set ray_shadows, 0
set cartoon_fancy_helices, 1
ray 1600, 1100
png pred_pdbs_r5/restored/510A5_overlay.png, dpi=150
