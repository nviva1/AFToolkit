# Windsor_C98C7: predicted complexes for gamma(ref), WT, alpha, delta, omicron — RBD-aligned,
# mutated positions as sticks. Chains: A=RBD, H/L=antibody. ref=gamma (K417T,E484K,N501Y).
# WT/delta revert those at 417/484/501 (+delta adds 452,478). Run: pymol view_windsor.pml
cd reproductions/rbd_aim_fig2g/pred_pdbs_r3/restored
load Windsor_C98C7__REF_gamma.pdb,                  gamma_REF
load Windsor_C98C7__WT.pdb,                          WT
load Windsor_C98C7__delta.pdb,                        delta
load Windsor_C98C7__alpha.pdb,                        alpha
load Windsor_C98C7__omicron_BA1_B.1.1.529.pdb,        omicron

bg_color white
hide everything
show cartoon
set cartoon_transparency, 0.2

# align everything on the RBD (chain A) of the gamma reference
align WT and chain A,      gamma_REF and chain A
align delta and chain A,   gamma_REF and chain A
align alpha and chain A,   gamma_REF and chain A
align omicron and chain A, gamma_REF and chain A

color grey70,   gamma_REF
color marine,   WT
color magenta,  delta
color orange,   alpha
color forest,   omicron

# antibody faded, RBD prominent
set cartoon_transparency, 0.6, chain H or chain L

# the residues that distinguish WT/delta from the gamma reference
select keymuts, chain A and resi 417+452+478+484+501
show sticks, keymuts and not name N+C+O
set stick_radius, 0.25, keymuts
util.cnc keymuts
set label_size, 15
label gamma_REF and keymuts and name CA, "%s%s" % (resn, resi)

orient chain A and resi 417+452+478+484+501
zoom chain A and resi 417+452+478+484+501, 8
set ray_shadows, 0
set cartoon_fancy_helices, 1
ray 1600, 1100
png windsor_C98C7_overlay.png, dpi=150
