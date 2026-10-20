# AFToolkit: 2OCJ.pdb A E285K
# B-factor column holds mutant-minus-wildtype pLDDT, written onto the
# mutant structure. Red = confidence lost on mutation, blue = gained.
# Residues with no counterpart across the pair (indels) are 0.

load E285K_delta.pdb, delta
hide everything
bg_color white
show cartoon, delta
set cartoon_transparency, 0.15, delta

# diverging ramp, symmetric about zero so the midpoint is "no change"
spectrum b, red_white_blue, delta and polymer, minimum=-5.170, maximum=5.170

# the mutated site(s)
show sticks, delta and (resi 285) and not hydrogens
color yellow, delta and (resi 285) and elem C
set stick_radius, 0.18
label delta and (resi 285) and name CA, "%s%s" % (resn, resi)
set label_size, 16
set label_color, black

orient delta
orient delta and (resi 285)
zoom delta and (resi 285), 8
set ray_opaque_background, 0

# colour-scale limit used: +/-5.17 pLDDT
save E285K_delta.pse
