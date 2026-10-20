# J08: 10 A interface (heavy-atom contacts) highlighted RED on the crystal J08_7S6K_FGE.pdb
load x_ray_pdbs/J08_7S6K_FGE.pdb, J08_xray
bg_color white
hide everything
show cartoon
color grey80
select iface_rbd, chain E and resi 403+417+421+446+447+449+452+453+454+455+456+457+469+470+471+472+473+474+475+476+477+478+479+480+481+482+483+484+485+486+487+488+489+490+491+492+493+494+495+496+497+498+501+505
select iface_ab, (chain F and resi 29+33+35+47+49+50+51+52+53+54+55+56+57+58+59+60+61+64+73+95+97+98+99+100) or (chain G and resi 1+2+27+28+29+30+31+32+33+34+50+68+71+89+90+91+92+93)
color red, iface_rbd or iface_ab
show sticks, (iface_rbd or iface_ab) and not name N+C+O
color slate, chain E and not iface_rbd
set cartoon_transparency, 0.15
orient
set ray_shadows,0
set cartoon_fancy_helices,1
ray 1500,1100
png interface_J08.png, dpi=150
