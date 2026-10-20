#!/usr/bin/env python
"""10 most-diverse-ΔlogKD points. Correct protocol: chids chains, kalign-RBD +
ANARCII-VH/VL crop, WT canonicalized to Wuhan, MT = strain (asserts on both).
Escape residues = strain-vs-WUHAN (real mutations, not crop construct reverts).
Saves the FULL PAE matrix + indices per structure, and ALL iPAE variants:
  ipae_allpairs  = <PAE> over all RBD<->Ab pairs
  ipae_mutres    = <PAE> over (escape residues)<->Ab pairs
  ipae_permut    = sum_i <PAE(res_i<->Ab)> / N_mut over escape residues
ΔiPAE = MT - WT for each. Compared to paper at the end."""
import csv, json, os, sys, ast, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from biopandas.pdb import PandasPdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_full as R, native_crop as N, repro_asserts as A
from aftoolkit.structure import load_protein, mutate_protein, AA_3_TO_1_DICT
from aftoolkit.mutations import parse_mutations
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); GAP = 25
PAEDIR = os.path.join(HERE, "diverse10_pae"); os.makedirs(PAEDIR, exist_ok=True)
OUT = os.path.join(HERE, "diverse10.csv")
# all (antibody, strain) points built from the affinity table in main()


def main():
    strains = json.load(open(os.path.join(ROOT, "data/rbdaim/datasets/rbd_strains.json")))
    ref = strains["Wuhan"]; refn = list(range(334, 334 + len(ref)))
    aff = pd.read_csv(os.path.join(ROOT, "data/rbdaim/fig2g_affinity.csv"))
    pp = pd.read_excel(os.path.expanduser("~/Downloads/41467_2026_70553_MOESM9_ESM.xlsx"),
                       sheet_name="Fig 2g", header=1).dropna(how="all")
    pp.columns = ["dlogkd", "dipae"]; pp = pp.apply(pd.to_numeric, errors="coerce").dropna()
    paper = lambda dk: (lambda n: round(float(pp.dipae[n.idxmin()]),3) if n.min()<0.03 else None)((pp.dlogkd-abs(dk)).abs())
    from aftoolkit.folding import OpenFoldWrapper
    w = OpenFoldWrapper(device="mps", inference_n_recycle=2, always_use_template=False, return_all_cycles=True); w.init_model()
    CYC=(0,1,2)
    fields=(["antibody","Xray","strain","dlogKD","n_mut","esc_res"]
            +[f"d_allpairs_c{c}" for c in CYC]+[f"d_mutres_c{c}" for c in CYC]
            +[f"d_permut_c{c}" for c in CYC]+["diPAE_paper"])
    done=set(pd.read_csv(OUT)[["antibody","strain"]].itertuples(index=False,name=None)) if os.path.exists(OUT) and os.path.getsize(OUT)>0 else set()
    new=not (os.path.exists(OUT) and os.path.getsize(OUT)>0)
    fh=open(OUT,"a",newline=""); wri=csv.DictWriter(fh,fieldnames=fields)
    if new: wri.writeheader(); fh.flush()
    allpts=[(r.ab,r.strain) for _,r in aff[aff.delta_log_kd.abs()>1e-6].iterrows()]
    cropcache={}

    def crop(ab):
        if ab in cropcache: return cropcache[ab]
        row=aff[aff.ab==ab].iloc[0]; H,L,RBD=ast.literal_eval(row.chids)
        df=PandasPdb().read_pdb(R.fetch_pdb(row.pdbid)).df["ATOM"]; df=df[df.alt_loc.isin(["","A"])]
        kr=N.crop_rbd_resns(*N.chain_seq(df,RBD),ref); kh=N.vhvl_resns(*N.chain_seq(df,H)); kl=N.vhvl_resns(*N.chain_seq(df,L))
        parts=[df[(df.chain_id==RBD)&(df.residue_number.isin(kr))].assign(chain_id="1"),
               df[(df.chain_id==H)&(df.residue_number.isin(kh))],df[(df.chain_id==L)&(df.residue_number.isin(kl))]]
        out=pd.concat(parts).reset_index(drop=True); out["atom_number"]=range(1,len(out)+1); out["line_idx"]=range(len(out))
        fp=os.path.join(PAEDIR,f"{ab}_crop.pdb"); p=PandasPdb(); p.df["ATOM"]=out; p.to_pdb(fp)
        ca=out[(out.chain_id=="1")&(out.atom_name=="CA")].sort_values("residue_number")
        cs="".join(AA_3_TO_1_DICT.get(x,"X") for x in ca.residue_name); cr=list(ca.residue_number)
        A.assert_rbd_region(cs,ref,ab); A.assert_unique_residues(cr,ab); A.assert_canonical_numbering(dict(zip(cr,cs)),ab)
        cropcache[ab]=(fp,H,L,cs,cr,row.pdbid); return cropcache[ab]

    def infer(ab,fp,H,L,cs,cr,tgt,tag):
        d=load_protein(fp,chains=["1",H,L],chain_gap=GAP)
        ms=",".join(f"1:{a}{pp_}{m}" for a,pp_,m in R.rbd_mutations(cs,cr,tgt))
        if ms: d=mutate_protein(d,parse_mutations(ms))
        bca=d[(d.chain_id_original=="1")&(d.atom_name=="CA")].sort_values("residue_number")
        built="".join(AA_3_TO_1_DICT.get(x,"X") for x in bca.residue_name)
        A.assert_built_matches_target(built,tgt,f"{ab}/{tag}")
        w.evaluate(d); c=w.get_of_protein(); c=c[c.atom_name=="CA"].reset_index(drop=True)
        lab=c.chain_id_original.values; resn=c.residue_number_original.values
        cyc={ci: np.asarray(w.of_output_per_cycle[ci]["predicted_aligned_error"]) for ci in (0,1,2)}
        np.savez_compressed(os.path.join(PAEDIR,f"{ab}_{tag}.npz"),labels=lab,resnums=resn,
                            **{f"pae_c{ci}":cyc[ci] for ci in (0,1,2)})   # FULL per-cycle PAE saved
        return cyc,lab,resn,built

    for ab,strain in allpts:
        if (ab,strain) in done: continue
        try:
            key=R.STRAIN_MAP.get(strain)
            if key is None or key not in strains:
                print(f"{ab}/{strain} SKIP no-seq",flush=True); continue
            tgt=strains[key]
            fp,H,L,cs,cr,pdbid=crop(ab)
            escres=[p for _,p,_ in R.rbd_mutations(ref,refn,tgt)]      # TRUE escape residues (strain vs Wuhan)
            dk=float(aff[(aff.ab==ab)&(aff.strain==strain)].delta_log_kd.iloc[0])
            cycw,lw,rw,wtseq=infer(ab,fp,H,L,cs,cr,ref,"WT"); cycm,lm,rm,mtseq=infer(ab,fp,H,L,cs,cr,tgt,"MT")
            A.assert_wt_mt_strains(wtseq,mtseq,ref,tgt,f"{ab}/{strain}")   # WT==Wuhan, MT==strain, diff==escape muts
            def allp(pae,lab): ab_=np.where(lab!="1")[0]; rb=np.where(lab=="1")[0]; return float(pae[np.ix_(rb,ab_)].mean())
            def mres(pae,lab,resn): ab_=np.where(lab!="1")[0]; mi=np.where((lab=="1")&np.isin(resn,escres))[0]; return float(np.concatenate([pae[np.ix_(mi,ab_)].ravel(),pae[np.ix_(ab_,mi)].ravel()]).mean()) if len(mi) else float("nan")
            def permut(pw,pm):
                ab_=np.where(lw!="1")[0]; tot=0; nn=0
                for rr in escres:
                    iw=np.where((lw=="1")&(rw==rr))[0]; im=np.where((lm=="1")&(rm==rr))[0]
                    if len(iw) and len(im):
                        vw=np.concatenate([pw[np.ix_(iw,ab_)].ravel(),pw[np.ix_(ab_,iw)].ravel()]).mean()
                        vm=np.concatenate([pm[np.ix_(im,ab_)].ravel(),pm[np.ix_(ab_,im)].ravel()]).mean()
                        tot+=(vm-vw); nn+=1
                return tot/nn if nn else float("nan")
            row={"antibody":ab,"Xray":pdbid,"strain":strain,"dlogKD":round(dk,3),"n_mut":len(escres),
                 "esc_res":"+".join(map(str,escres[:6]))+("..." if len(escres)>6 else ""),"diPAE_paper":paper(dk)}
            for c in CYC:
                pw,pm=cycw[c],cycm[c]
                row[f"d_allpairs_c{c}"]=round(-(allp(pm,lm)-allp(pw,lw)),3)
                row[f"d_mutres_c{c}"]=round(-(mres(pm,lm,rm)-mres(pw,lw,rw)),3)
                row[f"d_permut_c{c}"]=round(-permut(pw,pm),3)
            wri.writerow(row); fh.flush(); os.fsync(fh.fileno())
            print(f"{ab:<13}{pdbid:>6} {strain.split()[0]:<9} dlogKD={dk:+.2f} N={len(escres):>2} | "
                  f"d_all(c0/1/2)={row['d_allpairs_c0']:+.2f}/{row['d_allpairs_c1']:+.2f}/{row['d_allpairs_c2']:+.2f} | "
                  f"d_mutres={row['d_mutres_c0']:+.2f}/{row['d_mutres_c1']:+.2f}/{row['d_mutres_c2']:+.2f} | paper={row['diPAE_paper']}",flush=True)
        except Exception as e:
            print(f"{ab}/{strain} FAILED: {e}",flush=True)
    fh.close()
    o=pd.read_csv(OUT)
    from scipy.stats import pearsonr,spearmanr
    print(f"\n=== how well each iPAE variant/cycle tracks (n={len(o)}) ===")
    for col in [f"{v}_c{c}" for v in ["d_allpairs","d_mutres","d_permut"] for c in CYC]:
        x=o.dlogKD; y=o[col]; m=o.dropna(subset=["diPAE_paper"])
        pap=f"vsPaper P={pearsonr(m.diPAE_paper,m[col])[0]:+.2f}" if len(m)>=3 else ""
        print(f"  {col:<14} vs dlogKD: P={pearsonr(x,y)[0]:+.3f} S={spearmanr(x,y)[0]:+.3f}   {pap}")


if __name__ == "__main__":
    main()
