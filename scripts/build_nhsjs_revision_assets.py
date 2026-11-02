"""Build three main figures, two main tables and supplementary displays from authenticated saved inputs.

This presentation adds no experiment, changes no recorded population, and leaves
the original manuscript displays and frozen analysis outputs intact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from analysis.revision.response_geometry_theory import polynomial_rank_reversal
from analysis.revision_gpu.v1.figures import panel_population, read_csv as numeric_csv

INPUTS = ROOT / "analysis/manuscript_revision/nhsjs_r1/inputs"
R1 = ROOT / "analysis/revision_gpu/v1/results/r1-2026-10-24/analysis"
BLUE, ORANGE, INK, GREY = "#15558a", "#e99b59", "#222222", "#777777"
COLORS = {"Q": BLUE, "N": ORANGE, "L2": ORANGE}
CASES = [f"{r}:{m}:{d}" for r in ("Q", "L2") for m in ("vision", "lang") for d in ("insertion", "deletion")]
METHODS = ["IG_Q_A1T512", "IG_N_A1T512", "PG_Q_A1T512", "PG_N_A1T512", "input_difference", "IG_Q_A0m64", "IG_N_A0m64"]
METHOD_LABELS = {"IG_Q_A1T512": "IG-Q", "IG_N_A1T512": "IG-N", "PG_Q_A1T512": "Path-gradient Q",
 "PG_N_A1T512": "Path-gradient N", "input_difference": "Input difference", "IG_Q_A0m64": "Equal-weight IG-Q", "IG_N_A0m64": "Equal-weight IG-N",
 "fixed_raster": "Raster", "fixed_reverse_raster": "Reverse raster", "fixed_centre_out": "Center-out",
 "fixed_token_position": "Forward tokens", "fixed_reverse_token_position": "Reverse tokens",
 "fixed_joint_order": "Forward joints", "fixed_reverse_joint_order": "Reverse joints"}
USED: dict[str, str] = {}
POPULATIONS: dict[str, object] = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    USED[path.relative_to(ROOT).as_posix()] = digest(path)
    return path


def rows(path):
    path = source(path)
    data = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text(encoding='utf-8')
    return list(csv.DictReader(io.StringIO(data)))


def style():
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .75, "savefig.facecolor": "white", "svg.hashsalt": "nhsjs-r1-20261031"})


def save(fig, out, name):
    for extension in ("pdf", "png"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {"dpi": 300}
        fig.savefig(out / f"{name}.{extension}", **kwargs)
    plt.close(fig)


def numerics(out):
    residual=numeric_csv(source(R1/'numerics/residuals.csv'))
    raw=numeric_csv(source(R1/'numerics/references.csv')); comparisons=[]
    for r in raw:
        candidate=r.get('candidate')
        if re.fullmatch(r'(fp32|bf16)_vs_A2T\d+',str(r['family'])) and isinstance(candidate,str):
            match=re.fullmatch(r'(A[012])([TEMm])(\d+)',candidate)
            if match: comparisons.append(dict(r,arm=match[1],rule=match[2],budget=float(match[3])))
    fig,axes=plt.subplots(2,2,figsize=(5.85,4.7),layout='constrained'); details={}
    for i,(data,field,title) in enumerate(((residual,'relative_residual','Median completeness gap (%)'),
                                          (comparisons,'relative_l1','Median relative L1 difference'))):
        subset=[r for r in data if r['bank']=='P' and r['modality']=='vision']
        fp32={r['context'] for r in subset if r['arm']=='A2'}
        subset=[r for r in subset if r['context'] in fp32]
        for j,target in enumerate(('Q','N')):
            ax=axes[i,j];points,shared=panel_population(subset,'vision',target,field,16)
            if not shared: raise ValueError(f'Empty matched population {field} {target}')
            details[f'{field}/{target}']={'contexts':shared,'n':len(shared),'series':{},'display_unit':'percent' if i==0 else 'fraction'}
            color=COLORS[target]
            for arm,linestyle in (('A1','--'),('A2','-')):
                for rule,marker in (('T','o'),('E','^'),('M','s')):
                    xs=sorted(b for a,ru,b in points if a==arm and ru==rule)
                    if not xs: continue
                    ys=[float(np.median([points[(arm,rule,b)][c] for c in shared])) for b in xs]
                    details[f'{field}/{target}']['series'][f'{arm}{rule}']={'budget':xs,'median':ys}
                    positive=[(x,y*(100 if i==0 else 1)) for x,y in zip(xs,ys) if y>0]
                    if positive:
                        ax.plot([p[0] for p in positive],[p[1] for p in positive],ls=linestyle,marker=marker,
                                mfc='white' if arm=='A1' else color,mec=INK,ms=4,color=color,lw=1)
            if ('A0',None,64) in points:
                star=float(np.median([points[('A0',None,64)][c] for c in shared]))
                details[f'{field}/{target}']['old_equal_weight_64']=star
                ax.plot([64,71],[star*(100 if i==0 else 1)]*2,color=INK,lw=.75)
                ax.plot(71,star*(100 if i==0 else 1),'*',ms=7,color=INK)
            if i==0: ax.axhline(1,color=GREY,ls=':',lw=.8)
            ax.set_xscale('log',base=2);ax.set_yscale('log')
            ax.set_xticks([16,64,256,512],['16','64','256','512'])
            ax.set_title(f"{chr(65+2*i+j)}  Target {target} (n = {len(shared)})",loc='left',fontweight='bold');ax.grid(alpha=.18)
            if j==0:ax.set_ylabel(title)
            if i==1:ax.set_xlabel('Integration intervals')
    handles=[Line2D([],[],color=GREY,marker=m,label=l,ms=4) for m,l in (('o','Trapezoid'),('^','Equal weights'),('s','Midpoint'))]
    handles += [Line2D([],[],color=GREY,ls='--',marker='o',mfc='white',label='bf16 calculation'),
                Line2D([],[],color=GREY,marker='o',label='fp32 calculation'),
                Line2D([],[],color=INK,ls='none',marker='*',label='Old equal-weight 64')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    details['display_notes']={'star':'64-interval star offset to x=71 with connector for visibility',
        'zero_reference':'zero difference of the fp32 trapezoid with itself omitted on the logarithmic axis',
        'N_comparator':'finite-budget trapezoid has an unresolved endpoint discrepancy; midpoint plateau is a difference from that comparator, not evidence of midpoint error'}
    POPULATIONS['r1_numerics']=details;save(fig,out,'r1_numerics')


from scripts.nhsjs_audit_displays import AuditDisplays


def audit_displays():
    return AuditDisplays(ROOT, INPUTS, R1, source, rows, save, POPULATIONS, METHOD_LABELS)


def tables(out):
    audit_displays().tables(out)


def build(figures, tables_out):
    USED.clear(); POPULATIONS.clear()
    for path in ('analysis/revision/response_geometry_theory.py', 'analysis/revision_gpu/v1/figures.py',
                 'scripts/nhsjs_audit_displays.py'):
        source(ROOT / path)
    manifest=json.loads(source(INPUTS/'manifest.json').read_text(encoding='utf-8'))
    for name,record in manifest['inputs'].items():
        if digest(INPUTS/name)!=record['sha256']: raise ValueError(f'Stale display input {name}')
    figures.mkdir(parents=True,exist_ok=True); tables_out.mkdir(parents=True,exist_ok=True); style()
    audit=audit_displays()
    audit.design(figures); audit.saved_geometry(figures); audit.tail_geometry(figures)
    audit.success_tasks(figures); audit.methods(figures); numerics(figures); audit.offline_numerics(figures); audit.tables(tables_out)
    registry={'schema_version':2,'presentation':'NHSJS audit revision 2026-11-01','inputs_sha256':USED,
              'manuscript_inputs_sha256':{p:digest(ROOT/p) for p in ['paper/paper.tex','paper/supplementary_derivations.tex','paper/references.bib','paper/tables_revision/revision_facts.tex']},
              'builder_sha256':digest(Path(__file__)), 'populations':POPULATIONS,
              'outputs_sha256':{p.relative_to(ROOT).as_posix():digest(p) for folder in (figures,tables_out) for p in sorted(folder.glob('r1_*')) if p.suffix in ('.pdf','.png','.tex')}}
    (figures/'r1_lineage.json').write_text(json.dumps(registry,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(f"Built {len(registry['outputs_sha256'])} displays with {len(USED)} source bindings")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',type=Path,default=ROOT/'paper/figures_revision')
    parser.add_argument('--tables',type=Path,default=ROOT/'paper/tables_revision')
    args=parser.parse_args(); build(args.figures,args.tables)
