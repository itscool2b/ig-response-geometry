"""Build five R1 manuscript figures and four tables from authenticated saved inputs.

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
BLUE, ORANGE, INK, GREY = "#2a78d6", "#d65326", "#222222", "#777777"
COLORS = {"Q": BLUE, "N": ORANGE, "L2": ORANGE}
CASES = [f"{r}:{m}:{d}" for r in ("Q", "L2") for m in ("vision", "lang") for d in ("insertion", "deletion")]
METHODS = ["IG_Q_A1T512", "IG_N_A1T512", "PG_Q_A1T512", "PG_N_A1T512", "input_difference", "IG_Q_A0m64", "IG_N_A0m64"]
METHOD_LABELS = {"IG_Q_A1T512": "IG Q T512", "IG_N_A1T512": "IG N T512", "PG_Q_A1T512": "PG Q T512",
 "PG_N_A1T512": "PG N T512", "input_difference": "Input difference", "IG_Q_A0m64": "IG Q E64", "IG_N_A0m64": "IG N E64",
 "fixed_raster": "Raster", "fixed_reverse_raster": "Reverse raster", "fixed_centre_out": "Centre-out"}
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
    data = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    return list(csv.DictReader(io.StringIO(data)))


def style():
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "savefig.facecolor": "white", "svg.hashsalt": "nhsjs-r1-20261031"})


def save(fig, out, name):
    for extension in ("pdf", "png"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {"dpi": 300}
        fig.savefig(out / f"{name}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def design(out):
    fig = plt.figure(figsize=(6.5, 3.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(.65, 2))
    ax = fig.add_subplot(grid[0, :]); ax.axis("off")
    labels = ["Policy context", "Target Q or N\nIG ranking", "Insertion or\ndeletion", "Response Q or N\nNormalized AUC"]
    for i, label in enumerate(labels):
        x = .11 + i * .255
        ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.4", fc="#f4f6f8", ec="#a8b2ba", lw=.7), fontsize=8)
        if i < 3:
            ax.annotate("", xy=(x+.17,.5),xytext=(x+.09,.5),xycoords="axes fraction", arrowprops=dict(arrowstyle="->",lw=.8))
    ax.set_title("A) Attribution target and scoring response are separate choices", loc="left")
    ax = fig.add_subplot(grid[1, 0]); u = np.linspace(0, 4, 401)
    ax.plot(u, 1-u, label="Q", color=BLUE); ax.plot(u, 1-np.sqrt(u), label="N (zero-stabilizer limit)", color=ORANGE, ls="--")
    ax.axvspan(0, 1, color="#eff5ec", zorder=0); ax.axvline(1, color=GREY, ls=":", lw=.8); ax.axhline(0,color=GREY,lw=.6)
    ax.set(xlabel="Squared residual / baseline residual", ylabel="Normalized response", title="B) In range and overshoot", xlim=(0,4)); ax.legend(frameon=False)
    ax = fig.add_subplot(grid[1, 1]); result = polynomial_rank_reversal()
    from fractions import Fraction
    q = np.array([float(Fraction(v)) for v in result["quadratic_ig_exact"]]); n = np.array(result["stabilized_l2_ig"])
    for i,(values,color,label) in enumerate(((q,BLUE,"Target Q"),(n,ORANGE,"Target N"))):
        ax.bar(np.arange(2)+(i-.5)*.32, values/values.sum(), .32, color=color, label=label)
    ax.set(xticks=[0,1],xticklabels=["Coordinate 1", "Coordinate 2"],ylabel="Share of target endpoint gap",title="C) Constructed ranking reversal",ylim=(0,.7));ax.legend(frameon=False)
    POPULATIONS["r1_design"] = {"kind": "analytic illustration, not an observed policy reversal", "construction": result}
    save(fig,out,"r1_design")


def saved_geometry(out):
    data = rows(ROOT / "analysis/paired_rescoring/residual_geometry/results/2026-10-23-v1/calls.csv.gz")
    data = [r for r in data if r["figure"] == "headline" and r["paired_status"] == "defined"]
    fig, axes = plt.subplots(2,2,figsize=(6.5,4.4),layout="constrained")
    rng = np.random.default_rng(0); counts = {}
    labels = [f"{c.split(':')[0].replace('L2','N')} {'V' if ':vision:' in c else 'L'} {'ins' if c.endswith('insertion') else 'del'}" for c in CASES]
    for y,case in enumerate(CASES):
        group = [r for r in data if r["case"]==case]; vals=np.array([float(r["delta"]) for r in group]); jitter=rng.uniform(-.18,.18,len(vals))
        color=COLORS[case.split(':')[0]]; outside=int(np.sum((vals<-.05)|(vals>.25)))
        counts[case]={"calls":len(vals),"outside_zoom":outside,"log_region":int(np.sum(abs(vals)>.25))}
        for ax in axes[0]:
            ax.scatter(vals,y+jitter,s=3,color=color,alpha=.25,linewidths=0,rasterized=True)
            ax.plot(np.median(vals),y,"D",ms=3,color=INK)
        axes[0,1].text(.246,y,str(outside),ha="right",va="center",fontsize=6,bbox=dict(fc="white",ec="none",pad=.15))
    for ax in axes[0]:
        ax.set_yticks(range(8),labels);ax.set_ylim(7.5,-.5);ax.axvline(0,color=GREY,lw=.6);ax.set_xlabel(r"Paired change $\Delta=\mathrm{AUC}_Q-\mathrm{AUC}_N$")
    axes[0,0].set_xscale("symlog",linthresh=.25);axes[0,0].set_xlim(-6500,.26)
    axes[0,0].set_xticks([-1000,-10,-1,0,.2],['-1000','-10','-1','0','0.2'])
    axes[0,0].set_title("A) Every defined saved call",loc="left")
    axes[0,1].set_xlim(-.05,.25);axes[0,1].set_title("B) Central zoom; counts outside",loc="left")
    for case in CASES:
        group=[r for r in data if r["case"]==case]; color=COLORS[case.split(':')[0]]
        for first,marker,size in ((False,"o",3),(True,"^",9)):
            selected=[r for r in group if (r["first_call"]=="True")==first]
            y=[float(r["delta"]) for r in selected]
            axes[1,0].scatter([float(r["residual_baseline"]) for r in selected],y,s=size,marker=marker,color=color,alpha=.25,linewidths=0,rasterized=True)
            axes[1,1].scatter([float(r["ratio_max_interior_to_baseline"]) for r in selected],y,s=size,marker=marker,color=color,alpha=.25,linewidths=0,rasterized=True)
    for ax in axes[1]:
        ax.set_xscale("log");ax.set_yscale("symlog",linthresh=.25);ax.set_ylim(-6500,.26)
        ax.set_yticks([-1000,-100,-10,-1,0],['-1000','-100','-10','-1','0'])
        ax.axhline(0,color=GREY,lw=.6);ax.set_ylabel(r"Paired change $\Delta$")
    axes[1,0].set(xlabel="All-baseline squared residual B",title="C) Baseline size")
    axes[1,1].set(xlabel="Largest interior residual / B",title="D) Intermediate excursion")
    handles=[Line2D([],[],marker='o',ls='none',color=BLUE,label='Q-ranked'),Line2D([],[],marker='o',ls='none',color=ORANGE,label='N-ranked'),Line2D([],[],marker='^',ls='none',color=INK,label='First call')]
    axes[1,0].legend(handles=handles,loc="lower right",frameon=False,fontsize=6)
    POPULATIONS["r1_saved_geometry"] = counts;save(fig,out,"r1_saved_geometry")


def success_tasks(out):
    old=rows(ROOT/"analysis/paired_rescoring/oneb_success/results/2026-10-23-v1/summary.csv")
    extra=rows(ROOT/"analysis/paired_rescoring/extra_tasks/results/2026-10-23-v1/summary.csv")
    new=rows(R1/"shifts/B1/summary_vision.csv")
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.5),layout="constrained",sharey=True)
    plotted=[]
    for ax,direction in zip(axes,("insertion","deletion")):
        group=[]
        for outcome in ("success","failure"):
            match=[r for r in old if r['population']=='pooled_primary' and r['modality']=='vision' and r['direction']==direction and r['outcome_group']==outcome]
            if len(match)!=1:raise ValueError(f"Ambiguous historical success group {outcome}: {len(match)}")
            r=match[0];group.append((f"Saved 1B {outcome}\n{r['n_defined_episodes']} episodes / {r['n_defined_calls']} calls",float(r['paired_median']),[float(r['paired_median_ci_lo']),float(r['paired_median_ci_hi'])],BLUE))
        for outcome,label,episodes in (("True","success",31),("False","failure",9)):
            r=next(r for r in new if r['ranking']=='IG_Q_A1T512' and r['direction']==direction and r['success']==outcome)
            group.append((f"R1 B1 {label}\n{episodes} episodes / {r['n']} calls",float(r['median']),None,ORANGE))
        for population,label in (("PegInsertionSide_pooled","PegInsertionSide"),("PickSingleYCB_pooled","PickSingleYCB")):
            r=next(r for r in extra if r['population']==population and r['modality']=='vision' and r['direction']==direction and r['outcome_group']=='all')
            group.append((f"{label}\n{r['n_defined_episodes']} episodes / {r['n_defined_calls']} calls",float(r['paired_median']),[float(r['paired_median_ci_lo']),float(r['paired_median_ci_hi'])],GREY))
        for y,(label,value,interval,color) in enumerate(group):
            if interval:ax.plot(interval,[y,y],color=color,lw=1)
            ax.plot(value,y,'D' if interval is None else 'o',color=color,ms=4)
        ax.set_yticks(range(len(group)),[r[0] for r in group]);ax.set_ylim(len(group)-.5,-.5);ax.set_title(('A) ' if direction=='insertion' else 'B) ')+direction.capitalize(),loc='left');ax.set_xlabel(r"Median $\mathrm{AUC}_Q-\mathrm{AUC}_N$");ax.axvline(0,color=GREY,lw=.6);ax.grid(axis='x',alpha=.2)
        plotted.append({"direction":direction,"groups":group})
    POPULATIONS['r1_success_tasks']={"new_ranking":"IG_Q_A1T512","groups":plotted,"intervals":"saved groups only: recorded episode-bootstrap median intervals; R1 diamonds have no interval"}
    save(fig,out,'r1_success_tasks')


def methods(out):
    data=rows(INPUTS/'methods_advantage.csv');fig,axes=plt.subplots(3,2,figsize=(6.5,5.1),layout='constrained',sharey=True)
    names=METHODS+['fixed_raster','fixed_reverse_raster','fixed_centre_out']
    for i,bank in enumerate(('P','B1','S')):
        for j,response in enumerate(('Q','N')):
            ax=axes[i,j]; selected=[r for r in data if r['bank']==bank and r['modality']=='vision' and r['direction']=='deletion' and r['response']==response]
            keyed={r['ranking']:r for r in selected}
            if set(names)!=set(keyed):raise ValueError(f'Unexpected ranking set {bank} {response}: {set(keyed)}')
            for y,name in enumerate(names):
                r=keyed[name];color=BLUE if '_Q_' in name else ORANGE if '_N_' in name else GREY
                ax.plot([float(r['interval95_lo']),float(r['interval95_hi'])],[y,y],color=color,lw=1)
                ax.plot(float(r['mean']),y,'o',color=color,ms=3)
            ax.set_yticks(range(len(names)),[METHOD_LABELS[n] for n in names]);ax.set_ylim(9.5,-.5);ax.axvline(0,color=GREY,lw=.6)
            ax.set_title(f"{chr(65+2*i+j)}) {bank}: {response} deletion ({selected[0]['episodes']} episodes)",loc='left');ax.grid(axis='x',alpha=.2)
            if i==2:ax.set_xlabel('Calibrated advantage over random')
    POPULATIONS['r1_methods']={'banks':{'P':60,'B1':40,'S':30},'intervals':'descriptive 95% episode bootstrap; seven adjusted contrasts in Table 4','ranks':names}
    save(fig,out,'r1_methods')


def numerics(out):
    residual=numeric_csv(source(R1/'numerics/residuals.csv')); raw=numeric_csv(source(R1/'numerics/references.csv')); comparisons=[]
    for r in raw:
        candidate=r.get('candidate')
        if re.fullmatch(r'(fp32|bf16)_vs_A2T\d+',str(r['family'])) and isinstance(candidate,str):
            match=re.fullmatch(r'(A[012])([TEMm])(\d+)',candidate)
            if match:comparisons.append(dict(r,arm=match[1],rule=match[2],budget=float(match[3])))
    fig,axes=plt.subplots(2,2,figsize=(6.5,3.8),layout='constrained'); details={}
    for i,(data,field,title) in enumerate(((residual,'relative_residual','Completeness residual'),(comparisons,'relative_l1','Coordinate error to fp32 T512'))):
        subset=[r for r in data if r['bank']=='P' and r['modality']=='vision'];fp32={r['context'] for r in subset if r['arm']=='A2'};subset=[r for r in subset if r['context'] in fp32]
        for j,target in enumerate(('Q','N')):
            ax=axes[i,j];points,shared=panel_population(subset,'vision',target,field,16)
            if not shared:raise ValueError(f'Empty matched population {field} {target}')
            details[f'{field}/{target}']={'contexts':shared,'n':len(shared),'series':{}}
            for arm,linestyle in (('A1','--'),('A2','-')):
                for rule,color in (('T',BLUE),('E',ORANGE),('M','#177e60')):
                    xs=sorted(b for a,ru,b in points if a==arm and ru==rule)
                    if not xs:continue
                    ys=[float(np.median([points[(arm,rule,b)][c] for c in shared])) for b in xs]
                    details[f'{field}/{target}']['series'][f'{arm}{rule}']={'budget':xs,'median':ys}
                    ax.plot(xs,[max(y,1e-12) for y in ys],ls=linestyle if len(xs)>=3 else 'none',marker='o',mfc='white' if arm=='A1' else color,ms=3,color=color,lw=1)
            if ('A0',None,64) in points:ax.plot(64,np.median([points[('A0',None,64)][c] for c in shared]),'*',ms=7,color=INK)
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(f"{chr(65+2*i+j)}) {target}, {len(shared)} matched contexts",loc='left');ax.grid(alpha=.2)
            if j==0:ax.set_ylabel(title)
            if i==1:ax.set_xlabel('Integration intervals')
    handles=[Line2D([],[],color=c,label=l) for c,l in ((BLUE,'Trapezoid'),(ORANGE,'Endpoint-inclusive'),('#177e60','Midpoint'))]
    handles += [Line2D([],[],color=GREY,ls='--',label='bf16 forward'),Line2D([],[],color=GREY,label='fp32 forward'),Line2D([],[],color=INK,ls='none',marker='*',label='Historical E64')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    POPULATIONS['r1_numerics']=details;save(fig,out,'r1_numerics')


def tex(value):
    return str(value).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')


def table(out,name,caption,label,headers,data,widths=None):
    columns=' '.join(f'p{{{w:.2f}\\linewidth}}' for w in widths) if widths else 'l'+'r'*(len(headers)-1)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+columns+'}',r'\toprule', ' & '.join(r'\textbf{'+tex(h)+'}' for h in headers)+r' \\',r'\midrule']
    lines += [' & '.join(tex(v) for v in row)+r' \\' for row in data]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def tables(out):
    experiments=json.loads(source(INPUTS/'experiments.json').read_text())
    table(out,'r1_experiments',r'\textbf{Experimental and numerical configurations.} E is the historical endpoint-inclusive rule; T is trapezoidal; * marks an inferred budget. Precision gives forward/accumulation where both are specified. A0--A4 are defined in Targets, features and numerical estimators. Peg and YCB denote PegInsertionSide and PickSingleYCB. The offline sample has no completed 1B context. Saved curves do not preserve the original policy inputs.', 'tab:r1-experiments', ['Experiment','Estimator / budget','Precision','Context source','Preserved inputs'],experiments,[.19,.17,.17,.18,.18])
    names=['Saved Q (170M)','Saved N (170M)','Saved 1B M4','Saved 1B M6','Saved 1B M5','Saved m128 M4','Saved m128 M7','Offline M8','R1 P','R1 B1','R1 S','R1 M8']
    identity=rows(INPUTS/'identity_status.csv'); counts=[[name,*[r[k] for k in ('recorded','recovered','inferred','unknown')]] for name,r in zip(names,identity)]
    values=json.loads(source(INPUTS/'identity_values.json').read_text())
    actual=[[r['component'],r['value'],r['status'].capitalize()] for r in values['rows']]
    totals={k:sum(int(r[k]) for r in identity) for k in ('recorded','recovered','inferred','unknown')}
    caption=(r'\textbf{Pinned R1 identities and their evidence status.} Revisions and digests use 12-character prefixes; the archive gives full strings, adaptor hashes and per-phase evidence. Recorded means captured by the experiment. Inferred software/preprocessing entries reflect incomplete host captures, not a claim that every host was individually captured. Across all 12 historical and new populations the inventory has '
             +f"{totals['recorded']} recorded, {totals['recovered']} recovered, {totals['inferred']} inferred and {totals['unknown']} unknown fields. Historical unknowns remain unknown.")
    table(out,'r1_identities',caption,'tab:r1-identities',['Component','Value','Status'],actual,[.19,.57,.13])
    paired=rows(ROOT/'analysis/manuscript_revision/results/2026-10-03-v2/paired_summary.csv')
    data=[]
    for r in paired:
        label=r['case'].replace('L2','N').replace(':vision:', ' V ').replace(':lang:', ' L ').replace('insertion','ins').replace('deletion','del')
        data.append([label,f"{float(r['median']):.4f}",f"{float(r['positive_percent']):.1f}",f"{float(r['overshoot_percent']):.1f}",f"{float(r['mean']):.3g} [{float(r['mean_ci_low']):.3g}, {float(r['mean_ci_high']):.3g}]",f"{float(r['minimum']):.3g}"])
    table(out,'r1_saved_summary',r'\textbf{Saved response-only changes retain positive medians and negative tails.} Each row contains 750 calls from 30 episodes. Median and positive share weight calls equally; means weight episodes equally. Brackets are descriptive 95\% whole-episode bootstrap intervals, conditional on these records, not training runs, checkpoints or tasks. V denotes vision, L language; ins insertion, del deletion. The unmatched Q- and N-ranked cohorts do not compare ranking quality.', 'tab:r1-saved-summary',['Case','Median','Positive (%)','Overshoot (%)','Mean [interval]','Minimum'],data,[.12,.10,.11,.13,.30,.11])
    decisions=rows(INPUTS/'methods_decisions.csv'); data=[]
    for bank in ('P','B1','S'):
        for response,direction in (('Q','deletion'),('Q','insertion'),('N','deletion'),('N','insertion')):
            r=next(r for r in decisions if r['bank']==bank and r['modality']=='vision' and r['response']==response and r['direction']==direction)
            preferred=METHOD_LABELS.get(r['preferred'],r['preferred']) if r['preferred'] not in ('','null') else 'Tie'
            data.append([bank,f'{response} {direction}',preferred,'Yes' if r['unique']=='True' else 'No','Yes' if r['all_fixed_below']=='True' else 'No'])
    contrasts=[r for r in rows(INPUTS/'contrasts.csv') if r['bank']=='P' and r['metric']=='normalized_auc']
    if len(contrasts)!=7:raise ValueError('The primary family must contain exactly seven contrasts')
    table(out,'r1_methods_summary',r'\textbf{Preferred methods and primary contrasts.} Preference maximizes calibrated advantage on complete-case episodes (P: 60; B1: 40; S: 30). Clear means the adjusted best-minus-second interval excludes zero. Above fixed orders compares point means, not significance. E64 denotes the historical A0 estimator. The following primary contrasts use P vision deletion and Bonferroni-adjusted percentile intervals (seven comparisons).', 'tab:r1-methods-summary',['Bank','Response / direction','Preferred','Clear','Above fixed means'],data,[.06,.20,.29,.09,.22])
    target=out/'r1_methods_summary.tex'; text=target.read_text();text=text.replace(r'\end{table}', '')
    text += '\n'+r'\par\smallskip'+'\n'+r'\begin{tabular}{p{.40\linewidth}rrr}'+'\n'+r'\toprule Contrast & Response & Estimate & Adjusted interval \\ \midrule'+'\n'
    for r in contrasts:
        method=METHOD_LABELS.get(r['method'],r['method']); control=METHOD_LABELS.get(r['control'],r['control'])
        if r['contrast']=='method_minus_random':label=method+' minus random'
        elif r['contrast']=='method_minus_method':label=method+' minus '+control
        else:label='IG Q at Q minus IG N at N'
        response=r['response'] or 'matched'
        text += f"{tex(label)} & {tex(response)} & {float(r['estimate']):.3g} & [{float(r['percentile_lo']):.3g}, {float(r['percentile_hi']):.3g}]"+r' \\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}'+'\n';target.write_text(text,encoding='utf-8',newline='\n')


def build(figures,tables_out):
    USED.clear();POPULATIONS.clear()
    # Imported calculations and population selection are numerical inputs too.
    source(ROOT/'analysis/revision/response_geometry_theory.py')
    source(ROOT/'analysis/revision_gpu/v1/figures.py')
    manifest=json.loads(source(INPUTS/'manifest.json').read_text())
    for name,record in manifest['inputs'].items():
        if digest(INPUTS/name)!=record['sha256']:raise ValueError(f'Stale display input {name}')
    figures.mkdir(parents=True,exist_ok=True);tables_out.mkdir(parents=True,exist_ok=True);style()
    for function in (design,saved_geometry,success_tasks,methods,numerics):function(figures)
    tables(tables_out)
    registry={'schema_version':1,'presentation':'NHSJS R1 2026-10-31','inputs_sha256':USED,
              'manuscript_inputs_sha256':{p:digest(ROOT/p) for p in ['paper/paper.tex','paper/references.bib','paper/appendix_aliasing.tex','paper/tables_revision/revision_facts.tex']},
              'builder_sha256':digest(Path(__file__)),'populations':POPULATIONS,
              'outputs_sha256':{p.relative_to(ROOT).as_posix():digest(p) for folder in (figures,tables_out) for p in sorted(folder.glob('r1_*')) if p.suffix in ('.pdf','.png','.tex')}}
    (figures/'r1_lineage.json').write_text(json.dumps(registry,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(f"Built {len(registry['outputs_sha256'])} displays with {len(USED)} source bindings")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',type=Path,default=ROOT/'paper/figures_revision')
    parser.add_argument('--tables',type=Path,default=ROOT/'paper/tables_revision')
    args=parser.parse_args();build(args.figures,args.tables)
