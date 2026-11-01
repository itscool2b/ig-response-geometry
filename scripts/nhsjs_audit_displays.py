"""Reader-facing audit displays, generated only from authenticated records."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json
import math
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analysis.revision.response_geometry_theory import polynomial_rank_reversal

BLUE, ORANGE, INK, GREY = "#15558a", "#e99b59", "#222222", "#777777"
CASES = [f"{r}:{m}:{d}" for r in ("Q", "L2") for m in ("vision", "lang") for d in ("insertion", "deletion")]
BANKS = {"P": "PickCube 170M", "B1": "PickCube 1B", "S": "StackCube 170M"}
AUDIT = "analysis/manuscript_revision/nhsjs_audit_2026_11_01"


def text(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def number(value, digits=3):
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace('-', '−')
    places=max(0,digits-1-int(math.floor(math.log10(abs(value)))))
    return f"{value:.{places}f}".replace('-', '−')


def interval(lo, hi):
    return f"[{number(lo)}, {number(hi)}]"


def table(out, name, caption, label, headers, data, widths, long=False):
    columns = ' '.join(f'p{{{w:.3f}\\linewidth}}' for w in widths)
    header = ' & '.join(r'\textbf{' + text(h) + '}' for h in headers) + r' \\'
    if long:
        lines = [r'\setlength{\tabcolsep}{2pt}', r'\begin{longtable}{' + columns + '}',
                 r'\caption{' + caption + r'}\label{' + label + r'}\\', r'\toprule', header,
                 r'\midrule\endfirsthead', r'\toprule', header, r'\midrule\endhead',
                 r'\bottomrule\endfoot']
    else:
        lines = [r'\begin{table}[htbp]', r'\centering', r'\caption{' + caption + '}',
                 r'\label{' + label + '}', r'\setlength{\tabcolsep}{2pt}',
                 r'\begin{tabular}{' + columns + '}', r'\toprule', header, r'\midrule']
    lines += [' & '.join(text(v) for v in row) + r' \\' for row in data]
    lines += [r'\end{longtable}'] if long else [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


class AuditDisplays:
    def __init__(self, root, inputs, r1, source, rows, save, populations, method_labels):
        self.root, self.inputs, self.r1 = root, inputs, r1
        self.source, self.rows, self.save, self.populations = source, rows, save, populations
        self.labels = method_labels
        path = root / AUDIT / 'results/2026-11-01-v1/report.json'
        self.report = json.loads(source(path).read_text(encoding='utf-8'))
        for name, expected in self.report['inputs_sha256'].items():
            import hashlib
            if hashlib.sha256(source(root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Stale retrospective reporting input: {name}')

    def read(self, path):
        return self.rows(self.root / path)

    def design(self, out):
        fig, axes = plt.subplots(2, 2, figsize=(5.85, 4.7), layout='constrained')
        ax = axes[0, 0]; ax.axis('off')
        labels = ['Fix the policy and its situation', 'Choose target Q or N; rank inputs',
                  'Remove or restore ranked inputs', 'Choose response Q or N; score AUC']
        for i, label in enumerate(labels):
            y = .82 - i * .23
            ax.text(.5, y, label, ha='center', va='center', transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round,pad=.35', fc='#f4f6f8', ec='#a8b2ba', lw=.75))
            if i < 3:
                ax.annotate('', xy=(.5, y-.16), xytext=(.5, y-.07), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=.8))
        ax.set_title('A  Two separate choices', loc='left', fontweight='bold')
        data = self.read('analysis/paired_rescoring/results/2026-10-01-v1/calls.csv.gz')
        candidates = [r for r in data if r['case'] == 'Q:vision:deletion' and r['paired_status'] == 'defined'
                      and -1.5 < min(json.loads(r['normalized_q'])) < -.1]
        selected = min(candidates, key=lambda r: r['record_id'])
        x = np.array(json.loads(selected['grid_percent']), dtype=float) / 100
        q, n = (np.array(json.loads(selected[k])) for k in ('normalized_q', 'normalized_l2'))
        ax = axes[0, 1]
        for y, color, ls, label in ((q, BLUE, '-', 'Scoring Q'), (n, ORANGE, '--', 'Scoring N')):
            ax.plot(x, y, color=color, ls=ls, marker='o', ms=3.5, lw=1, label=label)
            ax.fill_between(x, 0, y, color=color, alpha=.12)
        k = int(np.argmin(q)); ax.annotate('Overshoot', xy=(x[k], q[k]), xytext=(.48, .28),
                                          textcoords='axes fraction', fontsize=8, arrowprops=dict(arrowstyle='->', lw=.8))
        ax.axhline(0, color=GREY, lw=.75)
        ax.set(xlabel='Fraction of image patches removed', ylabel='Normalized response', xlim=(0, 1))
        ax.set_title('B  One saved deletion curve', loc='left', fontweight='bold')
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        ax = axes[1, 0]; u = np.linspace(0, 4, 401)
        ax.plot(u, 1-u, color=BLUE, label='Scoring Q')
        ax.plot(u, 1-np.sqrt(u), color=ORANGE, ls='--', label='Scoring N')
        ax.fill_between(u, 1-u, 1-np.sqrt(u), color=GREY, alpha=.15)
        ax.axvline(1, color=INK, ls=':', lw=.8); ax.axhline(0, color=GREY, lw=.75)
        ax.text(.03, .06, 'In range', transform=ax.transAxes, fontsize=8)
        ax.text(.60, .90, 'Overshoot', transform=ax.transAxes, fontsize=8)
        ax.set(xlabel=r'Residual ratio $r/B$', ylabel='Normalized response', xlim=(0, 4))
        ax.set_title('C  Analytic response difference', loc='left', fontweight='bold')
        ax = axes[1, 1]; result = polynomial_rank_reversal()
        q = np.array([float(Fraction(v)) for v in result['quadratic_ig_exact']]); n = np.array(result['stabilized_l2_ig'])
        for i, (values, color, label, hatch) in enumerate(((q, BLUE, 'Target Q', ''), (n, ORANGE, 'Target N', '//'))):
            ax.bar(np.arange(2)+(i-.5)*.32, values/values.sum(), .32, color=color, label=label, hatch=hatch, edgecolor=INK, linewidth=.75)
        ax.set(xticks=[0, 1], xticklabels=[r'Feature $x_1$', r'Feature $x_2$'], ylabel='Share of target endpoint gap', ylim=(0, .75))
        ax.set_title('D  Constructed IG reversal', loc='left', fontweight='bold'); ax.legend(frameon=False)
        self.populations['r1_design'] = {'construction': result, 'example_record_id': selected['record_id'],
                                       'example_selection': 'lexicographically first Q-ranked vision-deletion record with normalized Q minimum between -1.5 and -0.1; illustrative, not representative',
                                       'example_auc_q': float(selected['auc_q']), 'example_auc_n': float(selected['auc_l2'])}
        self.save(fig, out, 'r1_design')

    def success_rows(self):
        old = self.read('analysis/paired_rescoring/oneb_success/results/2026-10-23-v1/summary.csv')
        extra = self.read('analysis/paired_rescoring/extra_tasks/results/2026-10-23-v1/summary.csv')
        old_calls = self.read('analysis/paired_rescoring/oneb_success/results/2026-10-23-v1/calls.csv.gz')
        extra_calls = self.read('analysis/paired_rescoring/extra_tasks/results/2026-10-23-v1/calls.csv.gz')
        result = []
        for direction in ('insertion', 'deletion'):
            for outcome in ('success', 'failure'):
                r = next(r for r in old if r['population']=='pooled_primary' and r['modality']=='vision' and r['direction']==direction and r['outcome_group']==outcome)
                calls = [c for c in old_calls if c['role']=='primary' and c['modality']=='vision' and c['direction']==direction and c['success']==str(outcome=='success') and c['paired_status']=='defined']
                result.append(dict(group='Saved 1B '+outcome, direction=direction, episodes=int(r['n_defined_episodes']), calls=int(r['n_defined_calls']),
                    median=float(r['paired_median']), ci_lo=float(r['paired_median_ci_lo']), ci_hi=float(r['paired_median_ci_hi']),
                    overshoot=int(r['overshoot_calls']), below_minus_1=sum(float(c['delta']) < -1 for c in calls), minimum=float(r['min_delta'])))
            for success in (True, False):
                r = next(r for r in self.report['success_intervals'] if r['direction']==direction and r['success']==success)
                result.append(dict(r, group='New 1B '+('success' if success else 'failure')))
            for population, label, prefix in (('PegInsertionSide_pooled', 'Peg insertion', 'PegInsertionSide'), ('PickSingleYCB_pooled', 'Object pickup', 'PickSingleYCB')):
                r = next(r for r in extra if r['population']==population and r['modality']=='vision' and r['direction']==direction and r['outcome_group']=='all')
                calls = [c for c in extra_calls if c['run'].startswith(prefix) and c['role']=='primary' and c['modality']=='vision' and c['direction']==direction and c['paired_status']=='defined']
                result.append(dict(group=label, direction=direction, episodes=int(r['n_defined_episodes']), calls=int(r['n_defined_calls']),
                    median=float(r['paired_median']), ci_lo=float(r['paired_median_ci_lo']), ci_hi=float(r['paired_median_ci_hi']),
                    overshoot=int(r['overshoot_calls']), below_minus_1=sum(float(c['delta']) < -1 for c in calls), minimum=float(r['min_delta'])))
        return result

    def saved_geometry(self, out):
        data = self.read('analysis/paired_rescoring/residual_geometry/results/2026-10-23-v1/calls.csv.gz')
        data = [r for r in data if r['figure']=='headline' and r['paired_status']=='defined']
        summaries = {r['case']: r for r in self.read('analysis/paired_rescoring/results/2026-10-01-v1/summary.csv')}
        fig, axes = plt.subplots(2, 2, figsize=(5.85, 6.3), layout='constrained', gridspec_kw={'height_ratios': [1.2, 1]})
        rng = np.random.default_rng(0); counts = {}
        labels = [c.replace('L2', 'N').replace(':', ' ').replace('lang', 'language').replace('insertion', 'ins.').replace('deletion', 'del.') for c in CASES]
        for y, case in enumerate(CASES):
            vals = np.array([float(r['delta']) for r in data if r['case']==case]); jitter=rng.uniform(-.15, .15, len(vals))
            color = BLUE if case.startswith('Q:') else ORANGE
            low, high = int((vals < -.05).sum()), int((vals > .25).sum())
            if high: raise ValueError('Unexpected upper-window observations')
            counts[case] = {'calls':len(vals), 'outside_zoom_below':low, 'outside_zoom_above':high, 'outside_zoom':low+high}
            axes[0, 0].scatter(vals, y+jitter, s=12.25, color=color, alpha=.16, linewidths=0, rasterized=True)
            axes[0, 1].scatter(vals, y+jitter, s=12.25, color=color, alpha=.10, linewidths=0, rasterized=True)
            r=summaries[case]; lo, hi=float(r['paired_call_median_ci_lo']), float(r['paired_call_median_ci_hi'])
            axes[0, 1].plot([lo, hi], [y, y], color=INK, lw=1.5)
            for ax in axes[0]: ax.plot(np.median(vals), y, 'D', ms=4, color=INK)
            axes[0, 1].text(-.06, y, str(low), ha='right', va='center', fontsize=8, clip_on=False)
        for ax in axes[0]:
            ax.set_yticks(range(8)); ax.set_ylim(7.6, -.7); ax.axvline(0,color=GREY,lw=.75)
            ax.set_xlabel(r'Paired response change $\Delta$')
        axes[0, 0].set_yticklabels(labels); axes[0, 1].set_yticklabels([])
        axes[0, 0].set_xscale('symlog', linthresh=.25); axes[0, 0].set_xlim(-6500, .26)
        axes[0, 0].axvspan(-.25, .25, color='#dddddd', alpha=.25, zorder=0)
        axes[0, 0].set_xticks([-1000, -100, -10, -1, 0], ['−1000', '−100', '−10', '−1', '0'],rotation=35,ha='right')
        axes[0, 0].tick_params(axis='x', labelsize=8)
        axes[0, 0].set_title('A  Full saved range', loc='left', fontweight='bold')
        axes[0, 1].set_xlim(-.05, .25); axes[0, 1].set_title('B  Median intervals', loc='left', fontweight='bold')
        axes[0, 1].text(-.06, -.68, '< −0.05', fontsize=8, ha='right', va='bottom', clip_on=False)
        success = self.success_rows(); groups=list(dict.fromkeys(r['group'] for r in success))
        for r in success:
            i=groups.index(r['group']); offset=-.12 if r['direction']=='insertion' else .12
            marker='o' if r['direction']=='insertion' else 's'; y=i+offset
            axes[1, 0].plot([r['ci_lo'], r['ci_hi']], [y, y], color=INK, lw=1)
            axes[1, 0].plot(r['median'], y, marker, color=INK, mfc=INK, ms=4)
            axes[1, 1].plot(100*r['overshoot']/r['calls'], y, marker, color=INK, mfc='white', ms=5)
            axes[1, 1].plot(100*r['below_minus_1']/r['calls'], y, marker, color=INK, mfc=INK, ms=4)
        group_labels=[]
        for group in groups:
            r=next(r for r in success if r['group']==group)
            group_labels.append(f"{group}\n{r['episodes']} episodes, {r['calls']:,} calls")
        for ax in axes[1]: ax.set_yticks(range(len(groups))); ax.set_ylim(len(groups)-.5,-.5); ax.grid(axis='x', alpha=.2)
        axes[1, 0].set_yticklabels(group_labels); axes[1, 1].set_yticklabels([])
        axes[1, 0].set(xlabel=r'Median vision $\Delta$', xlim=(-.21,.20)); axes[1, 0].axvline(0,color=GREY,lw=.75)
        axes[1, 0].set_title('C  Success and task splits',loc='left',fontweight='bold')
        axes[1, 1].set(xlabel='Share of calls (%)',xlim=(-4,104)); axes[1, 1].set_title('D  Overshoots and tails',loc='left',fontweight='bold')
        handles=[Line2D([],[],color=BLUE,lw=3,label='Saved target Q'),Line2D([],[],color=ORANGE,lw=3,label='Saved target N'),
                 Line2D([],[],marker='o',color=INK,ls='none',label='Insertion'),Line2D([],[],marker='s',color=INK,ls='none',label='Deletion')]
        fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
        self.populations['r1_saved_geometry']={'saved_cases':counts,'success_task_rows':success,
            'tail_panel':'open markers: overshoot share; filled markers: delta below -1 share; shape indicates direction'}
        self.save(fig,out,'r1_saved_geometry')

    def tail_geometry(self,out):
        data=[r for r in self.read('analysis/paired_rescoring/residual_geometry/results/2026-10-23-v1/calls.csv.gz') if r['figure']=='headline' and r['paired_status']=='defined']
        fig,axes=plt.subplots(2,2,figsize=(5.85,5.5),layout='constrained')
        for i,direction in enumerate(('insertion','deletion')):
            rng=np.random.default_rng(0)
            for first in (False,True):
                selected=[r for r in data if ':vision:' in r['case'] and r['case'].endswith(direction) and (r['first_call']=='True')==first]
                selected=[selected[k] for k in rng.permutation(len(selected))]
                colors=[BLUE if r['case'].startswith('Q:') else ORANGE for r in selected]
                for j,key in enumerate(('residual_baseline','ratio_max_interior_to_baseline')):
                    axes[i,j].scatter([float(r[key]) for r in selected],[float(r['delta']) for r in selected],
                        s=20 if first else 12.25,marker='^' if first else 'o',color=colors,
                        alpha=.85 if first else .2,linewidths=.4 if first else 0,edgecolors=INK if first else 'none',rasterized=True)
            for j,ax in enumerate(axes[i]):
                ax.set_xscale('log');ax.set_yscale('symlog',linthresh=.25);ax.axhspan(-.25,.25,color='#ddd',alpha=.25,zorder=0)
                ax.set_title(f'{chr(65+i*2+j)}  Vision {direction}',loc='left',fontweight='bold')
                ax.set_xlabel('Baseline residual B' if j==0 else 'Largest interior residual / B')
                ax.set_ylabel(r'Paired response change $\Delta$');ax.set_ylim(-6500,.26)
                ax.set_yticks([-1000,-100,-10,-1,0],[r'$-10^3$',r'$-10^2$','-10','-1','0'])
                if j: ax.axvline(1,color=INK,ls=':',lw=.8);ax.axvline(2,color=GREY,ls='--',lw=.8)
        fig.legend(handles=[Line2D([],[],marker='o',color=BLUE,ls='none',label='Target Q'),Line2D([],[],marker='o',color=ORANGE,ls='none',label='Target N'),Line2D([],[],marker='^',color=INK,ls='none',label='First call')],loc='outside lower center',ncol=3,frameon=False)
        self.populations['r1_tail_geometry']={'scope':'saved vision, both directions and ranking targets','calls':3000,'first_call_shape':'triangle','y_scale':'symmetric log, linear within absolute delta 0.25'}
        self.save(fig,out,'r1_tail_geometry')

    def success_tasks(self,out):
        data=self.success_rows();groups=list(dict.fromkeys(r['group'] for r in data))
        fig,axes=plt.subplots(1,2,figsize=(5.85,3.8),layout='constrained',sharey=True)
        for ax,direction in zip(axes,('insertion','deletion')):
            for r in data:
                if r['direction']!=direction:continue
                y=groups.index(r['group']);ax.plot([r['ci_lo'],r['ci_hi']],[y,y],color=INK,lw=1)
                ax.plot(r['median'],y,'o',ms=4,color=INK)
            labels=[f"{g}\n{next(r['episodes'] for r in data if r['group']==g)} episodes, {next(r['calls'] for r in data if r['group']==g):,} calls" for g in groups]
            ax.set_yticks(range(len(groups)),labels);ax.set_ylim(len(groups)-.5,-.5);ax.axvline(0,color=GREY,lw=.75)
            ax.set_title('Vision '+direction,loc='left',fontweight='bold');ax.set_xlabel(r'Median $\Delta$ and 95% interval');ax.set_xlim(-.21,.20)
        self.populations['r1_success_tasks']={'rows':data,'intervals':'whole-episode bootstrap; new 1B intervals are retrospective'}
        self.save(fig,out,'r1_success_tasks')

    def methods(self,out):
        data=self.rows(self.inputs/'methods_advantage.csv')
        names=list(self.report['methods']['P']['raw_mean']['point']['Q']['values'])+['fixed_raster','fixed_reverse_raster','fixed_centre_out']
        fig,axes=plt.subplots(3,2,figsize=(5.85,6.6),layout='constrained',sharey=True)
        for i,bank in enumerate(('P','B1','S')):
            for j,response in enumerate(('Q','N')):
                ax=axes[i,j];selected={r['ranking']:r for r in data if r['bank']==bank and r['modality']=='vision' and r['direction']=='deletion' and r['response']==response}
                winner=self.report['methods'][bank]['raw_mean']['point'][response]['leaders']
                for y,name in enumerate(names):
                    r=selected[name];color=BLUE if '_Q_' in name else ORANGE if '_N_' in name else INK
                    marker='*' if name in winner else 'D' if name=='input_difference' else 'o'
                    ax.plot([float(r['interval95_lo']),float(r['interval95_hi'])],[y,y],color=color,lw=1)
                    ax.plot(float(r['mean']),y,marker,color=color,ms=6 if name in winner else 4)
                ax.axhline(6.5,color=GREY,lw=.75);ax.axvline(0,color=GREY,lw=.75)
                ax.set_yticks(range(len(names)),[self.labels[n] for n in names]);ax.set_ylim(9.5,-.5)
                ax.set_title(f'{chr(65+2*i+j)}  {BANKS[bank]}\nScoring {response}',loc='left',fontweight='bold')
                if i==2:ax.set_xlabel('Mean vision-deletion advantage')
        fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ls='none',label='Target Q'),
                            Line2D([],[],color=ORANGE,marker='o',ls='none',label='Target N'),
                            Line2D([],[],color=INK,marker='*',ls='none',label='Raw-mean winner'),
                            Line2D([],[],color=INK,marker='D',ls='none',label='Input difference unless winner')],
                   loc='outside lower center',ncol=2,frameon=False)
        self.populations['r1_methods']={'banks':{'P':60,'B1':40,'S':30},'scope':'vision deletion','intervals':'descriptive 95%; paired differences are separate','methods':names}
        self.save(fig,out,'r1_methods')

    def offline_numerics(self,out):
        """Retain the original twelve-context diagnostic trajectories separately."""
        comparisons=self.read('analysis/manuscript_revision/results/2026-10-03-v2/numerical_by_context.csv')
        projection=json.loads(self.source(self.root/'analysis/numerical_case/2026-10-01-v1/inputs/sealed_v6.json').read_text(encoding='utf-8'))
        cells=[r for r in projection['rows'] if r['status']=='complete_diagnostics']
        if len(cells)!=72 or len({r['context_id'] for r in cells})!=12:
            raise ValueError('Original offline diagnostic population changed')
        fig,axes=plt.subplots(2,3,figsize=(5.85,5.3),layout='constrained')
        populations={}
        for i,target in enumerate(('Q','L2')):
            color=BLUE if target=='Q' else ORANGE
            for j,metric in enumerate(('IG','path_gradient','completeness')):
                ax=axes[i,j];maximum=0
                for modality,marker,ls in (('vision','o','-'),('language','s','--'),('state','^',':')):
                    grouped=defaultdict(list)
                    if metric=='completeness':
                        for cell in cells:
                            if cell['target']==target and cell['modality']==modality:
                                grouped[cell['context_id']]=[(r['m'],100*r['relative_residual']) for r in cell['observations'] if r['repeat']==0]
                    else:
                        for row in comparisons:
                            if row['target']==target and row['modality']==modality and row['ranking']==metric and int(row['candidate_m'])*2==int(row['reference_m']):
                                grouped[row['context_id']].append((int(row['reference_m']),100*float(row['coordinate_relative_l1'])))
                    expected={r['context_id'] for r in cells if r['target']==target and r['modality']==modality}
                    if len(grouped)!=12 or set(grouped)!=expected:raise ValueError('Incomplete or mismatched offline figure group')
                    populations[f'{target}/{modality}/{metric}']=sorted(grouped)
                    for values in grouped.values():
                        xs,ys=zip(*sorted(values));maximum=max(maximum,max(ys))
                        ax.plot(xs,ys,color=color,ls=ls,marker=marker,ms=3.5,lw=.75,alpha=.55)
                ax.axhline(1,color=INK,ls=':',lw=.75)
                ax.set_xscale('log',base=2);ax.set_yscale('symlog',linthresh=.01)
                top=max(12,maximum*1.4);ax.set_ylim(0,top)
                ticks=[v for v in (0,.01,.1,1,10,100,1000) if v<=top]
                ax.set_yticks(ticks,[str(v) for v in ticks])
                budgets=[128,256,512,1024] if metric=='completeness' else [256,512,1024]
                ax.set_xticks(budgets,[str(v) for v in budgets],rotation=35,ha='right')
                ax.set_xlim(118 if metric=='completeness' else 235,1150)
                name={'IG':'IG coordinates','path_gradient':'Path gradients','completeness':'Completeness'}[metric]
                ax.set_title(f"{chr(65+3*i+j)}  Target {'Q' if target=='Q' else 'N'}\n{name}",loc='left',fontweight='bold')
                ax.grid(axis='y',alpha=.15)
                if j==0:ax.set_ylabel('Relative difference (%)')
                if j==2:ax.set_ylabel('Relative residual (%)')
                if i==1:ax.set_xlabel('Intervals' if j==2 else 'Finer intervals')
        fig.legend(handles=[Line2D([],[],color=INK,marker=m,ls=ls,ms=3.5,label=name)
                            for name,m,ls in (('Vision','o','-'),('Language','s','--'),('State','^',':'))],
                   loc='outside lower center',ncol=3,frameon=False)
        self.populations['r1_offline_numerics']={'contexts':12,'episodes':6,'completed_arms':72,'panels':populations,
            'scope':'Original sealed offline 32-bit downstream probe, not the 21-context reruns or new Figure3',
            'zero_handling':'Exact zeros retained on symmetric-log axes, linear through 0.01 percent'}
        self.save(fig,out,'r1_offline_numerics')

    def tables(self,out):
        self.main_tables(out)
        self.supplement_tables(out)

    def main_tables(self,out):
        experiments=[
            ['Saved PickCube 170M, target Q','30 (0) / 750','Equal weights, 64*','bf16 / bf16','Scored curves'],
            ['Saved PickCube 170M, target N','30 (0) / 750','Equal weights, 64','bf16 / bf16','Scored curves'],
            ['Saved PickCube 1B, target Q','60 (15) / 1,199','Equal weights, 64','bf16 / bf16','Scored curves'],
            ['Saved peg and object tasks, 170M, target Q','16 (0) / 400 each','Equal weights, 128*','bf16 / bf16','Scored curves'],
            ['New PickCube 170M, targets Q/N','60 (0) / 120','Trapezoid, 512; checks','bf16 / fp32; fp32 subset','Complete contexts'],
            ['New PickCube 1B, targets Q/N','40 (31) / 80','Trapezoid, 512; old rule','bf16 / fp32','Complete contexts'],
            ['New StackCube 170M, targets Q/N','30 (0) / 60','Trapezoid, 512; old rule','bf16 / fp32','Complete contexts'],
            ['Preserved four-task sample','21 contexts','Numerical checks','bf16 and fp32','Contexts'],
            ['Separate offline diagnostics','6 / 12 per input and target','Trapezoid, 128 to 1,024','fp32 / fp32','Contexts and outputs']]
        table(out,'r1_experiments',r'\textbf{Study groups answer different questions.} Counts are episodes (successes) / calls unless stated. Precision is model / accumulation: bf16 is 16-bit, fp32 32-bit. Budgets count intervals; equal weighting uses one extra evaluation. Asterisks indicate inferred budgets. The saved 1B rerun is excluded. Preserved and offline samples overlap. Supplementary Tables S2 and S8 give identities and variants.', 'tab:r1-experiments',
              ['Group','Episodes / calls','Rule and budget','Precision','Preserved'],experiments,[.25,.14,.21,.18,.14])
        contrasts=[r for r in self.rows(self.inputs/'contrasts.csv') if r['bank']=='P' and r['metric']=='normalized_auc']
        if len(contrasts)!=7:raise ValueError('Expected exactly seven primary contrasts')
        data=[]
        for method,control,label in (('IG_Q_A1T512','','IG-Q minus random'),('IG_N_A1T512','','IG-N minus random'),('IG_Q_A1T512','IG_N_A1T512','IG-Q minus IG-N')):
            row=[label]
            for response in ('Q','N'):
                r=next(r for r in contrasts if r['method']==method and r['control']==control and r['response']==response)
                row.append(number(r['estimate'])+' '+interval(r['percentile_lo'],r['percentile_hi']))
            data.append(row)
        mixed=next(r for r in contrasts if r['contrast']=='matched_calibrated')
        data.append(['IG-Q at Q minus IG-N at N',number(mixed['estimate'])+' '+interval(mixed['percentile_lo'],mixed['percentile_hi']),'Mixed responses'])
        table(out,'r1_methods_summary',r"\textbf{Primary contrasts and response-dependent choices.} A: seven preplanned vision-deletion contrasts on 60 PickCube 170M episodes, with Bonferroni-adjusted intervals. The last row mixes response units. B: vision-deletion point-estimate winners. Scaling divides each call's advantage by its random-order standard deviation before averaging; medians use episode means. IG uses a 512-interval trapezoid, equal-weight IG 64 intervals, both with 16-bit model calculations. Winners need not separate statistically. Supplementary Tables S3 and S4 retain all comparisons and controls.",
              'tab:r1-methods-summary',['Contrast','Q: estimate [interval]','N: estimate [interval]'],data,[.35,.29,.29])
        target=out/'r1_methods_summary.tex';content=target.read_text(encoding='utf-8').replace(r'\end{table}','')
        content=content.replace(r'\begin{tabular}',r'\par\textbf{A. Primary contrasts}\par\smallskip\begin{tabular}',1)
        content+='\n'+r'\par\smallskip\textbf{B. Point-estimate choices}\par\smallskip\begin{tabular}{p{.22\linewidth}p{.08\linewidth}p{.21\linewidth}p{.21\linewidth}p{.21\linewidth}}'+'\n'+r'\toprule \textbf{Group (episodes)} & \textbf{Score} & \textbf{Raw mean} & \textbf{Scaled mean} & \textbf{Episode median} \\ \midrule'+'\n'
        for bank,n in (('P',60),('B1',40),('S',30)):
            for response in ('Q','N'):
                cells=[f'{BANKS[bank]} ({n})' if response=='Q' else '',response]
                for criterion in ('raw_mean','scaled_mean','episode_median'):
                    winners=self.report['methods'][bank][criterion]['point'][response]['leaders']
                    cells.append(self.labels[winners[0]] if len(winners)==1 else 'Exact tie')
                content+=' & '.join(text(v) for v in cells)+r' \\'+'\n'
        content+=r'\bottomrule\end{tabular}\end{table}'+'\n'
        target.write_text(content,encoding='utf-8',newline='\n')
        self.populations['r1_methods_summary']={'primary_contrasts':contrasts,'preference_criteria':['raw_mean','scaled_mean','episode_median'],'vision_cells':6}

    def supplement_tables(self,out):
        paired=self.read('analysis/manuscript_revision/results/2026-10-03-v2/paired_summary.csv')
        ci={r['case']:r for r in self.read('analysis/paired_rescoring/results/2026-10-01-v1/summary.csv')}
        data=[]
        for r in paired:
            c=ci[r['case']];label=r['case'].replace('L2','N').replace(':',' ').replace('lang','language')
            data.append([label,f"{float(r['median']):.4f} "+interval(c['paired_call_median_ci_lo'],c['paired_call_median_ci_hi']),
                         f"{float(r['positive_percent']):.1f} / {float(r['overshoot_percent']):.1f}",
                         number(r['mean'])+' '+interval(r['mean_ci_low'],r['mean_ci_high']),number(r['minimum'])])
        table(out,'r1_saved_summary',r'\textbf{All eight saved response-change cases.} Each row contains 750 calls in 30 episodes. Q/N identifies the ranking target. Brackets give descriptive 95\% whole-episode bootstrap intervals, conditional on recorded episodes, not variation across training runs, checkpoints or tasks. Medians weight calls equally and means weight episodes equally. Positive means $\Delta>0$; overshoot means $r>B$ at an interior sample. Both columns are percentages. These unmatched runs do not compare ranking quality. Three negative mean intervals include zero.',
              'tab:r1-saved-summary',['Target, input, direction','Median [interval]','Positive / overshoot (%)','Mean [interval]','Minimum'],data,[.23,.23,.14,.23,.10],long=True)
        self.identity_table(out)
        decisions=[r for r in self.rows(self.inputs/'methods_decisions.csv') if r['bank'] in BANKS]
        if len(decisions)!=20:raise ValueError(f'Expected 20 method cells, got {len(decisions)}')
        data=[];cal=[]
        advantages=self.rows(self.inputs/'methods_advantage.csv')
        for r in decisions:
            choice=self.labels.get(r['preferred'],'Exact tie');name=f"{BANKS[r['bank']]} {r['modality']}"
            cell=r['response']+' '+r['direction']
            preferred=next(a for a in advantages if a['bank']==r['bank'] and a['modality']==r['modality'] and a['direction']==r['direction'] and a['response']==r['response'] and a['ranking']==r['preferred'])
            data.append([name,cell,choice,number(preferred['mean']),number(r['best_minus_second_mean'])+' '+interval(r['best_minus_second_adjusted_lo'],r['best_minus_second_adjusted_hi']),
                         'Yes' if r['unique']=='True' else 'No'])
            cal.append([name,cell,self.labels.get(r['preferred_z_scaled'],'Exact tie'),self.labels.get(r['best_fixed'],r['best_fixed']),
                        number(r['best_minus_best_fixed_mean'])+' '+interval(r['best_minus_best_fixed_adjusted_lo'],r['best_minus_best_fixed_adjusted_hi'])])
        table(out,'r1_all_methods',r'\textbf{All 20 new-experiment method comparisons.} The winner maximizes the raw episode-mean advantage. The gap is winner minus runner-up, with the registered interval adjusted across four response/direction cells within each group and input type. These selected contrasts do not correct for choosing the winner on the same data, nor across all groups. Three intervals exclude zero: one vision cell and two language cells.',
              'tab:r1-all-methods',['Group and input','Scoring / direction','Raw winner','Mean advantage','Gap [adjusted interval]','Excludes zero'],data,[.21,.14,.18,.11,.21,.08],long=True)
        table(out,'r1_calibration',r'\textbf{Scaled choices and fixed-order comparisons.} Scaling is per context before episode averaging. Fixed orders use no learned attribution. The gap compares the best raw attribution method with the highest-mean fixed order; negative values favor that fixed order. Brackets retain the registered adjusted intervals. Mean comparisons can be tail-sensitive and do not establish a generally superior explanation.',
              'tab:r1-calibration',['Group and input','Scoring / direction','Scaled winner','Best fixed order','Gap [adjusted interval]'],cal,[.23,.16,.19,.17,.18],long=True)
        self.fixed_bar_table(out)
        self.fixed_order_tail_table(out)
        self.diagnostic_table(out)
        success=self.success_rows();data=[]
        for r in success:
            data.append([r['group']+' '+r['direction'],f"{r['episodes']} / {r['calls']}",f"{r['median']:.4f}".replace('-','−')+' '+interval(r['ci_lo'],r['ci_hi']),
                         f"{r['overshoot']}/{r['calls']}",f"{r['below_minus_1']}/{r['calls']}",number(r['minimum'])])
        table(out,'r1_success_summary',r'\textbf{Vision response changes by success and task.} Counts are episodes / calls. All intervals resample whole episodes; the new 1B intervals are retrospective. Overshoot and extreme-tail counts are call counts. Unequal sample sizes affect extrema, and episode dependence remains. The saved 1B rows use the 64-interval equal-weight rule, the new 1B rows the 512-interval trapezoid, and the additional tasks the 128-interval equal-weight rule. All use target Q. All additional-task episodes failed.',
              'tab:r1-success-summary',['Group / direction','Episodes / calls','Median [95% interval]','Overshoot','Below -1','Min.'],data,[.25,.12,.24,.11,.11,.09],long=True)
        self.language_success_table(out)
        data=[]
        for bank in ('P','B1','S'):
            for criterion,label in (('raw_mean','Raw mean'),('scaled_mean','Scaled mean'),('episode_median','Episode median')):
                r=self.report['methods'][bank][criterion];s=r['bootstrap_shares']
                data.append([BANKS[bank],label,*[f'{100*s[k]:.2f}' for k in ('same_unique_winner','different_unique_winners','at_least_one_tie')]])
        table(out,'r1_stability',r'\textbf{Retrospective stability of vision-deletion choices.} Entries are percentages of 10,000 paired whole-episode bootstrap draws, seed 930011. Every draw uses the same episodes for both responses and all seven methods. Exact ties are retained separately. These are resampling frequencies for these episodes, not probabilities of a true population winner. The median criterion uses per-episode mean raw advantages.',
              'tab:r1-stability',['Group','Criterion','Same winner (%)','Different winners (%)','Any tie (%)'],data,[.24,.21,.16,.18,.13],long=True)
        table(out,'r1_estimators',r'\textbf{Numerical variants and their purpose.} The old rule generated saved attributions and is reapplied to new contexts; these are distinct data sources. Numerical agreement is a diagnostic, not an exactness guarantee.',
              'tab:r1-estimators',['Variant','Calculation / sum','Rule and evaluations','Purpose'],[
                  ['Old equal-weight rule','bf16 / bf16','65 uniformly spaced nodes, equal weights','Recover the old implementation'],
                  ['Reduced-precision check','bf16 / fp32','513 shared nodes: equal-weight/trapezoid through 512 intervals; midpoint through 256','Separate summation from model precision'],
                  ['32-bit check','fp32 / fp32','The same 513 nodes and nested rules','Matched precision comparison'],
                  ['Gauss-Legendre','fp32 / fp32','32, 64 or 128 interior nodes','Independent rule comparison'],
                  ['Adaptive Gauss-Kronrod','fp32 / fp32','Adaptive interior nodes','Independent convergence check']], [.23,.18,.27,.24],long=True)
        # longtable increments the table counter even when its caption is starred.
        # Continuations belong to the preceding S-number, not a new numbered table.
        for path in out.glob('r1_*.tex'):
            content=path.read_text(encoding='utf-8')
            content=re.sub(r'(\\begin\{longtable\}\{[^\n]+\}\n)(?=\\caption\*)',
                           lambda match: match.group(1)+r'\noalign{\addtocounter{table}{-1}}'+'\n',content)
            path.write_text(content,encoding='utf-8',newline='\n')

    def identity_table(self,out):
        projection=json.loads(self.source(self.root/AUDIT/'inputs/identity_projection.json').read_text(encoding='utf-8'))
        selected={p['id']:p for p in projection['populations']}
        columns=['saved_Q_ranked_cohort','saved_N_ranked_cohort','R1_P','R1_B1','R1_S']
        field_names=list(selected[columns[0]]['fields'])
        data=[]
        for field in field_names:
            if field=='fingerprint':continue
            data.append([field.replace('_',' '),*[selected[p]['fields'].get(field,{'status':'not applicable'})['status'] for p in columns]])
        table(out,'r1_identities',r'\textbf{Component identity status for the main saved and new groups.} Recorded means captured during that experiment; recovered means reconstructed from dated evidence; inferred means supported but not established; unknown means unavailable. Recovery does not imply a runtime hash. Full values and all 12 population inventories accompany the public source projection. Saved checkpoint and encoder files can be recovered, while most runtime versions remain unknown. New runtime entries include incomplete host captures and do not establish every host individually.',
              'tab:r1-identities',['Component','Saved Q','Saved N','New 170M PickCube','New 1B PickCube','New 170M StackCube'],data,[.29,.12,.12,.13,.13,.13],long=True)
        values=json.loads(self.source(self.inputs/'identity_values.json').read_text(encoding='utf-8'))
        value_rows=[[r['component'],r['value'].replace('d10de3d)','d10de3d61747)').replace('384 x 384','384 × 384'),r['status']] for r in values['rows']]
        path=out/'r1_identities.tex';existing=path.read_text(encoding='utf-8')
        remaining=[p['id'] for p in projection['populations'] if p['id'] not in columns]
        population_names={'saved_RDT1B_seed42_month4':'Saved 1B, seed 42',
            'saved_RDT1B_seeds142_242_month6':'Saved 1B, seeds 142/242',
            'saved_RDT1B_seed42_month5_rerun':'Saved 1B rerun',
            'saved_m128_seed142_month4':'Additional tasks, seed 142',
            'saved_m128_seed42_month7_reruns':'Additional tasks, seed 42',
            'month8_context_bank_and_numerical_sample':'Four-task collection',
            'R1_M8_replay':'Preserved-context checks'}
        for begin in (0,4):
            group=remaining[begin:begin+4]
            if not group:continue
            data=[[field.replace('_',' '),*[selected[p]['fields'].get(field,{'status':'not applicable'})['status'] for p in group]]
                  for field in field_names if field!='fingerprint']
            table(out,'r1_status_continuation',r'\textbf{Identity status, continued: additional saved and preserved populations.} The source projection retains full values and evidence status for every field.',
                  'tab:r1-status-continuation',['Component']+[population_names[p] for p in group],data,[.32]+[.60/len(group)]*len(group),long=True)
            extra=(out/'r1_status_continuation.tex').read_text(encoding='utf-8').replace(r'\caption{',r'\caption*{').replace(r'\label{tab:r1-status-continuation}','')
            existing+='\n'+extra
        (out/'r1_status_continuation.tex').unlink()
        table(out,'r1_identity_values',r'\textbf{Values for the new experiment.} Revisions and digests are 12-character prefixes. The saved 170M checkpoint and SigLIP revision were recovered as the same files, without runtime hashes. Complete per-population values remain in the source projection.',
              'tab:r1-identity-values',['Component','Value','Status'],value_rows,[.21,.57,.14],long=True)
        # Keep S2 one numbered table; the continuation uses an unnumbered table.
        values_text=(out/'r1_identity_values.tex').read_text(encoding='utf-8')
        values_text=values_text.replace(r'\caption{',r'\caption*{').replace(r'\label{tab:r1-identity-values}','')
        path.write_text(existing+'\n'+values_text,encoding='utf-8',newline='\n')
        (out/'r1_identity_values.tex').unlink()

    def fixed_bar_table(self,out):
        rows=[]
        for bank in ('P','B1','S'):
            d=json.loads(self.source(self.r1/'methods'/bank/'decisions.json').read_text(encoding='utf-8'))['vision']['cells']
            for response,key in (('Q','Q/deletion'),('N','L2/deletion')):
                c=d[key];r=[BANKS[bank],response]
                for bar in ('0.4','0.5'):
                    shares=c['bars'][bar]['call_share_at_or_below'];best=max(shares.values())
                    names=[self.labels[n] for n,v in shares.items() if v==best]
                    r.append('; '.join(names)+f' ({100*best:.1f}%)')
                rows.append(r)
        path=out/'r1_calibration.tex';before=path.read_text(encoding='utf-8')
        table(out,'r1_bar_details',r'\textbf{Illustrative fixed-bar choices for vision deletion.} Each cell lists all methods tied for the largest fraction of calls with AUC at or below the bar, and that fraction. These arbitrary bars are not validated decision rules. This criterion differs from the raw mean choice in four of six cells for either bar.',
              'tab:r1-bar-details',['Group','Scoring','Bar 0.40','Bar 0.50'],rows,[.23,.09,.30,.30],long=True)
        addition=(out/'r1_bar_details.tex').read_text(encoding='utf-8').replace(r'\caption{',r'\caption*{').replace(r'\label{tab:r1-bar-details}','')
        path.write_text(before+'\n'+addition,encoding='utf-8',newline='\n');(out/'r1_bar_details.tex').unlink()

    def language_success_table(self,out):
        # These runs deliberately remain separate because their language baselines differ.
        self.source(self.root/'analysis/paired_rescoring/oneb_success/protocol.json')
        source_rows=self.read('analysis/paired_rescoring/oneb_success/results/2026-10-23-v1/summary.csv')
        data=[]
        for run,label in (('seed42','42, zeroed'),('seed142','142, encoded'),('seed242','242, encoded'),('m5_seed42','42 rerun, encoded')):
            for outcome in ('success','failure'):
                for direction in ('insertion','deletion'):
                    r=next(r for r in source_rows if r['population']==run and r['modality']=='lang' and r['direction']==direction and r['outcome_group']==outcome)
                    ci=interval(r['paired_median_ci_lo'],r['paired_median_ci_hi']) if r['paired_median_ci_lo'] else 'Not estimable'
                    data.append([label,outcome+' '+direction,r['n_defined_episodes']+' / '+r['n_defined_calls'],f"{float(r['paired_median']):.4f}".replace('-','−')+' '+ci,number(100*float(r['overshoot_share'])),number(r['min_delta'])])
        path=out/'r1_success_summary.tex';before=path.read_text(encoding='utf-8')
        table(out,'r1_language_success',r'\textbf{Saved task-trained language results, separate by seed and baseline.} Seed 42 uses the RDT developers\textquotesingle{} precomputed instruction embedding and a zeroed baseline; seeds 142 and 242 use generated T5-XXL embeddings and encoded PAD/EOS baselines. The separate seed-42 rerun uses generated embeddings and an encoded baseline and is a sensitivity sample, excluded from the 60 primary episodes. All use target Q. Intervals resample whole episodes. The single successful seed-142 episode cannot support an episode-bootstrap interval. Overshoots are percentages of calls.',
              'tab:r1-language-success',['Seed, baseline','Outcome / direction','Episodes / calls','Median [95% interval]','Overshoot (%)','Min.'],data,[.17,.22,.12,.23,.11,.07],long=True)
        addition=(out/'r1_language_success.tex').read_text(encoding='utf-8').replace(r'\caption{',r'\caption*{').replace(r'\label{tab:r1-language-success}','')
        path.write_text(before+'\n'+addition,encoding='utf-8',newline='\n');(out/'r1_language_success.tex').unlink()

    def fixed_order_tail_table(self,out):
        data=[];details=[]
        for bank,response,direction in (('P','Q','insertion'),('P','N','insertion'),('B1','Q','deletion')):
            archive_response='L2' if response=='N' else response
            decisions=json.loads(self.source(self.r1/'methods'/bank/'decisions.json').read_text(encoding='utf-8'))['vision']['cells'][archive_response+'/'+direction]
            episodes=decisions['advantage']['episode_ids']
            source_rows=self.rows(self.r1/'methods'/bank/'context_rows_vision.csv')
            summary=next(r for r in self.rows(self.inputs/'methods_decisions.csv') if r['bank']==bank and r['modality']=='vision' and r['response']==response and r['direction']==direction)
            methods=list(dict.fromkeys([summary['best_fixed'],summary['preferred'],summary['preferred_z_scaled']]+(['fixed_centre_out'] if bank=='B1' else [])))
            for method in methods:
                grouped=defaultdict(list);scaled=defaultdict(list)
                for row in source_rows:
                    if row['response']==archive_response and row['direction']==direction and row['ranking']==method and row['episode_id'] in episodes:
                        grouped[row['episode_id']].append(float(row['advantage']));scaled[row['episode_id']].append(float(row['z']))
                if set(grouped)!=set(episodes):raise ValueError('Missing complete episode in fixed-order diagnostic')
                raw=np.array([np.mean(grouped[e]) for e in episodes]);z=np.array([np.mean(scaled[e]) for e in episodes])
                record=dict(bank=bank,response=response,direction=direction,ranking=method,episodes=len(episodes),
                            raw_mean=float(np.mean(raw)),episode_median=float(np.median(raw)),positive_share=float(np.mean(raw>0)),scaled_mean=float(np.mean(z)))
                details.append(record)
                data.append([f"{BANKS[bank]}, {response} {direction} ({len(episodes)})",self.labels[method],number(record['raw_mean']),
                             f"{record['episode_median']:.4f}".replace('-','−'),f"{100*record['positive_share']:.1f}",number(record['scaled_mean'])])
        path=out/'r1_calibration.tex';before=path.read_text(encoding='utf-8')
        table(out,'r1_fixed_tail_details',r'\textbf{Fixed-order raw means can be driven by unusual episodes.} These three vision cells have a fixed-order raw mean above every attribution method. Rows retain that fixed winner, the raw and scaled attribution winners, and the additional task-trained center-out comparator. Counts in parentheses are complete episodes. All statistics first average calls within episodes; positive is the percentage of episode-mean advantages above zero. Scaling divides each call by its random-order standard deviation before averaging. These descriptive contrasts show sensitivity to aggregation, not significance or general method superiority.',
              'tab:r1-fixed-tail-details',['Group, scoring, direction (episodes)','Method / order','Raw mean','Episode median','Positive (%)','Scaled mean'],data,[.25,.22,.11,.12,.10,.12],long=True)
        addition=(out/'r1_fixed_tail_details.tex').read_text(encoding='utf-8').replace(r'\caption{',r'\caption*{').replace(r'\label{tab:r1-fixed-tail-details}','')
        path.write_text(before+'\n'+addition,encoding='utf-8',newline='\n');(out/'r1_fixed_tail_details.tex').unlink()
        self.populations['r1_fixed_order_tails']=details

    def diagnostic_table(self,out):
        rows=self.rows(self.r1/'numerics/qualification.csv');data=[]
        for target in ('Q','N'):
            for modality in ('vision','language','state'):
                r=[r for r in rows if r['target']==target and r['modality']==modality]
                data.append([target+' '+modality,f"{sum(x['qualified']=='True' for x in r)}/{len(r)}",
                    *[number(np.median([float(x[field]) for x in r])) for field in ('relative_l1[A4|A3GL128]','relative_l1[A4|A2T512]','relative_l1[A3GL128|A2T512]')]])
        table(out,'r1_diagnostics',r'\textbf{Reference qualification distinguishes agreement from accuracy.} Eight contexts comprise four new PickCube contexts and four preserved contexts, one per task. Columns show median relative L1 differences (fractions). Qualification retains the registered three-way 0.001 tolerance and adaptive convergence condition: Q 15/24, N 5/24. Close adaptive/Gauss-Legendre agreement for N vision and language, with a larger trapezoid discrepancy, is consistent with unresolved endpoint behavior; it does not certify an exact integral. State retains larger independent-rule differences.',
              'tab:r1-diagnostics',['Target / input','Qualified','Adaptive vs Gauss','Adaptive vs trapezoid','Gauss vs trapezoid'],data,[.22,.12,.20,.20,.18],long=True)
        target_path=out/'r1_diagnostics.tex';content=target_path.read_text(encoding='utf-8')
        def append(caption,headers,data,widths):
            nonlocal content
            table(out,'r1_diagnostic_details',caption,'tab:r1-diagnostic-details',headers,data,widths,long=True)
            extra=(out/'r1_diagnostic_details.tex').read_text(encoding='utf-8').replace(r'\caption{',r'\caption*{').replace(r'\label{tab:r1-diagnostic-details}','')
            content+='\n'+extra
        residual=self.rows(self.r1/'numerics/residuals.csv')
        kept={r['context'] for r in residual if r['bank']=='P' and r['modality']=='vision' and r['arm']=='A2'}
        data=[]
        for target in ('Q','N'):
            for arm,label in (('A1','16-bit model'),('A2','32-bit model')):
                values=[]
                for budget in ('64','512'):
                    r=[r for r in residual if r['bank']=='P' and r['context'] in kept and r['modality']=='vision' and r['target']==target and r['arm']==arm and r['rule']=='T' and r['budget']==budget]
                    if len(r)!=12:raise ValueError('Wrong matched completeness population')
                    values.append(number(100*np.median([float(x['relative_residual']) for x in r])))
                data.append([target,label,'12',*values])
        append(r'\textbf{Matched trapezoidal completeness residuals.} These percentages use the same twelve pretrained-PickCube vision contexts in every row. Model precision changes; accumulation remains 32-bit. Figure 3 lower panels use eight of these twelve contexts because the 16-bit rule maps were available for only eight. Those eight are not the mixed-task qualification sample above. This comparison supports the precision statement without mixing the 120-, twelve- and eight-context subsets.',
               ['Target','Calculation','Contexts','64 intervals (%)','512 intervals (%)'],data,[.12,.25,.12,.22,.21])
        comparisons=self.rows(self.r1/'numerics/comparison_summary_by_bank.csv')
        rank=self.rows(self.r1/'numerics/scored_order_spearman_summary.csv')
        auc=self.rows(self.r1/'numerics/auc_change_summary.csv')
        data=[]
        for target in ('Q','N'):
            coord=next(r for r in comparisons if r['bank']=='P' and r['modality']=='vision' and r['target']==target and r['family']=='bf16_vs_A1_comparator' and r['candidate']=='A0m64')
            order=next(r for r in rank if r['group']=='P' and r['modality']=='vision' and r['target']==target and r['candidate']==f'IG_{target}_A0m64' and r['reference']==f'IG_{target}_A1T512')
            for response in ('Q','L2'):
                area=next(r for r in auc if r['group']=='P' and r['modality']=='vision' and r['candidate']==f'IG_{target}_A0m64' and r['reference']==f'IG_{target}_A1T512' and r['direction']=='deletion' and r['response']==response)
                data.append([target+'/'+response.replace('L2','N'),coord['n'],number(coord['median']),f"{float(order['min']):.5f}",number(area['median'])+' / '+number(area['max']),area['over_0_01']+'/'+area['n']])
        append(r'\textbf{Coordinate, ranking and perturbation diagnostics remain separate.} Each comparison applies the old equal-weight 64-interval estimator to new pretrained-PickCube vision contexts and compares it with the corrected 16-bit, 512-interval trapezoid. Coordinate differences are relative L1 fractions. Ranking entries are minimum scored-order Spearman correlations; AUC changes concern deletion. Target and scoring response are separate in the first column.',
               ['Target / score','Contexts','Median L1','Min. rank correlation','Median / max AUC change','Changes above 0.01'],data,[.15,.10,.15,.16,.23,.13])
        data=[]
        names={'M8-pickcube-170m':'PickCube 170M','M8-pickcube-1b':'PickCube 1B',
               'M8-stackcube-170m':'StackCube 170M','M8-stackcube-1b':'StackCube 1B',
               'M8-peginsertionside-170m':'Peg insertion 170M','M8-picksingleycb-170m':'Object pickup 170M'}
        for bank,label in names.items():
            for modality in ('vision','language','state'):
                for target in ('Q','N'):
                    cells=[];n=None
                    for arm in ('A0','A1','A2'):
                        rs=[r for r in residual if r['bank']==bank and r['modality']==modality and r['target']==target and r['arm']==arm
                            and (r['budget']=='64' if arm=='A0' else r['rule']=='T' and r['budget']=='512')]
                        if not rs: raise ValueError(f'Missing preserved-context results: {bank} {modality} {target} {arm}')
                        n=len(rs) if n is None else n
                        if len(rs)!=n:raise ValueError('Unmatched preserved-context comparison')
                        cells.append(number(100*np.median([float(r['relative_residual']) for r in rs])))
                    data.append([label,modality+' '+target,str(n),*cells])
        append(r'\textbf{The 21 preserved contexts also receive numerical checks.} Entries are median completeness residuals in percent, with the same contexts across each row. These 21 contexts are all available members of the 24-context collection underlying the separate twelve-context completed diagnostic sample. They are not independent new robot runs. Each stratum has only two source episodes and supports no population inference. Old denotes the bf16 equal-weight 64-interval estimator. The two corrected columns use the 512-interval trapezoid and 32-bit accumulation, with 16-bit or 32-bit model calculations.',
               ['Preserved group','Input / target','Contexts','Old (%)','16-bit (%)','32-bit (%)'],data,[.24,.18,.10,.13,.13,.14])
        original=self.read('analysis/numerical_case/2026-10-01-v1/outputs/modality_target.csv')
        original_cells=self.read('analysis/numerical_case/2026-10-01-v1/outputs/per_cell.csv')
        original_summary=json.loads(self.source(self.root/'analysis/numerical_case/2026-10-01-v1/outputs/summary.json').read_text(encoding='utf-8'))
        if original_summary['repeatability_equality_check_counts']!={'satisfies':1800}:
            raise ValueError('Original repeatability count changed')
        data=[]
        for row in original:
            cells=[r for r in original_cells if r['modality']==row['modality'] and r['target']==row['target']]
            if len(cells)!=12:raise ValueError('Original diagnostic group must contain twelve contexts')
            for aggregate,field in (('max_finer_pair_coordinate_relative_l1_IG','coordinate_relative_l1_IG'),
                                    ('max_finest_relative_residual','finest_relative_residual'),
                                    ('max_rms_curve_difference_over_baseline','max_rms_curve_difference_over_baseline')):
                if not math.isclose(float(row[aggregate]),max(float(r[field]) for r in cells),rel_tol=1e-12,abs_tol=1e-15):
                    raise ValueError('Original offline aggregate differs from its per-context records')
            data.append([row['modality']+' '+row['target'].replace('L2','N'),row['contexts_with_finest_pair_threshold_violation']+'/12',
                number(100*float(row['max_finer_pair_coordinate_relative_l1_IG'])),number(100*max(float(r['coordinate_relative_l1_path_gradient']) for r in cells)),
                number(100*float(row['max_finest_relative_residual'])),number(100*float(row['max_rms_curve_difference_over_baseline']))])
        append(r'\textbf{Separate original offline diagnostics: twelve contexts, six episodes, four pretrained tasks.} This is the completed subset of the sealed collection, distinct from the later 21-context reruns. The model and summation use 32-bit arithmetic on stored 16-bit-valued weights and encoded image/language inputs; state is readapted and the self-reference recomputed in 32 bits. It is not a native 32-bit rollout. All 1,800 recorded equality checks passed, but repeatability did not establish quadrature accuracy. Entries are maxima for the two finest budgets: 256 versus 512 intervals for vision/language, 512 versus 1,024 for state. The finer estimate is a comparator, not ground truth. Any failed check counts contexts violating at least one recorded finer-pair criterion; these dependent checks do not estimate a failure frequency. IG and path-gradient columns are relative L1 differences. RMS is the largest pointwise response-curve change divided by baseline RMS, across both ranking types and directions. Extrema can come from different contexts. Two state contexts exceed 1\% finest-budget completeness under each target. Figure S4 retains all adjacent-budget trajectories, including exact zeros.',
               ['Input / target','Any failed check','Max. IG L1 (%)','Max. path L1 (%)','Max. residual (%)','Max. RMS (%)'],data,[.18,.13,.16,.16,.16,.14])
        self.populations['r1_original_offline_summary']={'contexts':12,'episodes':6,'completed_arms':len(original_cells),'equality_checks_passed':1800,'modality_target_rows':original,
                                                     'source':'analysis/numerical_case/2026-10-01-v1/outputs/per_cell.csv'}
        target_path.write_text(content,encoding='utf-8',newline='\n');(out/'r1_diagnostic_details.tex').unlink()
