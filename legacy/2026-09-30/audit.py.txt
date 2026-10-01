import json, glob, os
import numpy as np

#np.trapz was removed in NumPy 2.0 (renamed trapezoid); same shim as faithfulness.py.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def load(fn, count_skips=False):
    rows = []
    skipped = 0
    with open(os.path.join(DATA, fn) if not fn.startswith('/') else fn, 'rb') as f:
        for raw in f:
            if raw.strip() == b'':
                continue
            try:
                d = json.loads(raw)
            except Exception:
                skipped += 1
                continue
            rows.append(d)
    if count_skips:
        return rows, skipped
    return rows

def med(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return float('nan')
    return float(np.median(a))

def pct_le(vals, thr=0.03):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return float('nan')
    return float(np.mean(a <= thr) * 100.0)

def sig_step(rows):
    return [r for r in rows if r.get('event') == 'step' and r.get('ref_norm_maniskill', -1) >= 15]

print("="*70)
print("SKIP TRACKING (look for NUL corrupted lines)")
total_skips = 0
for fn in sorted(os.listdir(DATA)):
    if fn.endswith('.jsonl'):
        rows, sk = load(fn, count_skips=True)
        if sk > 0:
            print(f"  {fn}: skipped {sk}")
            total_skips += sk
print(f"TOTAL skipped lines across all files: {total_skips}")

print("="*70)
print("COMPLETENESS - regeneration files, POOLED over both seeds, signal ON")
TASKS = ['PickCube', 'StackCube', 'PegInsertionSide', 'PickSingleYCB']
regen_total_sig = 0
for task in TASKS:
    pooled = []
    for seed in [42, 142]:
        fn = f'metrics_{task}-v1_170m_seed{seed}.jsonl'
        rows = load(fn)
        pooled += sig_step(rows)
    n = len(pooled)
    regen_total_sig += n
    v = [r['vision_err'] for r in pooled]
    l = [r['lang_err'] for r in pooled]
    s = [r['state_err'] for r in pooled]
    print(f"  {task}: n={n}")
    print(f"    vision: med%={med(v)*100:.4f}  %<=3={pct_le(v):.4f}")
    print(f"    lang:   med%={med(l)*100:.4f}  %<=3={pct_le(l):.4f}")
    print(f"    state:  med%={med(s)*100:.4f}  %<=3={pct_le(s):.4f}")
print(f"TOTAL signal-bearing n across 4 regen files (both seeds): {regen_total_sig}")

print("="*70)
print("1B step files: with and without signal filter, per seed")
for fn in sorted(glob.glob(os.path.join(DATA, 'metrics_PickCube-v1_1b_seed*.jsonl'))):
    base = os.path.basename(fn)
    rows = load(base)
    allstep = [r for r in rows if r.get('event') == 'step']
    if not allstep:
        continue
    sig = sig_step(rows)
    ends = [r for r in rows if r.get('event') == 'episode_end']
    succ = sum(1 for r in ends if r.get('success'))
    print(f"  {base}: episodes={len(ends)} successes={succ}")
    print(f"    with signal filter: n={len(sig)}")
    print(f"    NO filter (all step): n={len(allstep)}")
    print(f"      vision med%={med([r['vision_err'] for r in allstep])*100:.4f}  lang med%={med([r['lang_err'] for r in allstep])*100:.4f}")
    rn = [r['ref_norm_maniskill'] for r in allstep]
    print(f"      ref_norm range: min={min(rn):.3f} max={max(rn):.3f}  #>=15={sum(1 for x in rn if x>=15)} (signal filter inapplicable at 1B if 0)")

print("="*70)
print("m=128 files: PER FILE (no pooling), signal ON")
for task in ['PegInsertionSide', 'PickSingleYCB']:
    for seed in [42, 142]:
        fn = f'metrics_{task}-v1_170m_seed{seed}_m128.jsonl'
        rows = load(fn)
        sig = sig_step(rows)
        n = len(sig)
        v = [r['vision_err'] for r in sig]
        l = [r['lang_err'] for r in sig]
        print(f"  {task} seed{seed}: n={n}  vis_med%={med(v)*100:.4f}  vis%<=3={pct_le(v):.4f}  lang%<=3={pct_le(l):.4f}")
        # check duplicate episodes
        eps = [(r['episode'], r['policy_call_idx']) for r in rows if r.get('event')=='step']
        dup = len(eps) - len(set(eps))
        if dup:
            print(f"    DUPLICATE (episode,call) pairs: {dup}")

print("="*70)
print("TARGET ABLATION - PickCube, pooled seed42+142, signal ON")
print("  reporting median absolute vision_gap, vision %<=3 (on vision_err), n")
def target_block(suffix, label):
    pooled = []
    for seed in [42, 142]:
        fn = f'metrics_PickCube-v1_170m_seed{seed}{suffix}.jsonl'
        rows = load(fn)
        pooled += sig_step(rows)
    n = len(pooled)
    gaps = [abs(r['vision_gap']) for r in pooled]
    verr = [r['vision_err'] for r in pooled]
    print(f"  {label}: n={n}  median|vision_gap|={med(gaps):.6f}  vision%<=3={pct_le(verr):.4f}")
    # duplicate check
    for seed in [42,142]:
        fn = f'metrics_PickCube-v1_170m_seed{seed}{suffix}.jsonl'
        rows = load(fn)
        eps = [(r['episode'], r['policy_call_idx']) for r in rows if r.get('event')=='step']
        dup = len(eps)-len(set(eps))
        if dup: print(f"      seed{seed} duplicate (ep,call): {dup}")
target_block('', 'logpi')
target_block('_l2', 'l2')
target_block('_maxdev', 'maxdev')

print("="*70)
print("FAITHFULNESS groups")
FAITH_FIELDS = ['vision_insertion_auc','vision_deletion_auc','vision_dlogp_k5',
                'lang_insertion_auc','lang_deletion_auc','lang_dlogp_k5']
def faith_group(patterns, label):
    rows = []
    files = []
    for pat in patterns:
        for fn in sorted(glob.glob(os.path.join(DATA, pat))):
            files.append(os.path.basename(fn))
            rows += load(fn)
    fa = [r for r in rows if r.get('event')=='faithfulness']
    n = len(fa)
    print(f"  {label}: n={n}  files={files}")
    for fld in FAITH_FIELDS:
        print(f"    {fld}: median={med([r.get(fld) for r in fa]):.6f}")
    return fa

for _s in (42, 142, 242):
    if glob.glob(os.path.join(DATA, f'metrics_faithfulness_metrics_PickCube-v1_1b_seed{_s}_*.jsonl')):
        faith_group([f'metrics_faithfulness_metrics_PickCube-v1_1b_seed{_s}_*.jsonl'], f'1B_s{_s}')
faith_group(['metrics_faithfulness_metrics_PickCube-v1_170m_seed*_l2_*.jsonl'], 'l2')
faith_group(['metrics_faithfulness_metrics_PickCube-v1_170m_seed*_maxdev_*.jsonl'], 'maxdev')
faith_group(['metrics_faithfulness_metrics_PegInsertionSide-v1_170m_seed*_m128_*.jsonl'], 'm128_PegInsertionSide')
faith_group(['metrics_faithfulness_metrics_PickSingleYCB-v1_170m_seed*_m128_*.jsonl'], 'm128_PickSingleYCB')

print("="*70)
print("INDEPENDENT AUC RECOMPUTE - 1B files, per seed")
kgrid = np.array([0,1,5,10,20,30,50,75,100], dtype=float) / 100.0  # fractions 0..1
def recompute_1b(pattern, label):
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    if not files:
        return
    fa = []
    for fn in files:
        fa += [r for r in load(fn) if r.get('event')=='faithfulness']
    my_ins, my_del, stored_ins, stored_del = [], [], [], []
    for r in fa:
        denom = r['vision_f_input'] - r['vision_f_baseline']
        if denom == 0:
            continue
        ic = (np.array(r['vision_insertion_curve'], dtype=float) - r['vision_f_baseline']) / denom
        dc = (np.array(r['vision_deletion_curve'], dtype=float) - r['vision_f_baseline']) / denom
        my_ins.append(_trapezoid(ic, kgrid)); my_del.append(_trapezoid(dc, kgrid))
        stored_ins.append(r['vision_insertion_auc']); stored_del.append(r['vision_deletion_auc'])
    if not my_ins:
        print(f"  {label}: no nonzero-denom rows"); return
    di = [abs(a-b) for a,b in zip(my_ins, stored_ins)]
    dd = [abs(a-b) for a,b in zip(my_del, stored_del)]
    print(f"  {label}: n={len(my_ins)} of {len(fa)}  "
          f"ins recompute={med(my_ins):.6f} stored={med(stored_ins):.6f} maxdiff={max(di):.6f}  "
          f"del recompute={med(my_del):.6f} stored={med(stored_del):.6f} maxdiff={max(dd):.6f}")
# Seed 42 keeps its exact committed filename so the audit reproduces the published number;
# 142/242 match by glob once present.
recompute_1b('metrics_faithfulness_metrics_PickCube-v1_1b_seed42_*.jsonl', '1B seed42')
recompute_1b('metrics_faithfulness_metrics_PickCube-v1_1b_seed142_*.jsonl', '1B seed142')
recompute_1b('metrics_faithfulness_metrics_PickCube-v1_1b_seed242_*.jsonl', '1B seed242')

print("="*70)
print("SANITY - frozen and cascade, pooled over 8 files")
for kind in ['frozen','cascade']:
    rows = []
    files = sorted(glob.glob(os.path.join(DATA, f'metrics_sanity_C1_{kind}_*.jsonl')))
    for fn in files:
        rows += load(fn)
    n = len(rows)
    print(f"  {kind}: n={n}  ({len(files)} files)")
    for fld in ['spearman_vision','spearman_language','spearman_state']:
        label = fld.replace('spearman_','rho_')
        print(f"    {label}: median={med([r.get(fld) for r in rows]):.6f}")
# frozen rho_language excluding PickSingleYCB
rows = []
for fn in sorted(glob.glob(os.path.join(DATA, 'metrics_sanity_C1_frozen_*.jsonl'))):
    if 'PickSingleYCB' in fn:
        continue
    rows += load(fn)
print(f"  frozen rho_language EXCL PickSingleYCB: n={len(rows)} median={med([r['spearman_language'] for r in rows]):.6f}")

print("="*70)
print("BASELINE SENSITIVITY - per task")
for task in ['PickCube','StackCube']:
    files = sorted(glob.glob(os.path.join(DATA, f'metrics_baseline_sensitivity_metrics_{task}-v1_*.jsonl')))
    rows = []
    for fn in files:
        rows += load(fn)
    n = len(rows)
    print(f"  {task}: n={n}")
    for fld in ['rho_black_gray','rho_black_blur','rho_gray_blur']:
        print(f"    {fld}: median={med([r.get(fld) for r in rows]):.6f}")

print("="*70)
print("SIDECAR PROVENANCE FINGERPRINT")
print("  The step row records vision_gap = F(x) - F(x') at IG time on the run's")
print("  own observation and noise seed, and F(x) = 0 by construction, so the")
print("  faithfulness row's vision_f_baseline must equal -vision_gap exactly when")
print("  the faithfulness stage loaded that row's own sidecar. Disagreement means")
print("  the stage consumed a sidecar written by a different run under the shared")
print("  pre-fix filename. EXPECTED MISMATCH (documented in the paper and in")
print("  data/README.md): the two seed-42 m=128 faithfulness files, which loaded")
print("  the seed-142 m=128 sidecars that had overwritten their own.")

def step_file_for(faith_base):
    """Map a faithfulness filename to its base step filename, or None."""
    #m5_/m7_ prefixed runs: {p}_metrics_faithfulness_{rest}.jsonl -> {p}_metrics_{rest}.jsonl
    for p in ('m5_', 'm7_'):
        if faith_base.startswith(p + 'metrics_faithfulness_'):
            return p + 'metrics_' + faith_base[len(p + 'metrics_faithfulness_'):]
    #main convention: metrics_faithfulness_{step base}_{YYYYMMDD_HHMMSS}.jsonl
    if faith_base.startswith('metrics_faithfulness_'):
        stem = faith_base[len('metrics_faithfulness_'):-len('.jsonl')]
        parts = stem.rsplit('_', 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            stem = parts[0]
        return stem + '.jsonl'
    return None

for fn in sorted(os.listdir(DATA)):
    if 'faithfulness' not in fn or not fn.endswith('.jsonl'):
        continue
    step_fn = step_file_for(fn)
    if step_fn is None or not os.path.exists(os.path.join(DATA, step_fn)):
        print(f"  {fn}: no matching step file, skipped")
        continue
    steps = {}
    for r in load(step_fn):
        if r.get('event') == 'step':
            steps[(r['episode'], r['policy_call_idx'])] = r
    fa = [r for r in load(fn) if r.get('event') == 'faithfulness']
    diffs = []
    for r in fa:
        k = (r['episode'], r['policy_call_idx'])
        if k not in steps:
            continue
        g = -steps[k]['vision_gap']
        fb = r['vision_f_baseline']
        denom = max(abs(g), abs(fb), 1e-12)
        diffs.append(abs(g - fb) / denom)
    if not diffs:
        print(f"  {fn}: no joinable rows")
        continue
    a = np.array(diffs)
    pct_ok = float(np.mean(a < 0.02) * 100.0)
    verdict = 'OWN SIDECARS' if pct_ok > 99.0 else 'FOREIGN SIDECARS'
    print(f"  {fn}:")
    print(f"    rows={len(a)}  %within2%={pct_ok:.1f}  median_reldiff={float(np.median(a)):.4f}  -> {verdict}")
