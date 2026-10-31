"""Replay acceptance must reject changed evidence, not hide it behind tolerances."""
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import pytest


def certificate():
    path = Path(__file__).resolve().parents[1] / "scripts/reviewer_interval_certificate.py"
    spec = importlib.util.spec_from_file_location("certificate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitivity_enclosure_is_narrow_and_rejects_outside():
    np = pytest.importorskip("numpy")
    module = certificate()
    values = np.array([.1, .13, .28, .37, .52, .81])
    rng = np.random.Generator(np.random.PCG64(19))
    samples = values[rng.integers(0, len(values), size=(1000, len(values)))].mean(axis=1)
    samples += 1e-8
    result = module.enclose(values, samples, .05, kinds=("bca", "t"))
    for kind in ("bca", "t"):
        for item in result[kind]:
            low, high = map(float, item["enclosure"])
            assert high - low < 1e-10
            assert module.contains(item, (low + high) / 2)
            assert not module.contains(item, math.nextafter(low, -math.inf))
            assert not module.contains(item, math.nextafter(high, math.inf))


@pytest.mark.parametrize("values,samples", [([1., 1., 1.], [1.] * 100),
                                           ([0., 1., 2.], [1.] * 100),
                                           ([1., 1., math.nextafter(1., 2.)], [1.] * 100),
                                           ([1., math.inf, 3.], [1.] * 100)])
def test_sensitivity_uncertain_tie_variance_and_nonfinite_fail_closed(values, samples):
    with pytest.raises(ValueError):
        certificate().enclose(values, samples, .05)


def test_subnormal_intermediate_domain_is_rejected_explicitly():
    np = pytest.importorskip("numpy")
    values = np.array([.1, .13, .28, .37, .52, .81])*1e-160
    with pytest.raises(ValueError, match="normal range"):
        certificate().enclose(values, values, .05, kinds=("t",))


@pytest.mark.parametrize("degrees", [1, 2, 5, 39])
def test_independent_integer_t_quantile_brackets_current_scipy(degrees):
    from scipy.stats import t
    module = certificate()
    for probability in (.975, 1-.05/14, 1-.0125/2):
        checks = []
        box = module.contract(module.t_quantile(probability, degrees), t.ppf(probability, degrees), "student_t_ppf", checks)
        assert module.contains(module.packed(box), t.ppf(probability, degrees))
    if degrees == 1:
        exact = 1/module.mp.tan(module.mp.pi*(1-module.mp.mpf(.975)))
        assert module.contains(module.packed(module.t_quantile(.975, 1)), exact)


def test_pointwise_accuracy_contract_rejects_unbounded_library_error():
    module = certificate()
    with pytest.raises(ValueError, match="accuracy contract"):
        module.contract(module.iv(1), 1+32*2**-53, "normal_ppf", [])
    with pytest.raises(ValueError, match="accuracy contract"):
        module.contract(module.iv(1), 1+64*2**-53, "normal_cdf", [], module.CDF_ACCURACY_UNITS)


def test_interval_endpoint_serialization_preserves_exact_binary_values():
    from decimal import Decimal, localcontext
    from fractions import Fraction
    module = certificate()
    values = [module.mp.mpf(0), module.mp.mpf(1), module.mp.mpf(-1),
              module.mp.mpf(1)/3, -module.mp.mpf(1)/7,
              module.mp.mpf(2)**-1000, module.mp.mpf(2)**1000]
    for value in values:
        sign, mantissa, exponent, _ = value._mpf_
        exact = Fraction((-1 if sign else 1)*mantissa) * Fraction(2)**exponent
        with localcontext() as ctx:
            ctx.prec = 3
            encoded = module.exact_decimal(value)
        assert Fraction(Decimal(encoded)) == exact
        assert module.mp.mpf(encoded)._mpf_ == value._mpf_
    for value in (module.mp.inf, -module.mp.inf, module.mp.nan):
        with pytest.raises(ValueError, match="nonfinite"):
            module.exact_decimal(value)


def test_quantile_closed_endpoints_and_crossed_bins_are_enclosed():
    np = pytest.importorskip("numpy")
    module = certificate()
    samples = np.array([-1., -.2, .3, .7, 2.])
    for p in (0., 1., .25, .5, .75):
        bounds, _ = module.quantile_enclosure(samples, module.iv(p))
        assert module.contains(module.packed(bounds), np.quantile(samples, p))
    bounds, bins = module.quantile_enclosure(samples, module.iv(.25-1e-15, .25+1e-15))
    assert bins == [0, 1]
    for p in (.25-1e-15, .25, .25+1e-15):
        assert module.contains(module.packed(bounds), np.quantile(samples, p))
    assert module.contains(module.packed(module.t_quantile(.5, 1)), 0.)


def test_preserved_primary_mean_is_required_for_tie_fallback():
    np = pytest.importorskip("numpy")
    module = certificate()
    values = np.array([.2, .8])
    samples = np.array([.2]*25+[.5]*50+[.8]*25)
    with pytest.raises(ValueError, match="ties"):
        module.enclose(values, samples, .05, kinds=("bca",))
    proof = module.enclose(values, samples, .05, kinds=("bca",), authenticated_mean=.5)
    assert proof["samples_equal_mean"] == 50 and proof["bias_mean_bound_exactly"]
    with pytest.raises(ValueError, match="historical mean"):
        module.enclose(values, samples, .05, kinds=("bca",), authenticated_mean=math.nextafter(.5, 1.))


def test_manifest_bound_helper_ignores_ambient_module_and_checks_supplied_bytes(tmp_path, monkeypatch):
    module = verifier()
    archive = tmp_path / "archive"
    archive.mkdir()
    helper = archive / "reviewer_interval_certificate.py"
    blob = b"answer = 42\n"
    helper.write_bytes(blob)
    (tmp_path / "reviewer_interval_certificate.py").write_text("raise AssertionError('ambient import')")
    monkeypatch.syspath_prepend(str(tmp_path))
    index = {helper.name: sha(blob)}
    loaded, _, source = module._interval_helper(archive, index, (helper, blob))
    assert loaded.answer == 42 and source == blob
    with pytest.raises(ValueError, match="supplied"):
        module._interval_helper(archive, index, (helper, b"answer = 99\n"))
    helper.write_bytes(b"answer = 41\n")
    with pytest.raises(ValueError, match="hash"):
        module._interval_helper(archive, index)


def test_helper_source_change_during_import_is_rejected(tmp_path):
    helper = tmp_path / "reviewer_interval_certificate.py"
    blob = b"from pathlib import Path\nPath(__file__).write_bytes(b'changed')\n"
    helper.write_bytes(blob)
    with pytest.raises(ValueError, match="changed during import"):
        verifier()._interval_helper(tmp_path, {helper.name: sha(blob)})


def test_interval_exception_does_not_cover_primary_t_or_decision_fields():
    from collections import Counter
    report = dict(files={"fixture": {"numeric_differences": []}}, checks=Counter(), failures=[])
    compare = verifier()._Compare(report, "fixture", 8, {}, {"/sensitivity/bca/interval/0": {"enclosure": ["0.09", "0.11"]}})
    compare.numeric(.1, math.nextafter(.1, 1.), "/primary/interval/0")
    compare.numeric(.1, math.nextafter(.1, 1.), "/sensitivity/t/interval/0")
    compare.value(True, False, "/unique")
    assert len(report["failures"]) == 3


def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_reviewer_replay.py"
    assert path.is_file(), "The independent reviewer replay verifier must exist"
    spec = importlib.util.spec_from_file_location("reviewer_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csv_blob(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    torch = pytest.importorskip("torch")
    archive, actual, code = tmp_path / "archive", tmp_path / "actual", tmp_path / "code"
    expected = archive / "results/analysis"
    expected.mkdir(parents=True)
    actual.mkdir()
    code.mkdir()
    sources = {name: ("# preserved " + name + "\n").encode() for name in
               ["ig_quadrature.py", "analysis/revision_gpu/v1/analyze.py",
                "analysis/revision_gpu/v1/numerics.py", "analysis/revision_gpu/v1/methods.py",
                "analysis/revision_gpu/v1/coverage.py"]}
    source_manifest = {"members": {name: {"sha256": sha(blob), "bytes": len(blob)}
                                   for name, blob in sources.items()}}
    (archive / "code").mkdir()
    with zipfile.ZipFile(archive / "code/research-supplement.zip", "w") as z:
        z.writestr("SUPPLEMENT_MANIFEST.json", json.dumps(source_manifest))
        for name, blob in sources.items():
            z.writestr(name, blob)
            (code / name).parent.mkdir(parents=True, exist_ok=True)
            (code / name).write_bytes(blob)
    save_json(code / "SUPPLEMENT_MANIFEST.json", source_manifest)
    map_file = archive / "runs/P/e0000-c000/fp32/maps.pt"
    map_file.parent.mkdir(parents=True)
    torch.save({"vision/Q/A2T512": torch.ones((1, 8, 4), dtype=torch.float32)}, map_file)
    refs = [dict(bank="P", group="P", context="context0", episode=0, call=0,
                 modality="vision", target="Q", family="fp32_vs_A2T512", reference="A2T512",
                 candidate="A2T64", source="recomputed", l1=0.1, relative_l1=0.02,
                 signed_sum_difference=1e-16, max_abs=0.02, top1_equal="True", top5pct_size=1)]
    summary = [dict(group="P", modality="vision", target="Q", family="fp32_vs_A2T512",
                    reference="A2T512", candidate="A2T64", n=1, median=0.02,
                    mean=0.02, max=0.02, min=0.02, over_0_01=1)]
    methods = dict(best="IG_Q", tie=[], unique=True, preferred="IG_Q", episodes=4,
                   probability_best=0.75, adjusted=dict(interval=[0.1, 0.2],
                   sensitivity=dict(bca=dict(interval=[0.09, 0.21]))), coverage=0.95)
    files = {"numerics/references.csv": csv_blob(refs),
             "numerics/comparisons.csv": b"bank,group,modality,target,family,reference,candidate,relative_l1\n",
             "numerics/comparison_summary.csv": csv_blob(summary),
             "methods/P/decisions.json": json.dumps(methods).encode(),
             "shifts/P/rows_vision.csv": b"episode,success,delta\n0,False,0.2\n"}
    for root in [expected, actual]:
        for name, blob in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(blob)
        save_json(root / "manifest.json", dict(kind="nhsjs_r1_gpu_analysis", outputs={n: sha(b) for n,b in files.items()},
                 population={"P": {"available": 1}}, plan_sha256="a" * 64,
                 coverage_sha256={"coverage.json": "b" * 64}))
    save_json(archive / "reproduction.json", {"code": "code/research-supplement.zip"})

    def bind_archive():
        entries = sorted((p.relative_to(archive).as_posix(), sha(p.read_bytes()))
                         for p in archive.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
        (archive / "MANIFEST.sha256").write_text("".join(d + "  " + n + "\n" for n,d in entries), encoding="utf-8")

    def mutate(name, transform):
        path = actual / name
        path.write_bytes(transform(path.read_bytes()))
        manifest = json.loads((actual / "manifest.json").read_bytes())
        manifest["outputs"][name] = sha(path.read_bytes())
        save_json(actual / "manifest.json", manifest)

    bind_archive()
    return dict(archive=archive, expected=expected, actual=actual, code=code,
                mutate=mutate, bind_archive=bind_archive, refs=refs, summary=summary)


def run(case):
    return verifier().verify(case["expected"], case["actual"], case["archive"], case["code"])


def test_direct_cli_and_immutable_launcher_loaded_module_agree(case, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_reviewer_replay.py"
    report = tmp_path / "cli-report.json"
    command = [sys.executable, str(path), "--expected-results", str(case["expected"]),
               "--actual-results", str(case["actual"]), "--archive-root", str(case["archive"]),
               "--code-root", str(case["code"]), "--report", str(report)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    namespace = types.ModuleType("immutable_launcher_verifier")
    namespace.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), namespace.__dict__)
    loaded = namespace.verify(case["expected"], case["actual"], case["archive"], case["code"])
    assert loaded == json.loads(report.read_text())


def test_exact_replay_accepts_without_modifying_any_file(case):
    before = {p: sha(p.read_bytes()) for p in case["archive"].rglob("*") if p.is_file()}
    result = run(case)
    assert result["verified"] and result["byte_identity"]
    assert before == {p: sha(p.read_bytes()) for p in before}
    assert result["map_metadata"]["maximum_coordinates"] == 32
    assert str(case["archive"].parent) not in json.dumps(result)


def test_valid_reduction_difference_keeps_hash_mismatch_and_exact_ulp(case):
    case["refs"][0]["l1"] = math.nextafter(0.1, math.inf)
    case["mutate"]("numerics/references.csv", lambda _: csv_blob(case["refs"]))
    result = run(case)
    assert result["verified"] and not result["byte_identity"]
    changes = result["files"]["numerics/references.csv"]["numeric_differences"]
    assert len(changes) == 1 and changes[0]["ulp_distance"] == 1
    assert changes[0]["absolute_error"] == 1.3877787807814457e-17
    assert changes[0]["accepted"]


def test_signed_cancellation_uses_norm_scaled_bound_not_ulp_cutoff(case):
    case["refs"][0]["signed_sum_difference"] = 1e-16 + 1e-24
    case["mutate"]("numerics/references.csv", lambda _: csv_blob(case["refs"]))
    result = run(case)
    assert result["verified"]
    assert result["files"]["numerics/references.csv"]["numeric_differences"][0]["ulp_distance"] > 1000


def test_map_roundoff_bound_rejects_observed_scale_of_windows_anomaly(case):
    case["refs"][0]["l1"] += 2**-32
    case["mutate"]("numerics/references.csv", lambda _: csv_blob(case["refs"]))
    result = run(case)
    assert not result["verified"]
    assert any("bound" in failure["reason"] for failure in result["failures"])


def test_worker_record_changes_are_not_recomputed_map_roundoff(case):
    for root in [case["expected"], case["actual"]]:
        path = root / "numerics/references.csv"
        path.write_bytes(path.read_bytes().replace(b"recomputed", b"worker"))
        manifest = json.loads((root / "manifest.json").read_bytes())
        manifest["outputs"]["numerics/references.csv"] = sha(path.read_bytes())
        save_json(root / "manifest.json", manifest)
    case["bind_archive"]()
    case["mutate"]("numerics/references.csv", lambda b: b.replace(b",0.1,", b",0.10000000000000002,"))
    assert not run(case)["verified"]


@pytest.mark.parametrize("old,new", [(b"False",b""),(b"False",b"True"),(b"0.2",b"0.20000000000000004")])
def test_missing_outcome_or_unknown_numeric_difference_is_rejected(case,old,new):
    case["mutate"]("shifts/P/rows_vision.csv", lambda b: b.replace(old,new))
    assert not run(case)["verified"]


@pytest.mark.parametrize("field,value", [("best","PG_N"),("tie",["IG_Q","PG_N"]),("unique",False),
                                         ("episodes",5),("probability_best",0.7500000000000001)])
def test_method_ties_preference_counts_and_shares_are_exact(case,field,value):
    def change(blob):
        data=json.loads(blob);data[field]=value;return json.dumps(data).encode()
    case["mutate"]("methods/P/decisions.json",change)
    assert not run(case)["verified"]


@pytest.mark.parametrize("interval", [[-0.1,0.2],[0.10000000000000002,0.2]])
def test_primary_interval_sign_or_low_bit_change_is_rejected(case,interval):
    def change(blob):
        data=json.loads(blob);data["adjusted"]["interval"]=interval;return json.dumps(data).encode()
    case["mutate"]("methods/P/decisions.json",change)
    assert not run(case)["verified"]


def test_summary_must_derive_from_its_own_input_values(case):
    case["summary"][0]["mean"] = 0.01000000000001
    case["mutate"]("numerics/comparison_summary.csv",lambda _:csv_blob(case["summary"]))
    assert not run(case)["verified"]


def test_bounded_summary_reduction_preserves_hash_difference(case):
    case["summary"][0]["mean"] = math.nextafter(0.02,math.inf)
    case["mutate"]("numerics/comparison_summary.csv",lambda _:csv_blob(case["summary"]))
    result=run(case)
    assert result["verified"] and not result["byte_identity"]


def test_worker_comparisons_may_omit_the_optional_reference_column(case):
    comparison = b"bank,group,modality,target,family,candidate,relative_l1\nP,P,vision,Q,bf16_vs_A1_comparator,E1,0.02\n"
    extra = dict(case["summary"][0], family="bf16_vs_A1_comparator", reference="", candidate="E1")
    for root in (case["expected"], case["actual"]):
        changed = {"numerics/comparisons.csv": comparison,
                   "numerics/comparison_summary.csv": csv_blob(case["summary"] + [extra])}
        manifest = json.loads((root / "manifest.json").read_bytes())
        for name, blob in changed.items():
            (root / name).write_bytes(blob)
            manifest["outputs"][name] = sha(blob)
        save_json(root / "manifest.json", manifest)
    case["bind_archive"]()
    assert run(case)["verified"]


def test_metadata_is_authenticated_before_increasing_any_numeric_bound(case):
    path=case["archive"]/"runs/P/e0000-c000/fp32/maps.pt"
    path.write_bytes(path.read_bytes()+b"altered")
    result=run(case)
    assert not result["verified"]
    assert any("hash" in f["reason"] for f in result["failures"])


def test_preserved_analysis_source_cannot_be_replaced(case):
    (case["code"]/"ig_quadrature.py").write_text("# altered\n")
    assert not run(case)["verified"]


def test_stale_generated_manifest_rejects_even_a_permitted_roundoff(case):
    path=case["actual"]/"numerics/references.csv"
    path.write_bytes(path.read_bytes().replace(b",0.1,",b",0.10000000000000002,"))
    assert not run(case)["verified"]


def test_missing_file_and_extra_file_are_not_complete_replays(case):
    (case["actual"]/"shifts/P/rows_vision.csv").unlink()
    (case["actual"]/"unexpected.txt").write_text("not registered")
    assert not run(case)["verified"]


def test_self_comparison_cannot_claim_independent_replay(case):
    result=verifier().verify(case["expected"],case["expected"],case["archive"],case["code"])
    assert not result["verified"]


def test_population_plan_and_coverage_are_mandatory(case):
    path=case["actual"]/"manifest.json"
    data=json.loads(path.read_bytes());data["population"]={"P":{"available":2}}
    save_json(path,data)
    assert not run(case)["verified"]


@pytest.mark.parametrize("blob", [b'{"value":1e999}', b'{"value":-1e999}', b'{"value":Infinity}'])
def test_json_nonfinite_or_overflow_is_rejected_even_when_both_sides_match(case, blob):
    for root in (case["expected"], case["actual"]):
        name="methods/P/decisions.json"
        (root/name).write_bytes(blob)
        manifest=json.loads((root/'manifest.json').read_bytes())
        manifest['outputs'][name]=sha(blob)
        save_json(root/'manifest.json',manifest)
    case['bind_archive']()
    assert not run(case)['verified']


@pytest.mark.parametrize("marker", [b'nan',b'inf',b'-Infinity'])
def test_unregistered_csv_nonfinite_markers_are_rejected_even_when_equal(case, marker):
    name='shifts/P/rows_vision.csv'
    for root in (case['expected'],case['actual']):
        blob=b'episode,success,delta\n0,False,'+marker+b'\n'
        (root/name).write_bytes(blob)
        manifest=json.loads((root/'manifest.json').read_bytes())
        manifest['outputs'][name]=sha(blob)
        save_json(root/'manifest.json',manifest)
    case['bind_archive']()
    assert not run(case)['verified']


@pytest.mark.parametrize("value", [float('nan'),float('inf')])
def test_nonfinite_map_invalidates_finite_reduction_bound(case,value):
    import torch
    path=case['archive']/'runs/P/e0000-c000/fp32/maps.pt'
    torch.save({'vision/Q/A2T512':torch.tensor([value],dtype=torch.float32)},path)
    case['bind_archive']()
    assert not run(case)['verified']


def make_directory_alias(link,target):
    try:
        os.symlink(target,link,target_is_directory=True)
    except OSError:
        if os.name!='nt':
            pytest.skip('Directory aliases unavailable')
        result=subprocess.run(['cmd','/c','mklink','/J',str(link),str(target)],capture_output=True)
        if result.returncode:
            pytest.skip('Directory aliases unavailable')


def remove_directory_alias(link):
    if link.is_symlink():
        link.unlink()
    elif hasattr(link,'is_junction') and link.is_junction():
        link.rmdir()


def test_root_alias_is_rejected_before_resolve_erases_it(case):
    alias=case['archive'].parent/'linked-archive'
    make_directory_alias(alias,case['archive'])
    try:
        result=verifier().verify(alias/'results/analysis',case['actual'],alias,case['code'])
    finally:
        remove_directory_alias(alias)
    assert not result['verified']
    assert any('alias' in f['reason'] for f in result['failures'])


def test_parent_alias_is_rejected_before_resolve_erases_it(case):
    alias=case['archive'].parent/'linked-parent'
    make_directory_alias(alias,case['archive'].parent)
    try:
        result=verifier().verify(alias/'archive/results/analysis',case['actual'],alias/'archive',case['code'])
    finally:
        remove_directory_alias(alias)
    assert not result['verified']


def test_manifest_file_alias_is_rejected(case):
    original=case['actual']/'manifest.json'
    target=case['actual'].parent/'outside-manifest.json'
    target.write_bytes(original.read_bytes())
    original.unlink()
    try:
        os.symlink(target,original)
    except OSError:
        pytest.skip('File symlinks unavailable')
    assert not run(case)['verified']


def test_signed_zero_change_is_not_silently_equated(case):
    case['refs'][0]['signed_sum_difference']=0.0
    blob=csv_blob(case['refs'])
    name='numerics/references.csv'
    (case['expected']/name).write_bytes(blob)
    manifest=json.loads((case['expected']/'manifest.json').read_bytes())
    manifest['outputs'][name]=sha(blob)
    save_json(case['expected']/'manifest.json',manifest)
    case['bind_archive']()
    case['refs'][0]['signed_sum_difference']=-0.0
    case['mutate'](name,lambda _:csv_blob(case['refs']))
    result=run(case)
    assert not result['verified']
    assert any('signed zero' in f['reason'] for f in result['failures'])


def test_bounded_value_cannot_cross_a_diagnostic_threshold(case):
    case['refs'][0]['relative_l1']=0.01
    blob=csv_blob(case['refs']);name='numerics/references.csv'
    (case['expected']/name).write_bytes(blob)
    manifest=json.loads((case['expected']/'manifest.json').read_bytes())
    manifest['outputs'][name]=sha(blob)
    save_json(case['expected']/'manifest.json',manifest)
    case['bind_archive']()
    case['refs'][0]['relative_l1']=math.nextafter(.01,math.inf)
    case['mutate'](name,lambda _:csv_blob(case['refs']))
    result=run(case)
    assert not result['verified']
    assert any('threshold' in f['reason'] for f in result['failures'])


def test_zero_reference_norm_preserves_missing_ratio_without_zero_imputation():
    left={'source':'recomputed','l1':'0.1','relative_l1':'','signed_sum_difference':'0.1'}
    right=dict(left,signed_sum_difference=str(math.nextafter(.1,math.inf)))
    bound=verifier()._reference_bound('signed_sum_difference',left,right,32)
    assert 1.3877787807814457e-17 < bound < 1e-12
    assert left['relative_l1']==right['relative_l1']==''


def test_map_mutation_between_hashing_and_metadata_read_is_rejected(case,monkeypatch):
    module=verifier()
    authenticate=module._authenticate
    def mutate_after_authentication(root,index,path):
        authenticate(root,index,path)
        if path.name=='maps.pt':
            with path.open('ab') as stream:
                stream.write(b'concurrent change')
    monkeypatch.setattr(module,'_authenticate',mutate_after_authentication)
    result=module.verify(case['expected'],case['actual'],case['archive'],case['code'])
    assert not result['verified']
    assert any('changed' in f['reason'] for f in result['failures'])
