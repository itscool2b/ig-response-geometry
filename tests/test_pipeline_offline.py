import inspect
from types import SimpleNamespace

import pipeline


class RecordingHub:
    def __init__(self):
        self.calls = []

    def model_info(self, repo_id, revision=None):
        self.calls.append((repo_id, revision))
        return SimpleNamespace(sha="f" * 40)


def test_a_full_commit_id_needs_no_hub_call():
    hub = RecordingHub()
    sha = "8aa386cac3bbfd9540676c75b3d767cc7f88a10a"
    assert pipeline.resolve_revision(hub, "robotics-diffusion-transformer/rdt-170m", sha) == sha
    assert hub.calls == []


def test_names_short_ids_and_uppercase_ids_still_resolve_through_the_hub():
    hub = RecordingHub()
    for revision in (None, "main", "8aa386c", "8AA386CAC3BBFD9540676C75B3D767CC7F88A10A", "8aa386cac3bbfd9540676c75b3d767cc7f88a10a\n"):
        assert pipeline.resolve_revision(hub, "repo", revision) == "f" * 40
    assert len(hub.calls) == 5


def test_load_pipeline_resolves_both_revisions_through_the_helper():
    source = inspect.getsource(pipeline.load_pipeline)
    assert "model_info" not in source
    assert source.count("resolve_revision(hub, ") == 2


def test_the_170m_config_path_does_not_depend_on_the_working_directory():
    assert pipeline.CONFIG_170M.is_absolute() and pipeline.CONFIG_170M.is_file()
    assert "str(CONFIG_170M)" in inspect.getsource(pipeline.load_pipeline)
