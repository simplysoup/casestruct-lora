from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_single_readme():
    assert sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("README*.md")) == ["README.md"]


def test_single_root_launcher():
    assert (ROOT / "run.sh").is_file()
    assert not (ROOT / "run_v2.sh").exists()
    assert not (ROOT / "run_v21.sh").exists()


def test_audit_writer_available():
    from casestruct.v21.audit import write_markdown
    assert callable(write_markdown)
