"""Deletion failures must not look like missing keys or successful pops."""

from io import BytesIO
from zipfile import ZipFile

import pytest

from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.zip import ZipFileSystem
from fsspec.mapping import FSMap


def delete(mapper):
    del mapper["data"]


@pytest.mark.parametrize(
    "operation",
    [
        delete,
        lambda m: m.pop("data"),
        lambda m: m.pop("data", None),
        lambda m: m.popitem(),
    ],
)
def test_readonly_archive_deletion_failure(operation):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("data", b"original")
    buffer.seek(0)
    fs = ZipFileSystem(buffer)
    try:
        mapper = FSMap("", fs)
        # Independent backend control: deleting this member is unsupported.
        with pytest.raises(NotImplementedError):
            fs.rm("data")
        with pytest.raises(NotImplementedError):
            operation(mapper)
        assert mapper["data"] == b"original"
        assert list(mapper) == ["data"]
    finally:
        fs.close()


@pytest.mark.parametrize(
    "error_type",
    [PermissionError, TimeoutError, OSError, NotImplementedError, ValueError],
)
@pytest.mark.parametrize(
    "operation",
    [
        delete,
        lambda m: m.pop("data"),
        lambda m: m.pop("data", b"default"),
        lambda m: m.popitem(),
    ],
)
def test_mapper_deletion_preserves_backend_errors(monkeypatch, error_type, operation):
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = FSMap("root", fs)
    mapper["data"] = b"original"
    error = error_type("deletion failed")

    def fail_rm(path, *args, **kwargs):
        raise error

    monkeypatch.setattr(fs, "rm", fail_rm)
    with pytest.raises(error_type) as caught:
        operation(mapper)
    assert caught.value is error
    assert mapper["data"] == b"original"


@pytest.mark.parametrize(
    "error_type", [FileNotFoundError, IsADirectoryError, NotADirectoryError]
)
def test_mapper_delete_missing_exception(monkeypatch, error_type):
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = FSMap("root", fs)
    error = error_type("missing")

    def fail_rm(path, *args, **kwargs):
        raise error

    monkeypatch.setattr(fs, "rm", fail_rm)
    with pytest.raises(KeyError) as caught:
        del mapper["data"]
    assert caught.value.args == ("data",)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("missing_exceptions", [(), (ValueError,)])
def test_mapper_delete_unconfigured_missing_error(monkeypatch, missing_exceptions):
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = FSMap("root", fs, missing_exceptions=missing_exceptions)
    error = FileNotFoundError("not configured as a missing key")

    def fail_rm(path, *args, **kwargs):
        raise error

    monkeypatch.setattr(fs, "rm", fail_rm)
    with pytest.raises(FileNotFoundError) as caught:
        del mapper["data"]
    assert caught.value is error


@pytest.mark.parametrize("default", [None, b"default"])
def test_mapper_pop_preserves_missing_after_read_race(monkeypatch, default):
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = FSMap("root", fs)
    mapper["data"] = b"original"
    rm = fs.rm

    def remove_then_raise(path, *args, **kwargs):
        rm(path, *args, **kwargs)
        raise FileNotFoundError(path)

    monkeypatch.setattr(fs, "rm", remove_then_raise)
    assert mapper.pop("data", default) == b"original"
    assert "data" not in mapper


def test_mapper_delete_custom_missing_exception(monkeypatch):
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = FSMap("root", fs, missing_exceptions=(ValueError,))
    error = ValueError("missing")

    def fail_rm(path, *args, **kwargs):
        raise error

    monkeypatch.setattr(fs, "rm", fail_rm)
    with pytest.raises(KeyError) as caught:
        del mapper["data"]
    assert caught.value.args == ("data",)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("protocol", ["memory", "file"])
def test_mapper_delete_and_pop_controls(tmp_path, protocol):
    if protocol == "memory":
        fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
        mapper = FSMap("root", fs)
    else:
        from fsspec.implementations.local import LocalFileSystem

        mapper = FSMap(str(tmp_path), LocalFileSystem())
    mapper["data"] = b"value"
    assert mapper.pop("data") == b"value"
    assert "data" not in mapper
    assert mapper.pop("data", None) is None
    with pytest.raises(KeyError):
        del mapper["data"]
    mapper["data"] = b"value"
    del mapper["data"]
    assert "data" not in mapper
