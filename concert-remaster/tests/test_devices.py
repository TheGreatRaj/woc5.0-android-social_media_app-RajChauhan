from concert_remaster import cli


def test_windows_driver_versions_are_read():
    assert cli.nvidia_driver_from_windows("31.0.15.5222") == "552.22"
    assert cli.nvidia_driver_from_windows("30.0.14.7141") == "471.41"


def _problem(**info):
    return cli._why_no_cuda({"nvidia_gpu": "NVIDIA GeForce RTX 3060 Laptop GPU", **info})


def test_a_cpu_only_pytorch_is_explained():
    problem, fix = _problem(torch="2.14.0+cpu", torch_cuda=None, nvidia_driver="560.94")
    assert "no CUDA support" in problem and "RTX 3060" in problem and "setup.bat" in fix


def test_an_old_driver_is_explained():
    problem, fix = _problem(torch="2.14.1+cu126", torch_cuda="12.6", nvidia_driver="471.41")
    assert "too old" in problem and "Update the NVIDIA driver" in fix


def test_other_cuda_failures_pass_on_the_reason():
    problem, fix = _problem(torch="2.14.1+cu126", torch_cuda="12.6", nvidia_driver="560.94",
                            cuda_error="CUDA driver initialization failed")
    assert "initialization failed" in problem and "driver" in fix


def test_devices_require_cuda_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(cli, "describe_devices", lambda: {"cuda": False, "torch_cuda": None, "problem": "p", "fix": "f"})
    assert cli.main(["devices", "--require", "cuda"]) == 3
    monkeypatch.setattr(cli, "describe_devices", lambda: {"cuda": False, "torch_cuda": "12.6"})
    assert cli.main(["devices", "--require", "cuda"]) == 4
    monkeypatch.setattr(cli, "describe_devices", lambda: {"cuda": True, "torch_cuda": "12.6"})
    assert cli.main(["devices", "--require", "cuda"]) == 0
    assert "Fix: f" in capsys.readouterr().out
