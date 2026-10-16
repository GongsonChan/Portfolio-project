import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

# force UTF-8 for child processes' stdout/stderr — without this, Python falls back to the
# Windows console codepage (cp949 on Korean Windows) when output is piped/redirected (e.g. to a
# log file), which crashes on any Korean text or special punctuation like "—". BUT this somehow
# breaks yfinance's crumb/cookie SSL handshake entirely (confirmed by reproduction: the exact
# same yf.download() call that succeeds with a plain env fails every time with PYTHONUTF8=1 set
# -- not a Yahoo-side block, a side effect of this one env var on urllib3/ssl). So 01_data_load
# (the only module touching yfinance) runs with a plain environment instead.
_CHILD_ENV = os.environ.copy()
_CHILD_ENV["PYTHONUTF8"] = "1"
_CHILD_ENV_PLAIN = os.environ.copy()


def run_steps(steps: list[tuple[str, str]]) -> None:
    """steps: list of (module_dir, script_path_relative_to_module_dir)."""
    for module, script in steps:
        module_dir = ROOT / module
        script_path = module_dir / script
        print(f"\n=== {module}/{script} ===")
        env = _CHILD_ENV_PLAIN if module == "01_data_load" else _CHILD_ENV
        result = subprocess.run([str(PYTHON), str(script_path)], cwd=str(module_dir), env=env)
        if result.returncode != 0:
            print(f"\nFAILED: {module}/{script} (exit {result.returncode}) - 중단합니다.")
            sys.exit(1)
    print("\n전체 완료.")
