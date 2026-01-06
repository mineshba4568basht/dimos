set -euo pipefail
/usr/bin/python3.12 -V
/usr/bin/python3.12 -m venv /home/jalaj/dimensional/dimos/.venv
. /home/jalaj/dimensional/dimos/.venv/bin/activate
python -m pip install -U pip setuptools wheel
python --version
python -c "import sys; print(sys.executable)"
