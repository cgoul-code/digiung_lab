#!/bin/bash
# set -e

# echo "==> Python being used: $(which python)"
# echo "==> Checking Hypercorn in current env"
# python -m pip show hypercorn || echo "!! Hypercorn NOT installed"

# python -m hypercorn app:app --bind 0.0.0.0:${PORT:-8000}

# if [ ! -d "antenv" ]; then
#     python3 -m venv antenv
#     source antenv/bin/activate
#     pip install --no-cache-dir -r requirements.txt
# else
#     source antenv/bin/activate
# fi

# python -m hypercorn app:app --bind 0.0.0.0:${PORT:-8000}

#!/bin/bash
# set -e

# cd /home/site/wwwroot

# echo "===> Current dir: $(pwd)"
# echo "===> Files here:"
# ls

# if [ ! -d "antenv" ]; then
#     echo "==> antenv not found, creating venv..."
#     python -m venv antenv
# fi

# echo "===> Activating antenv"
# source antenv/bin/activate

# echo "==> Python being used: $(which python)"
# echo "==> Installing / updating dependencies in antenv..."
# python -m pip install --upgrade pip
# pip install --no-cache-dir -r requirements.txt

# echo "==> Forcing Hypercorn install into venv..."
# pip uninstall hypercorn
# pip install hypercorn==0.17.3

# echo "===> Hypercorn CLI version:"
# hypercorn --version || echo "!! Hypercorn CLI still not found"

# echo "==> Starting Hypercorn..."
# exec hypercorn app:app --bind 0.0.0.0:${PORT:-8000}



if [ ! -d "antenv" ]; then
    python3 -m venv antenv
    source antenv/bin/activate
    pip install --no-cache-dir -r requirements.txt
else
    source antenv/bin/activate
fi

hypercorn query_server:app --bind 0.0.0.0:8000 --workers 1
