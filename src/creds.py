import json
from pathlib import Path

ROOT_DIR: Path = Path.cwd()
CRED_DIR: Path = ROOT_DIR / "src"

data = json.load(open(CRED_DIR / 'creds.json'))

with open("../.env", "w") as f:
    for key, value in data.items():
        f.write(f"{key.upper()}={value}\n")