"""Load only an explicitly selected CareGuide configuration without printing it."""
import os
from pathlib import Path


def load_config(path):
    if path is None:
        return
    selected = Path(path)
    if not selected.is_file():
        raise ValueError('Configuration file does not exist')
    from dotenv import dotenv_values
    for name, value in dotenv_values(selected, interpolate=False).items():
        if name.startswith('CAREGUIDE_') and value is not None:
            os.environ[name] = value
