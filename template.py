import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] : %(message)s:')


list_of_files = [
    "src/__init__.py",
    "src/helper.py.py",
   
    
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    "test.py"
]





for filepath in list_of_files:
    filepath = Path(filepath)  # Convert to Path object
    filedir, filename = os.path.split(filepath)

    # If directory part is not empty, create the directory
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # If file doesn't exist or is empty, create an empty file
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")