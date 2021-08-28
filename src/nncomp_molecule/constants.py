import os
from pathlib import Path


INPUTDIR = Path(os.environ["INPUTDIR"])
OUTPUTDIR = Path(os.environ["OUTPUTDIR"])
CACHEDIR = Path(os.environ["CACHEDIR"])
COMPETITION_ID = os.environ["COMPETITION_ID"]
COMPETITION_DATADIR = INPUTDIR / COMPETITION_ID
