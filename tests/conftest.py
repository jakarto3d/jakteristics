from pathlib import Path
import shutil

import pytest

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    tmp = TEST_DATA / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)
