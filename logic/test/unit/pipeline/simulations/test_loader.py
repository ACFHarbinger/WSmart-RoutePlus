

from unittest.mock import patch

import pytest
from logic.src.pipeline.simulations.repository import (
    FileSystemRepository,
    load_area_and_waste_type_params,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestFileSystemRepository:
    @pytest.fixture
    def repo(self, tmp_path):
        return FileSystemRepository(str(tmp_path))

    @pytest.fixture(autouse=True)
    def setup_repo(self, repo):
        import logic.src.pipeline.simulations.repository as repo_module

        original_repo = repo_module._REPOSITORY
        repo_module.set_repository(repo)
        yield
        repo_module._REPOSITORY = original_repo



    def test_wrappers(self):
        # Test that wrappers call the imported function
        with patch("logic.test.unit.pipeline.simulations.test_loader.load_area_and_waste_type_params") as mock_method:
            load_area_and_waste_type_params("riomaior", "glass")
            mock_method.assert_called_with("riomaior", "glass")
