"""Unit tests for component cleanup tools and scripts."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from logic.package.cleanup_helper import (
    PROTECTED_DIRS,
    _match_acronym,
    clean_by_acronym,
    clean_factory_file,
    clean_init_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]




class TestCleanupComponents(unittest.TestCase):
    """Test cases for component cleanup functions."""

    def test_match_acronym(self):
        """Test acronym match helper."""
        self.assertTrue(_match_acronym("alns", "alns"))
        self.assertTrue(_match_acronym("policy_alns", "alns"))
        self.assertTrue(_match_acronym("selection_alns", "alns"))
        self.assertTrue(_match_acronym("alns_policy", "alns"))
        self.assertTrue(_match_acronym("my_alns_solver", "alns"))

        # Test initials matching
        self.assertTrue(_match_acronym("branch_and_price_and_cut", "bpc"))
        self.assertTrue(_match_acronym("nearest_neighbor", "nn"))
        self.assertTrue(_match_acronym("nn_policy", "nn"))

        # Non-matching cases
        self.assertFalse(_match_acronym("random_policy", "alns"))
        self.assertFalse(_match_acronym("base_policy", "bpc"))

    @patch("logic.package.cleanup_helper.Path.exists", return_value=True)
    def test_clean_init_file(self, mock_exists):
        """Test clean_init_file comments out correct import and registry lines."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        # Mock file content
        content = (
            "from .alns import ALNS\n"
            "from .bpc import BPC\n"
            "from .other import OTHER\n"
            "REGISTRY = {\n"
            "    'alns': ALNS,\n"
            "    'bpc': BPC\n"
            "}\n"
        )
        mock_path.read_text.return_value = content

        clean_init_file(mock_path, ["alns"])

        # Verify write_text was called with commented content
        mock_path.write_text.assert_called_once()
        written_content = mock_path.write_text.call_args[0][0]

        self.assertIn("# from .alns import ALNS  # AUTO-CLEANED", written_content)
        self.assertIn("from .bpc import BPC", written_content)
        self.assertIn("#     'alns': ALNS,  # AUTO-CLEANED", written_content)
        self.assertIn("'bpc': BPC", written_content)

    @patch("logic.package.cleanup_helper.Path.exists", return_value=True)
    def test_clean_factory_file(self, mock_exists):
        """Test clean_factory_file comments out factory entries."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True

        content = (
            "SUPPORTED = [\n"
            "    'alns',\n"
            "    'bpc'\n"
            "]\n"
        )
        mock_path.read_text.return_value = content

        clean_factory_file(mock_path, ["alns"])

        mock_path.write_text.assert_called_once()
        written_content = mock_path.write_text.call_args[0][0]

        self.assertIn("#     'alns',  # AUTO-CLEANED", written_content)
        self.assertIn("'bpc'", written_content)

    @patch("logic.package.cleanup_helper.get_project_root")
    @patch("logic.package.cleanup_helper.remove_path")
    @patch("logic.package.cleanup_helper.clean_init_file")
    @patch("logic.package.cleanup_helper.clean_factory_file")
    def test_clean_by_acronym_scans_and_deletes(
        self, mock_clean_factory, mock_clean_init, mock_remove, mock_get_root
    ):
        """Test clean_by_acronym sweeps paths, avoids protected dirs, and cleans files."""
        # Create a mock file system structure
        mock_root = MagicMock(spec=Path)
        mock_get_root.return_value = mock_root

        # Mock paths
        mock_yaml = MagicMock(spec=Path)
        mock_yaml.stem = "policy_alns"
        mock_yaml.suffix = ".yaml"
        mock_yaml.is_file.return_value = True
        mock_yaml.is_dir.return_value = False

        mock_impl = MagicMock(spec=Path)
        mock_impl.stem = "alns"
        mock_impl.suffix = ".py"
        mock_impl.is_file.return_value = True
        mock_impl.is_dir.return_value = False
        mock_impl.parent.name = "routing"  # Protected, so only delete the file, not parent

        mock_impl_dir = MagicMock(spec=Path)
        mock_impl_dir.name = "my_custom_alns"
        mock_impl_dir.is_dir.return_value = True
        mock_impl_dir.is_file.return_value = False

        # Set up glob returning our mocks
        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = True
        mock_dir.glob.side_effect = lambda pat: (
            [mock_yaml] if "yaml" in pat else
            [mock_impl, mock_impl_dir] if "*" in pat or "**/*" in pat else []
        )
        mock_root.__truediv__.return_value = mock_dir

        clean_by_acronym(
            acronym="alns",
            yaml_dirs=["logic/configs/policies"],
            config_dirs=["logic/src/configs/policies"],
            impl_dirs=["logic/src/policies/routing"]
        )

        # Verify deletes occurred
        self.assertTrue(mock_remove.called)

        # Ensure protected directories themselves were not deleted
        for call_args in mock_remove.call_args_list:
            deleted_path = call_args[0][0]
            self.assertNotIn(deleted_path.name, PROTECTED_DIRS)
