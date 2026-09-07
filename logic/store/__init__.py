"""
Cloud storage integration for WSmart+ Route artifacts.

Subpackages
-----------
upload/
    presentation.py  Upload generated .pptx decks to Microsoft OneDrive /
                     SharePoint via the Microsoft Graph API.
    results.py       Upload a selected subset of the simulation results under
                     assets/output/ to a cloud drive (OneDrive, Google Drive
                     or Dropbox).
    datasets.py      Upload dataset files from data/datasets and
                     data/simulator to a cloud drive.
download/
    datasets.py      Download dataset files from a cloud drive into
                     data/datasets / data/simulator.

Providers and credentials are documented in logic/store/providers.py; all
credentials are read from environment variables (see logic/store/config.py).
"""

from logic.store.providers import get_provider  # noqa: F401
