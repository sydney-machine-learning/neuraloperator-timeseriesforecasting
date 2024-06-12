import os
from camels_aus.repository import download_camels_aus, CamelsAus

repo = CamelsAus()

# Use HOME for Unix-like systems and USERPROFILE for Windows
home_dir = os.getenv("HOME") or os.getenv("USERPROFILE")

if home_dir is None:
    raise EnvironmentError("Neither HOME nor USERPROFILE environment variables are set.")

camels_dir = os.path.join(home_dir, 'data/camels/aus')
download_camels_aus(camels_dir)
repo.load_from_text_files(camels_dir)

if isinstance(repo.data, dict):
    print("Keys in repo.data:", repo.data.keys())
else:
    print("Attributes of repo.data:", dir(repo.data))