import os
from dataclasses import dataclass


@dataclass
class settings:
    project_path = os.path.join(os.environ["CODE_PATH"], "oblak")
    model_folder_path = os.path.join(project_path, "/serialization/models/")
