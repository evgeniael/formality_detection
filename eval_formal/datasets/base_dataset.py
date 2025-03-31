from abc import ABC, abstractmethod
from pathlib import Path

class BaseDataset(ABC):

    def __init__(self):

        # Create data folder in project root
        project_root = Path(__file__).resolve().parent.parent / ".."
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self):
        """ Load data, e.g. a pandas dataframe. """
        pass

    def __len__(self):
        return len(self.data)