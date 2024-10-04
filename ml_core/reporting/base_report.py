import datetime
from abc import ABC, abstractmethod


class Report(ABC):
    """Base report class for all reports.
    
    Args:
        name (str): The name of the report, for storing purposes.
        title (str): The title of the report the user wants to see displayed 
            at the top of the report.
        creation_date (datetime.datetime): The creation date of the report.
    """
    def __init__(
            self, 
            name: str, 
            title: str,
            creation_date: datetime.datetime
        ):
        self.name = name
        self.title = title
        self.creation_date = creation_date

    @abstractmethod
    def generate(self, data):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def print_report(self):
        pass