import datetime

from numpy.typing import ArrayLike

from ml_core.linear_models.linear_regression import LinearRegression
from ml_core.reporting.base_report import Report


class ReportLinearRegression(Report):
    """Report for the linear regression model parameter statistics.
    
    Args:
        name (str): The name of the report, for storing purposes.
        title (str): The title of the report the user wants to see displayed 
            at the top of the report.
        creation_date (datetime.datetime): The creation date of the report.
    """
    def __init__(
            self, 
            title: str,
            creation_date: datetime.datetime = datetime.datetime.now(),
        ):
        name = "linear_regression_stats_report"
        super().__init__(name, title, creation_date)

    def generate(self, model: LinearRegression, X: ArrayLike) -> str:
        pass

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def print_report(self):
        pass