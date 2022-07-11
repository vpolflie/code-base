"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports

# Internal imports


class ConvertTypeException(Exception):
    """
        An exception indicating that the initial data to convert didn't match one of the possible options
    """

    def __init__(self, data_type):
        self.message = "Incorrect type provided: %s" % str(data_type)
        super().__init__(self.message)
