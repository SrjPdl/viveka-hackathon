import sys

def get_error_info(err: str, err_detail: sys) -> str:
    """Get error information.\\
    Args:
        err : The error message.
        err_detail : The detailed error information.
    Returns:
        The formatted error information.

    Examples:
        >>> try:
        ...     # Some code that may raise an error
        ...     pass
        ... except Exception as e:
        ...     error_message = str(e)
        ...     error_info = get_error_info(error_message, sys.exc_info())
        ...     print(error_info)
        'Error occurred in example.py at line 10'


    """
    _, _, tb = sys.exc_info()
    exception_file = tb.tb_frame.f_code.co_filename
    exception_line = tb.tb_lineno
    exception_info = f"{err} in {exception_file} at line {exception_line}"
    return exception_info
    


class CustomException(Exception):
    """
    A custom exception class that provides formatted error information.

    """
    def __init__(self, err: str, err_detail: sys):
        """
        Args:
            err : The error message.
            err_detail : The detailed error information.
        Attributes:
            message : The formatted error information.

        Examples:
            >>> try:
            ...     # Some code that may raise an error
            ...     pass
            ... except Exception as e:
            ...     error_message = str(e)
            ...     raise CustomException(error_message, sys)
    """
        super().__init__(err)
        self.message = get_error_info(err, err_detail)
    
    def __str__(self):
        return self.message