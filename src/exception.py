# exception.py
import sys
from loguru import logger


def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {str(error)} (no traceback available)"

    # Walk to deepest frame
    tb = exc_tb
    while tb.tb_next is not None:
        tb = tb.tb_next

    file_name = tb.tb_frame.f_code.co_filename
    func_name = tb.tb_frame.f_code.co_name
    line_no   = tb.tb_frame.f_lineno       # fixed: f_lineno over tb_lineno

    return (
        f"Error occurred in script : [{file_name}] "
        f"function : [{func_name}] "
        f"line     : [{line_no}] "
        f"message  : [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        if isinstance(error_message, CustomException):
            # Already a CustomException — reuse existing message
            # Prevents re-wrapping and preserves original file/line info
            super().__init__(error_message.error_message)
            self.error_message = error_message.error_message
        else:
            super().__init__(error_message)
            self.error_message = error_message_detail(error_message, error_detail)
            # opt(colors=False) — prevents loguru misreading [...] as color tags
            logger.opt(colors=False).error(self.error_message)

    def __str__(self):
        return self.error_message