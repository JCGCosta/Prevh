import logging, inspect
from datetime import datetime

class ErrorHandler:
    def __init__(self):
        logging.basicConfig(
            filename=f"operation.log",
            encoding="utf-8",
            filemode="w",
            format="%(levelname)s:%(name)s:%(message)s",
        )
        self.logger = logging.getLogger("ErrorHandler")

    def exec(self, message: str, log_level: str, exception):
        # Dynamically log based on the level provided
        stack = inspect.stack()
        the_class = stack[1][0].f_locals["self"].__class__.__name__
        the_method = stack[1][0].f_code.co_name
        log_method = getattr(self.logger, log_level.lower(), self.logger.error)
        log_method(f"Caller {the_class}.{the_method}():\n" + message)
        raise exception(message)