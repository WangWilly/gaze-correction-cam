import datetime
import os
import threading


################################################################################


class Logger:
    """Logger utility class that formats messages in a specific format."""

    def __init__(self, class_name):
        self.class_name = class_name
        self.process_id = os.getpid()

    def log(self, message):
        """Log a message with the specified format."""
        # Get current timestamp with milliseconds precision
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Get thread ID
        thread_id = threading.get_ident()

        # Format the log message
        formatted_message = f"{timestamp} Python[{self.process_id}:{thread_id}] +[{self.class_name}]: {message}"

        # Print the formatted message
        print(formatted_message)
