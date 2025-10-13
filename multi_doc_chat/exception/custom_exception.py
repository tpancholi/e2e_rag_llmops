# import sys
# import traceback
# from typing import cast
#
#
# class DocumentPortalExceptionError(Exception):
#     def __init__(self, error_message: str, error_details: object | None = None):
#         # normalize message
#         norm_msg = str(error_message) if isinstance(error_message, BaseException) else error_message.strip().lower()
#
#         # resolve exc_info (supports: sys_module, Exception object, current context)
#         exc_type = exc_value = exc_tb = None
#         if error_details is None:
#             exc_type, exc_value, exc_tb = sys.exc_info()
#         elif hasattr(error_details, "exc_info"):
#             exc_info_obj = cast("sys", error_details)
#             exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
#         elif isinstance(error_details, BaseException):
#             exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
#         else:
#             exc_type, exc_value, exc_tb = sys.exc_info()
#
#         # walk the last frame to report the most relevant location
#         last_tb = exc_tb
#         while last_tb and last_tb.tb_next:
#             last_tb = last_tb.tb_next
#
#         self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
#         self.lineno = last_tb.tb_lineno if last_tb else -1
#         self.error_message = norm_msg
#
#         # full pretty traceback
#         if exc_tb and exc_type:
#             self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
#         else:
#             self.traceback_str = ""
#
#         super().__init__(self.__str__())
#
#     def __str__(self):
#         # Compact, logger-friendly message (no leading spaces)
#         base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
#         if self.traceback_str:
#             return f"{base}\nTraceback:\n{self.traceback_str}"
#         return base
#
#     def __repr__(self):
#         return (
#             f"DocumentPortalExceptionError(file={self.file_name!r}, /
#             line={self.lineno}, message={self.error_message!r})"
#         )
import sys
import traceback


class DocumentPortalExceptionError(Exception):
    def __init__(self, error_message: str, error_details: BaseException | object | None = None):
        # normalize message
        norm_msg = str(error_message) if isinstance(error_message, BaseException) else error_message.strip().lower()

        # resolve exc_info (supports: An exception object or current context)
        exc_type = exc_value = exc_tb = None

        if isinstance(error_details, BaseException):
            # If error_details is an exception, extract its traceback
            exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
        else:
            # Otherwise, get current exception info
            exc_type, exc_value, exc_tb = sys.exc_info()

        # walk the last frame to report the most relevant location
        last_tb = exc_tb
        if last_tb:
            while last_tb.tb_next:
                last_tb = last_tb.tb_next

            self.file_name = last_tb.tb_frame.f_code.co_filename
            self.lineno = last_tb.tb_lineno
        else:
            self.file_name = "<unknown>"
            self.lineno = -1

        self.error_message = norm_msg

        # full pretty traceback
        if exc_tb and exc_type:
            self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        # Compact, logger-friendly message (no leading spaces)
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return (
            f"DocumentPortalExceptionError(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"
        )
