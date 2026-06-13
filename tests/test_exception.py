import sys

from src.exception import CustomException, error_message_detail


def test_error_message_detail_with_traceback():
    try:
        raise ValueError("bad input")
    except Exception as e:
        msg = error_message_detail(e, sys)
        assert "ValueError" not in msg
        assert "bad input" in msg
        assert "function" in msg
        assert "line" in msg


def test_custom_exception_str():
    try:
        raise ValueError("something went wrong")
    except Exception as e:
        err = CustomException(e, sys)
        assert "something went wrong" in str(err)


def test_custom_exception_rewraps_existing_exception():
    try:
        raise ValueError("original error")
    except Exception as e:
        first = CustomException(e, sys)
        second = CustomException(first, sys)
        assert str(second) == str(first)