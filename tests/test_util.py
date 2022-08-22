"""Testing for pyrcn.utils module"""
import os
import pytest
from pyrcn.util import new_logger, argument_parser, get_mnist


def test_new_logger() -> None:
    directory = os.getcwd()
    logger = new_logger(name='test_logger', directory=directory)
    logger.info('Test')
    assert os.path.isfile(os.path.join(directory, 'test_logger.log'))


def test_argument_parser() -> None:
    args = argument_parser.parse_args(['-o', './', 'param0', 'param1'])
    assert os.path.isdir(args.out)
    assert 'param1' in args.params


@pytest.mark.skip(reason="no way of currently testing this")
def test_get_mnist() -> None:
    X, y = get_mnist(os.getcwd())
    assert X.shape[0] == 70000
