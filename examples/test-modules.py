import sys
import os
import logging
import argparse

import scipy
import numpy as np

import sklearn
import pyrcn


argument_parser = argparse.ArgumentParser(description='Standard input parser for HPC examples on PyRCN.')
argument_parser.add_argument('-o', '--out', nargs='?', help='output directory', type=str)
argument_parser.add_argument('params', metavar='params', nargs='*', help='optional parameter for scripts')


# noinspection PyArgumentList
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)


def new_logger(name, directory=os.getcwd()):
    logger = logging.getLogger(name)
    logger.setLevel(logging.NOTSET)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s')
    handler = logging.FileHandler(os.path.join(directory, '{0}.log'.format(name)))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def main(out_directory=os.getcwd(), param_list=None):
    logger = new_logger('main', directory=out_directory)
    logger.info('Created logger successfully')

    for module_name in ['scipy', 'numpy', 'sklearn', 'pyrcn']:
        if module_name not in sys.modules:
            logger.error('Module {0} was not loaded'.format(module_name))
        else:
            logger.info('Module {0} loaded'.format(module_name))

    from pyrcn.extreme_learning_machine.tests import test_elm
    test_elm.test_iris_ensemble_iterative_regression()

    logger.info('Test run fished')
    return


if __name__ == '__main__':
    parsed_args = argument_parser.parse_args(sys.argv[1:])
    if os.path.isdir(parsed_args.out):
        main(param_list=parsed_args.params, out_directory=parsed_args.out)
    else:
        main(parsed_args.params)
    exit(0)
