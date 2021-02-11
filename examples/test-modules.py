import sys
import os
import logging

import scipy
import numpy as np

import sklearn
import pyrcn

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


def main():
    logger = new_logger('main')
    logger.info('Created logger successfully')

    for modulename in ['scipy', 'numpy', 'sklearn', 'pyrcn']:
        if modulename not in sys.modules:
            logger.error('Module {0} was not loaded'.format(modulename))
        else:
            logger.info('Module {0} loaded'.format(modulename))

    from pyrcn.extreme_learning_machine.tests import test_elm
    test_elm.test_iris_ensemble_iterative_regression()

    logger.info('Test run fished')
    return


if __name__ == '__main__':
    main()
    exit(0)
