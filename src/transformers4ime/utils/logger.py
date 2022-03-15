"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

helper for logging
NOTE: loggers are global objects use with caution
"""
import logging

from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

_LOG_FMT = '%(asctime)s - %(levelname)s  {%(filename)s:%(lineno)d} - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')  # this is the global logger


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""

    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def add_log_to_file(log_path):
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(TqdmLoggingHandler())


class TensorboardLogger(object):
    def __init__(self):
        self._logger = None
        self._global_step = 0

    def create(self, path):
        self._logger = SummaryWriter(path)

    def noop(self, *args, **kwargs):
        return

    def step(self):
        self._global_step += 1

    @property
    def global_step(self):
        return self._global_step

    def log_scaler_dict(self, log_dict, prefix=''):
        """ log a dictionary of scalar values"""
        if self._logger is None:
            return
        if prefix:
            prefix = f'{prefix}_'
        for name, value in log_dict.items():
            if isinstance(value, dict):
                self.log_scaler_dict(value, self._global_step,
                                     prefix=f'{prefix}{name}')
            else:
                self._logger.add_scalar(f'{prefix}{name}', value,
                                        self._global_step)

    def __getattr__(self, name):
        if self._logger is None:
            return self.noop
        return self._logger.__getattribute__(name)


class RunningMeter(object):
    """ running meteor of a scalar value
        (useful for monitoring training loss)
    """

    def __init__(self, name, val=None, smooth=0.99):
        self._name = name
        self._sm = smooth
        self._val = val

    def __call__(self, value):
        self._val = (value if self._val is None
                     else value * (1 - self._sm) + self._val * self._sm)

    def __str__(self):
        return f'{self._name}: {self._val:.4f}'

    @property
    def val(self):
        return self._val

    @property
    def name(self):
        return self._name
