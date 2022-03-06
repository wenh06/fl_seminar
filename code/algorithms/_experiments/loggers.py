"""
loggers, including (planned) CSVLogger

with reference to `loggers` of `textattack` and `loggers` of `pytorch-lightning`
"""

import os, logging, csv, importlib, re
from datetime import datetime
from abc import ABC, abstractmethod
from typing import NoReturn, Optional, Union, List, Any, Dict
from numbers import Real

import torch
import pandas as pd

from misc import ReprMixin, LOG_DIR


__all__ = [
    "BaseLogger",
    "TxtLogger",
    "CSVLogger",
    "LoggerManager",
]


class BaseLogger(ReprMixin, ABC):
    """
    the abstract base class of all loggers
    """
    __name__ = "BaseLogger"

    @abstractmethod
    def log_metrics(self,
                    client_id:Union[int,type(None)],
                    metrics:Dict[str, Union[Real,torch.Tensor]],
                    step:Optional[int]=None,
                    epoch:Optional[int]=None,
                    part:str="val",) -> NoReturn:
        """

        Parameters
        ----------
        client_id: int,
            the index of the client, `None` for the server
        metrics: dict,
            the metrics to be logged
        step: int, optional,
            the number of (global) steps of training
        epoch: int, optional,
            the epoch number of training
        part: str, optional,
            the part of the training data the metrics computed from,
            can be "train" or "val" or "test", etc.
        """
        raise NotImplementedError

    @abstractmethod
    def log_message(self, msg:str, level:int=logging.INFO) -> NoReturn:
        """
        log a message

        Parameters
        ----------
        msg: str,
            the message to be logged
        level: int, optional,
            the level of the message,
            can be logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config:Dict[str, Any]) -> Any:
        """
        """
        raise NotImplementedError

    def epoch_start(self, epoch:int) -> NoReturn:
        """
        actions to be performed at the start of each epoch

        Parameters
        ----------
        epoch: int,
            the number of the epoch
        """
        pass

    def epoch_end(self, epoch:int) -> NoReturn:
        """
        actions to be performed at the end of each epoch

        Parameters
        ----------
        epoch: int,
            the number of the epoch
        """
        pass

    @property
    def log_dir(self) -> str:
        """
        """
        return self._log_dir

    @property
    @abstractmethod
    def filename(self) -> str:
        """
        """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        """
        extra keys to be displayed in the repr of the logger
        """
        return super().extra_repr_keys() + ["filename",]


class TxtLogger(BaseLogger):
    """
    """
    __name__ = "TxtLogger"

    def __init__(self,
                 algorithm:str,
                 dataset:str,
                 model:str,
                 log_dir:Optional[str]=None,
                 log_suffix:Optional[str]=None,) -> NoReturn:
        """

        Parameters
        ----------
        algorithm, dataset, model: str,
            used to form the prefix of the log file
        log_dir: str, optional,
            the directory to save the log file
        log_suffix: str, optional,
            the suffix of the log file
        """
        assert all([isinstance(x, str) for x in [algorithm, dataset, model]]), \
            "algorithm, dataset, model must be str"
        log_prefix = re.sub("[\s]+", "_", f"{algorithm}-{dataset}-{model}")
        self._log_dir = log_dir or LOG_DIR
        if log_suffix is None:
            log_suffix = ""
        else:
            log_suffix = f"_{log_suffix}"
        self.log_file = f"{log_prefix}_{get_date_str()}{log_suffix}.txt"
        self.logger = init_logger(self.log_dir, self.log_file, verbose=1)
        self.step = -1

    def log_metrics(self,
                    client_id:Union[int,type(None)],
                    metrics:Dict[str, Union[Real,torch.Tensor]],
                    step:Optional[int]=None,
                    epoch:Optional[int]=None,
                    part:str="val",) -> NoReturn:
        """

        Parameters
        ----------
        client_id: int,
            the index of the client, `None` for the server
        metrics: dict,
            the metrics to be logged
        step: int, optional,
            the number of (global) steps of training
        epoch: int, optional,
            the epoch number of training
        part: str, optional,
            the part of the training data the metrics computed from,
            can be "train" or "val" or "test", etc.
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        prefix = f"Step {step}: "
        if epoch is not None:
            prefix = f"Epoch {epoch} / {prefix}"
        _metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k,v in metrics.items()}
        spaces = len(max(_metrics.keys(), key=len))
        node = "Server" if client_id is None else f"Client {client_id}"
        msg = f"{node} {part.capitalize()} Metrics:\n{self.short_sep}\n" \
            + "\n".join([f"{prefix}{part}/{k} : {' '*(spaces-len(k))}{v:.4f}" for k,v in _metrics.items()]) \
            + f"\n{self.short_sep}"
        self.log_message(msg)

    def log_message(self, msg:str, level:int=logging.INFO) -> NoReturn:
        """
        log a message

        Parameters
        ----------
        msg: str,
            the message to be logged
        level: int, optional,
            the level of the message,
            can be logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        """
        self.logger.log(level, msg)

    @property
    def long_sep(self) -> str:
        """
        """
        return "-"*110

    @property
    def short_sep(self) -> str:
        """
        """
        return "-"*50

    def epoch_start(self, epoch:int) -> NoReturn:
        """
        message logged at the start of each epoch

        Parameters
        ----------
        epoch: int,
            the number of the epoch
        """
        self.logger.info(f"Train epoch_{epoch}:\n{self.long_sep}")

    def epoch_end(self, epoch:int) -> NoReturn:
        """
        message logged at the end of each epoch

        Parameters
        ----------
        epoch: int,
            the number of the epoch
        """
        self.logger.info(f"{self.long_sep}\n")

    def flush(self) -> NoReturn:
        """
        """
        for h in self.logger.handlers:
            if hasattr(h, flush):
                h.flush()

    def close(self) -> NoReturn:
        """
        """
        for h in self.logger.handlers:
            h.close()
            self.logger.removeHandler(h)
        logging.shutdown()

    @classmethod
    def from_config(cls, config:Dict[str, Any]) -> "TxtLogger":
        """
        """
        return cls(config.get("log_dir", None), config.get("log_suffix", None))

    @property
    def filename(self) -> str:
        """
        """
        return os.path.join(self.log_dir, self.log_file)


class CSVLogger(BaseLogger):
    """
    """
    __name__ = "CSVLogger"

    def __init__(self,
                 algorithm:str,
                 dataset:str,
                 model:str,
                 log_dir:Optional[str]=None,
                 log_suffix:Optional[str]=None,) -> NoReturn:
        """

        Parameters
        ----------
        algorithm, dataset, model: str,
            used to form the prefix of the log file
        log_dir: str, optional,
            the directory to save the log file
        log_suffix: str, optional,
            the suffix of the log file
        """
        assert all([isinstance(x, str) for x in [algorithm, dataset, model]]), \
            "algorithm, dataset, model must be str"
        log_prefix = re.sub("[\s]+", "_", f"{algorithm}-{dataset}-{model}")
        self._log_dir = log_dir or LOG_DIR
        if log_suffix is None:
            log_suffix = ""
        else:
            log_suffix = f"_{log_suffix}"
        self.log_file = f"{log_prefix}_{get_date_str()}{log_suffix}.csv"
        self.logger = pd.DataFrame()
        self.step = -1
        self._flushed = True

    def log_metrics(self,
                    client_id:Union[int,type(None)],
                    metrics:Dict[str, Union[Real,torch.Tensor]],
                    step:Optional[int]=None,
                    epoch:Optional[int]=None,
                    part:str="val",) -> NoReturn:
        """

        Parameters
        ----------
        client_id: int,
            the index of the client, `None` for the server
        metrics: dict,
            the metrics to be logged
        step: int, optional,
            the number of (global) steps of training
        epoch: int, optional,
            the epoch number of training
        part: str, optional,
            the part of the training data the metrics computed from,
            can be "train" or "val" or "test", etc.
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        row = {"step": self.step, "time": datetime.now(), "part": part}
        if epoch is not None:
            row.update({"epoch": epoch})
        node = "Server" if client_id is None else f"Client{client_id}"
        row.update(
            {
                f"{node}-{k}": v.item() if isinstance(v, torch.Tensor) else v \
                    for k,v in metrics.items()
            }
        )
        self.logger = self.logger.append(row, ignore_index=True)
        self._flushed = False

    def log_message(self, msg:str, level:int=logging.INFO) -> NoReturn:
        pass

    def flush(self) -> NoReturn:
        """
        """
        if not self._flushed:
            self.logger.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
            self._flushed = True

    def close(self) -> NoReturn:
        """
        """
        self.flush()

    def __del__(self):
        """
        """
        self.flush()
        del self

    @classmethod
    def from_config(cls, config:Dict[str, Any]) -> "TxtLogger":
        """
        """
        return cls(config.get("log_dir", None), config.get("log_suffix", None))

    @property
    def filename(self) -> str:
        """
        """
        return os.path.join(self.log_dir, self.log_file)


class LoggerManager(ReprMixin):
    """
    """
    __name__ = "LoggerManager"

    def __init__(self,
                 algorithm:str,
                 dataset:str,
                 model:str,
                 log_dir:Optional[str]=None,
                 log_suffix:Optional[str]=None,) -> NoReturn:
        """

        Parameters
        ----------
        algorithm, dataset, model: str,
            used to form the prefix of the log file
        log_dir: str, optional,
            the directory to save the log file
        log_suffix: str, optional,
            the suffix of the log file
        """
        self._algorith = algorithm
        self._dataset = dataset
        self._model = model
        self._log_dir = log_dir or LOG_DIR
        self._log_suffix = log_suffix
        self._loggers = []

    def _add_txt_logger(self) -> NoReturn:
        """
        """
        self.loggers.append(
            TxtLogger(self._algorith, self._dataset, self._model, self._log_dir, self._log_suffix)
        )
    
    def _add_csv_logger(self) -> NoReturn:
        """
        """
        self.loggers.append(
            CSVLogger(self._algorith, self._dataset, self._model, self._log_dir, self._log_suffix)
        )

    def log_metrics(self,
                    client_id:Union[int,type(None)],
                    metrics:Dict[str, Union[Real,torch.Tensor]],
                    step:Optional[int]=None,
                    epoch:Optional[int]=None,
                    part:str="val",) -> NoReturn:
        """

        Parameters
        ----------
        client_id: int,
            the index of the client, `None` for the server
        metrics: dict,
            the metrics to be logged
        step: int, optional,
            the number of (global) steps of training
        epoch: int, optional,
            the epoch number of training
        part: str, optional,
            the part of the training data the metrics computed from,
            can be "train" or "val" or "test", etc.
        """
        for l in self.loggers:
            l.log_metrics(metrics, step, epoch, part)

    def log_message(self, msg:str, level:int=logging.INFO) -> NoReturn:
        """
        log a message

        Parameters
        ----------
        msg: str,
            the message to be logged
        level: int, optional,
            the level of the message,
            can be logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        """
        for l in self.loggers:
            l.log_message(msg, level)

    def epoch_start(self, epoch:int) -> NoReturn:
        """
        action at the start of an epoch

        Parameters
        ----------
        epoch: int,
            the epoch number
        """
        for l in self.loggers:
            l.epoch_start(epoch)

    def epoch_end(self, epoch:int) -> NoReturn:
        """
        action at the end of an epoch

        Parameters
        ----------
        epoch: int,
            the epoch number
        """
        for l in self.loggers:
            l.epoch_end(epoch)

    def flush(self) -> NoReturn:
        """
        """
        for l in self.loggers:
            l.flush()

    def close(self) -> NoReturn:
        """
        """
        for l in self.loggers:
            l.close()

    @property
    def loggers(self) -> List[BaseLogger]:
        """
        """
        return self._loggers

    @property
    def log_dir(self) -> str:
        """
        """
        return self._log_dir

    @property
    def log_suffix(self) -> str:
        """
        """
        return self._log_suffix

    @classmethod
    def from_config(cls, config:Dict[str, Any]) -> "LoggerManager":
        """

        Parameters
        ----------
        config: dict,
            the configuration of the logger manager

        Returns
        -------
        LoggerManager
        """
        lm =  cls(config.get("log_dir", None), config.get("log_suffix", None))
        if config.get("txt_logger", True):
            lm._add_txt_logger()
        if config.get("csv_logger", True):
            lm._add_csv_logger()
        return lm

    def extra_repr_keys(self) -> List[str]:
        """
        extra keys to be displayed in the repr of the logger
        """
        return super().extra_repr_keys() + ["loggers",]



def init_logger(log_dir:str, log_file:Optional[str]=None, log_name:Optional[str]=None, mode:str="a", verbose:int=0) -> logging.Logger:
    """ finished, checked,

    Parameters
    ----------
    log_dir: str,
        directory of the log file
    log_file: str, optional,
        name of the log file
    log_name: str, optional,
        name of the logger
    mode: str, default "a",
        mode of writing the log file, can be one of "a", "w"
    verbose: int, default 0,
        log verbosity

    Returns
    -------
    logger: Logger
    """
    if log_file is None:
        log_file = f"log_{get_date_str()}.txt"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print(f"log file path: {log_file}")

    logger = logging.getLogger(log_name or DEFAULTS.prefix)  # "ECG" to prevent from using the root logger

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)

    if verbose >= 2:
        print("levels of c_handler and f_handler are set DEBUG")
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        print("level of c_handler is set INFO, level of f_handler is set DEBUG")
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        print("level of c_handler is set WARNING, level of f_handler is set INFO")
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def get_date_str(fmt:Optional[str]=None):
    """ finished, checked,

    Parameters
    ----------
    fmt: str, optional,
        format of the string of date

    Returns
    -------
    date_str: str,
        current time in the `str` format
    """
    now = datetime.datetime.now()
    date_str = now.strftime(fmt or "%m-%d_%H-%M")
    return date_str
