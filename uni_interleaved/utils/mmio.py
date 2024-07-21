from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union
from collections import abc
import json
import numpy as np
from abc import ABCMeta, abstractmethod
import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader, Dumper  # type: ignore

from .mmfile_client import FileClient

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)

class BaseFileHandler(metaclass=ABCMeta):
    # `str_like` is a flag to indicate whether the type of file object is
    # str-like object or bytes-like object. Pickle only processes bytes-like
    # objects but json only processes str-like object. If it is str-like
    # object, `StringIO` will be used to process the buffer.
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath: str, mode: str = 'r', **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath: str, mode: str = 'w', **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


class YamlHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault('Loader', Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        return yaml.dump(obj, **kwargs)



def set_default(obj):
    """Set default json values for non-serializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list.
    It also converts ``np.generic`` (including ``np.int32``, ``np.float32``,
    etc.) into plain numbers of plain python built-in types.
    """
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'{type(obj)} is unsupported for json dump')


class JsonHandler(BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('default', set_default)
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('default', set_default)
        return json.dumps(obj, **kwargs)
    
FileLikeObject = Union[TextIO, StringIO, BytesIO]

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
}


def load(file: Union[str, Path, FileLikeObject],
         file_format: Optional[str] = None,
         file_client_args: Optional[Dict] = None,
         **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Note:
        In v1.3.16 and later, ``load`` supports loading data from serialized
        files those can be storaged in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = file_handlers[file_format]
    f: FileLikeObject
    if isinstance(file, str):
        file_client = FileClient.infer_client(file_client_args, file)
        if handler.str_like:
            with StringIO(file_client.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_client.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj: Any,
         file: Optional[Union[str, Path, FileLikeObject]] = None,
         file_format: Optional[str] = None,
         file_client_args: Optional[Dict] = None,
         **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Note:
        In v1.3.16 and later, ``dump`` supports dumping data as strings or to
        files which is saved to different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 's3://path/of/your/file')  # ceph or petrel

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')
    f: FileLikeObject
    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        file_client = FileClient.infer_client(file_client_args, file)
        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_client.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_client.put(f.getvalue(), file)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')