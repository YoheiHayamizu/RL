import numpy as np


class MDPStateClass(object):
    def __init__(self, data, is_terminal=False):
        self.__data = data
        self._is_terminal = is_terminal

    # Accessors

    def get_data(self):
        return self.__data

    def is_terminal(self):
        return self._is_terminal

    # Setters

    def set_data(self, data):
        self.__data = data

    def set_terminal(self, is_terminal=True):
        self._is_terminal = is_terminal

    # Core

    def __hash__(self):
        if type(self.__data).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.__data))
        elif self.__data.__hash__ is None:
            return hash(tuple(self.__data))
        else:
            return hash(self.__data)

    def __eq__(self, other):
        assert isinstance(other, MDPStateClass), "Arg object is not " + type(self.__data).__module__
        return self.__data == other.__data

    def __getitem__(self, index):
        return self.__data[index]

    def __len__(self):
        return len(self.__data)
