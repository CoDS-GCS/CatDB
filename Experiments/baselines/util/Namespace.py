import pprint
from ast import literal_eval
from collections import defaultdict
from copy import deepcopy


def partition(iterable, predicate=id):
    truthy, falsy = [], []
    for i in iterable:
        if predicate(i):
            truthy.append(i)
        else:
            falsy.append(i)
    return truthy, falsy


class Namespace:
    printer = pprint.PrettyPrinter(indent=2, compact=True)

    @staticmethod
    def parse(*args, **kwargs):
        raw = dict(*args, **kwargs)
        parsed = Namespace()
        dots, nodots = partition(raw.keys(), lambda s: '.' in s)
        for k in nodots:
            v = raw[k]
            try:
                if isinstance(v, str):
                    v = literal_eval(v)
            except Exception:
                pass
            parsed[k] = v
        sublevel = {}
        for k in dots:
            k1, k2 = k.split('.', 1)
            entry = [(k2, raw[k])]
            if k1 in sublevel:
                sublevel[k1].update(entry)
            else:
                sublevel[k1] = dict(entry)
        for k, v in sublevel.items():
            parsed[k] = Namespace.parse(v)
        return parsed

    @staticmethod
    def merge(*namespaces, deep=False):
        merged = Namespace()
        for ns in namespaces:
            if ns is None:
                continue
            if not deep:
                merged += ns
            else:
                for k, v in ns:
                    if isinstance(v, Namespace):
                        merged[k] = Namespace.merge(merged[k], v, deep=True)
                    else:
                        merged[k] = v
        return merged

    @staticmethod
    def dict(namespace, deep=True):
        dic = dict(namespace)
        if not deep:
            return dic
        for k, v in dic.items():
            if isinstance(v, Namespace):
                dic[k] = Namespace.dict(v)
        return dic

    @staticmethod
    def from_dict(dic, deep=True):
        ns = Namespace(dic)
        if not deep:
            return ns
        for k, v in ns:
            if isinstance(v, dict):
                ns[k] = Namespace.from_dict(v)
        return ns

    @staticmethod
    def walk(namespace, fn, inplace=False):
        def _walk(namespace, fn, parents=None, inplace=inplace):
            parents = [] if parents is None else parents
            ns = namespace if inplace else Namespace()
            for k, v in namespace:
                nk, nv = fn(k, v, parents=parents)
                if nk is not None:
                    if v is nv and isinstance(v, Namespace):
                        nv = _walk(nv, fn, parents=parents + [k], inplace=inplace)
                    ns[nk] = nv
            return ns

        return _walk(namespace, fn, inplace=inplace)

    @staticmethod
    def get(namespace, key, default=None):
        """
        Allows access to a nested key using dot syntax.
        Doesn't raise if key doesn't exist.
        """
        ks = key.split('.', 1)
        if len(ks) > 1:
            n1 = getattr(namespace, ks[0], None)
            return default if n1 is None else Namespace.get(n1, ks[1], default)
        else:
            return getattr(namespace, key, default)

    @staticmethod
    def set(namespace, key, value):
        """
        Allows setting a nested key using dot syntax.
        """
        ks = key.split('.', 1)
        if len(ks) > 1:
            n1 = getattr(namespace, ks[0], None)
            if n1 is None:
                n1 = Namespace()
                setattr(namespace, ks[0], n1)
            Namespace.set(n1, ks[1], value)
        else:
            setattr(namespace, key, value)

    @staticmethod
    def delete(namespace, key):
        """
        Allows deleting a nested key using dot syntax.
        Doesn't raise if key doesn't exist.
        """
        ks = key.split('.', 1)
        if len(ks) > 1:
            n1 = getattr(namespace, ks[0], None)
            if n1 is not None:
                Namespace.delete(n1, ks[1])
        elif hasattr(namespace, key):
            delattr(namespace, key)

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and callable(args[0]):
            self.__dict__ = defaultdict(args[0])
            args = args[1:]
        self.__dict__.update(dict(*args, **kwargs))

    def __add__(self, other):
        res = Namespace()
        res += self
        res += other
        return res

    def __iadd__(self, other):
        """extends self with other (always overrides)"""
        if other is not None:
            self.__dict__.update(other)
        return self

    def __or__(self, other):
        res = Namespace()
        res |= self
        res |= other
        return res

    def __ior__(self, other):
        """extends self with other (adds only missing keys)"""
        if other is not None:
            for k, v in other:
                self.__dict__.setdefault(k, v)
        return self

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getattr__(self, item):
        if isinstance(self.__dict__, defaultdict):
            return self.__dict__[item]
        raise AttributeError(f"'Namespace' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        self.__dict__.pop(key, None)

    def __iter__(self):
        return iter(self.__dict__.items())

    def __copy__(self):
        return Namespace(self.__dict__.copy())

    def __deepcopy__(self, memo={}):
        new_dict = self.__dict__.copy()
        for k, v in new_dict.items():
            if isinstance(v, Namespace):
                new_dict[k] = deepcopy(v, memo)
        return Namespace(new_dict)

    def __dir__(self):
        return list(self.__dict__.keys())

    def __eq__(self, other):
        return isinstance(other, Namespace) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return Namespace.printer.pformat(Namespace.dict(self))

    def __repr__(self):
        return repr(self.__dict__)

    def __json__(self):
        return Namespace.dict(self)
