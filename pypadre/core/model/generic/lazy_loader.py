# TODO better proxy, partially like django https://docs.djangoproject.com/en/2.0/_modules/django/utils/functional/
# https://stackoverflow.com/questions/9942536/how-to-fake-proxy-a-class-in-python
# TODO write tests for this
import copy
import operator

empty = object()


def new_method_proxy(func):
    def inner(self, *args):
        if self._wrapped is empty:
            self._setup()
        return func(self._wrapped, *args)

    return inner


class SimpleLazyObject(object):

    def __init__(self, *, load_fn, id, clz, **kwargs):
        no_proxy_fields = ["_wrapped", "_setup_func", "id", "_clz", "_eager_fields"]
        no_proxy_fields.extend([k for k, v in kwargs.items()])
        self._no_proxy_fields = no_proxy_fields
        self._eager_fields = kwargs
        self._setup_func = load_fn
        self._wrapped = empty
        self._clz = clz
        self.id = id
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __getattr__(self, attr):
        if attr == "_no_proxy_fields" or attr in self._no_proxy_fields:
            return getattr(self, attr)
        self._setup()
        return getattr(self._wrapped, attr)

    def __setattr__(self, name, value):
        if name == "_no_proxy_fields" or name in self._no_proxy_fields:
            # Assign to __dict__ to avoid infinite __setattr__ loops.
            object.__setattr__(self, name, value)
        else:
            self._setup()
            setattr(self._wrapped, name, value)

    def _setup(self):
        if self._wrapped is empty:
            self._wrapped = self._setup_func()

    # Return a meaningful representation of the lazy object for debugging
    # without evaluating the wrapped object.
    def __repr__(self):
        if self._wrapped is empty:
            repr_attr = self._setup_func
        else:
            repr_attr = self._wrapped
        return '<%s: %r>' % (type(self).__name__, repr_attr)

    def __copy__(self):
        if self._wrapped is empty:
            # If uninitialized, copy the wrapper. Use SimpleLazyObject, not
            # self.__class__, because the latter is proxied.
            return SimpleLazyObject(load_fn=self._setup_func, id=self.id, clz=self._clz, **self._eager_fields)
        else:
            # If initialized, return a copy of the wrapped object.
            return copy.copy(self._wrapped)

    def __deepcopy__(self, memo):
        if self._wrapped is empty:
            # We have to use SimpleLazyObject, not self.__class__, because the
            # latter is proxied.
            # TODO deep copy eager fields
            result = SimpleLazyObject(load_fn=self._setup_func, id=self.id, clz=self._clz, **self._eager_fields)
            memo[id(self)] = result
            return result
        return copy.deepcopy(self._wrapped, memo)

    # Need to pretend to be the wrapped class, for the sake of objects that
    # care about this (especially in equality tests)
    __class__ = property(new_method_proxy(operator.attrgetter("__class__")))
    __eq__ = new_method_proxy(operator.eq)
    __lt__ = new_method_proxy(operator.lt)
    __gt__ = new_method_proxy(operator.gt)
    __ne__ = new_method_proxy(operator.ne)
    __hash__ = new_method_proxy(hash)

    # List/Tuple/Dictionary methods support
    __getitem__ = new_method_proxy(operator.getitem)
    __setitem__ = new_method_proxy(operator.setitem)
    __delitem__ = new_method_proxy(operator.delitem)
    __iter__ = new_method_proxy(iter)
    __len__ = new_method_proxy(len)
    __contains__ = new_method_proxy(operator.contains)
