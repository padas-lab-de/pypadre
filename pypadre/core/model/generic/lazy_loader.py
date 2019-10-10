class LazyObject(object):
    def __init__(self, *, id, load_fn):
        self._id = id
        self._load_fn = load_fn
        self._wrappee = None

    @property
    def id(self):
        return self._id

    def __getattr__(self, attr):
        if self._wrappee is None:
            self._wrappee = self._load_fn()
        return getattr(self._wrappee, attr)
