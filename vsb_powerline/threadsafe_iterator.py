# Taken from https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
import threading


class ThreadSafeIter:
    def __init__(self, iter):
        self._itr = iter
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self._itr.__next__()


def threadsafe(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g
