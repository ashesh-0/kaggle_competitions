from datetime import datetime


def timer(tag_name):
    def timer_decorator(fn):
        def _fn(*args, **kwargs):
            s = datetime.now()
            output = fn(*args, **kwargs)
            e = datetime.now()
            print('[{}] {} completed in {}'.format(tag_name, fn.__name__, e - s))
            return output

        return _fn

    return timer_decorator
