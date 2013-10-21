from decaf import base

class SleepLayer(base.Layer):
    """A sleep layer that does nothing other than mapping the blobs,
    and sleeps for a few seconds.
    kwargs:
        sleep: the seconds to sleep.
    """
    def forward(self, bottom, top):
        for bottom_b, bottom_t in zip(bottom, top):
            bottom_t.mirror(bottom_b)
        time.sleep(self.spec['sleep'])
        return

    def backward(self, bottom, up, propagate_down):
        for bottom_b, bottom_t in zip(bottom, top):
            bottom_b.mirror_diff(bottom_t)
        time.sleep(self.spec['sleep'])
        return

