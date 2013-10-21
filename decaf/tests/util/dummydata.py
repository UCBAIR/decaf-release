from decaf import base

class DummyDataLayer(base.Layer):
    """A layer that produces dummy data.
    kwargs:
        shape: the shape to produce.
        dtype: the dtype
    """
    def forward(self, bottom, top):
        data = top[0].init_data(self.spec['shape'], self.spec['dtype'])
        return
