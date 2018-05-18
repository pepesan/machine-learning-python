import numpy as np
from bokeh.sampledata.iris import flowers
from holoviews import Scatter
import holoviews as hv
np.random.seed(10)
data = np.random.rand(100,4)

scatter = hv.Scatter(data, vdims=['y', 'z', 'size'])

scatter + scatter[0.3:0.7, 0.3:0.7].hist()