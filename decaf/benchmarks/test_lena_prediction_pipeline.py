"""A bunch of test scripts to check the performance of decafnet.
Recommended running command:
    srun -p vision -c 8 --nodelist=orange6 python test_lena_prediction_pipeline.py
"""
from decaf.scripts import decafnet
from decaf import util
from decaf.util import smalldata
import numpy as np
import cProfile as profile

# We will use a larger figure size since many figures are fairly big.
data_root='/u/vis/common/deeplearning/models/'
net = imagenet.DecafNet(data_root+'imagenet.decafnet.epoch90', data_root+'imagenet.decafnet.meta')
lena = smalldata.lena()
timer = util.Timer()

print 'Testing single classification with 10-part voting (10 runs)...'
# run a pass to initialize data
scores = net.classify(lena)
timer.reset()
for i in range(10):
    scores = net.classify(lena)
print 'Elapsed %s' % timer.total()

print 'Testing single classification with center_only (10 runs)...'
# run a pass to initialize data
scores = net.classify(lena, center_only=True)
timer.reset()
for i in range(10):
    scores = net.classify(lena, center_only=True)
print 'Elapsed %s' % timer.total()

lena_ready = lena[np.newaxis, :227,:227].astype(np.float32)
print 'Testing direct classification (10 runs)...'
# run a pass to initialize data
scores = net.classify_direct(lena_ready)
timer.reset()
for i in range(10):
    scores = net.classify_direct(lena_ready)
print 'Elapsed %s' % timer.total()

print 'Dumping computation time for layers:'
decaf_net = net._net
timer.reset()
for name, layer, bottom, top in decaf_net._forward_order:
    for i in range(100):
        layer.predict(bottom, top)
    print '%15s elapsed %f ms' % (name, timer.lap(False) * 10)

print 'Testing direct classification with batches (10 runs)...'
for batch in [1,2,5,10,20,100]:
    lena_batch = np.tile(lena_ready, [batch, 1, 1, 1,]).copy()
    # run a pass to initialize data
    scores = net.classify_direct(lena_batch)
    timer.reset()
    for i in range(10):
        scores = net.classify_direct(lena_batch)
    print 'Batch size %3d, equivalent time %s' % (batch, timer.total(False) / batch)

print 'Profiling batch 1 (100 runs)...'
pr = profile.Profile()
lena_batch = lena_ready.copy()
# run a pass to initialize data
scores = net.classify_direct(lena_batch)
pr.enable()
for i in range(100):
    scores = net.classify_direct(lena_batch)
pr.disable()
pr.dump_stats('lena_profile_batch1.pstats')
print 'Profiling done.'

print 'Profiling batch 10 (10 runs)...'
lena_batch = np.tile(lena_ready, [10, 1, 1, 1,]).copy()
# run a pass to initialize data
scores = net.classify_direct(lena_batch)
pr = profile.Profile()
pr.enable()
for i in range(10):
    scores = net.classify_direct(lena_batch)
pr.disable()
pr.dump_stats('lena_profile_batch10.pstats')
print 'Profiling done.'
