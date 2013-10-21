import cPickle as pickle
from decaf import puff
import logging
import numpy as np
import numpy.testing as npt
import tempfile
import unittest


class TestPuff(unittest.TestCase):
    def setUp(self):
        pass

    def testPuffVectorForm(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4)
        puff.write_puff(data, fname)
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        npt.assert_array_almost_equal(puff_recovered.read_all(), data)
        self.assertTrue(puff_recovered.read_all().shape, (4,))
        puff_recovered.seek(0)
        npt.assert_array_almost_equal(puff_recovered.read(2), data[:2])
        self.assertTrue(puff_recovered.read(2).shape, (2,))

    def testPuffShardedVectorForm(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4)
        for i in range(3):
            puff.write_puff(data, fname + '-%d-of-3' % i)
        puff_recovered = puff.Puff(fname + '-*-of-3')
        npt.assert_array_almost_equal(puff_recovered.read_all(),
                                      np.hstack([data] * 3))
        self.assertTrue(puff_recovered.read_all().shape, (12,))
        puff_recovered.seek(0)
        npt.assert_array_almost_equal(puff_recovered.read(2), data[:2])
        self.assertTrue(puff_recovered.read(2).shape, (2,))

    def testPuffSingleWrite(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4,3)
        puff.write_puff(data, fname)
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        npt.assert_array_almost_equal(puff_recovered.read_all(), data)

    def testPuffMultipleWrites(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4,3)
        writer = puff.PuffStreamedWriter(fname)
        writer.write_batch(data)
        writer.write_batch(data)
        writer.write_single(data[0])
        writer.finish()
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        data_recovered = puff_recovered.read_all()
        npt.assert_array_almost_equal(data_recovered[:4], data)
        npt.assert_array_almost_equal(data_recovered[4:8], data)
        npt.assert_array_almost_equal(data_recovered[8], data[0])

    def testPuffMultipleWriteException(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4,3)
        writer = puff.PuffStreamedWriter(fname)
        writer.write_batch(data)
        self.assertRaises(
            TypeError, writer.write_batch, data.astype(np.float32))
        self.assertRaises(
            TypeError, writer.write_batch, np.random.rand(4,2))

    def testPuffIteration(self):
        fname = tempfile.mktemp()
        data = np.random.rand(10,3)
        puff.write_puff(data, fname)
        puff_recovered = puff.Puff(fname)
        count = 0
        for elem in puff_recovered:
            count += 1
        self.assertEqual(count, 10)
        for i, elem in zip(range(10), puff_recovered):
            npt.assert_array_almost_equal(data[i], elem)
        # test local slicing
        puff_recovered.set_range(3,7)
        count = 0
        for elem in puff_recovered:
            count += 1
        self.assertEqual(count, 4)
        for i, elem in zip(range(4), puff_recovered):
            npt.assert_array_almost_equal(data[i+3], elem)

    def testPuffShardedIteration(self):
        fname = tempfile.mktemp()
        data = np.random.rand(30,3)
        for i in range(3):
            puff.write_puff(data[i*10:(i+1)*10], fname + '-%d-of-3' % i)
        puff_recovered = puff.Puff(fname + '-*-of-3')
        count = 0
        for elem in puff_recovered:
            count += 1
        self.assertEqual(count, 30)
        for i, elem in zip(range(30), puff_recovered):
            npt.assert_array_almost_equal(data[i], elem)
        # test local slicing
        puff_recovered.set_range(3,17)
        count = 0
        for elem in puff_recovered:
            count += 1
        self.assertEqual(count, 14)
        for i, elem in zip(range(14), puff_recovered):
            npt.assert_array_almost_equal(data[i+3], elem)

    def testPuffReadBoundary(self):
        fname = tempfile.mktemp()
        data = np.random.rand(4,3)
        puff.write_puff(data, fname)
        # Now, let's read it
        puff_recovered = puff.Puff(fname)
        npt.assert_array_almost_equal(puff_recovered.read(3), data[:3])
        npt.assert_array_almost_equal(puff_recovered.read(3),
                                      data[np.array([3,0,1], dtype=int)])
        # test seeking
        puff_recovered.seek(1)
        npt.assert_array_almost_equal(puff_recovered.read(2), data[1:3])

    def testPuffReadSlice(self):
        fname = tempfile.mktemp()
        data = np.random.rand(10,3)
        puff.write_puff(data, fname)
        # Now, let's read it as a slice
        data = data[5:9]
        puff_recovered = puff.Puff(fname, start=5, end=9)
        npt.assert_array_almost_equal(puff_recovered.read(3), data[:3])
        npt.assert_array_almost_equal(puff_recovered.read(3),
                                      data[np.array([3,0,1], dtype=int)])
        # test incorrect seeking
        self.assertRaises(ValueError, puff_recovered.seek, 0)

    def testPuffShardedReadSlice(self):
        fname = tempfile.mktemp()
        data = np.random.rand(30,3)
        for i in range(3):
            puff.write_puff(data[i*10:(i+1)*10], fname + '-%d-of-3' % i)
        data = data[5:27]
        puff_recovered = puff.Puff(fname + '-*-of-3', start=5, end=27)
        npt.assert_array_almost_equal(puff_recovered.read(3), data[:3])
        npt.assert_array_almost_equal(puff_recovered.read_all(), data)
        self.assertRaises(ValueError, puff_recovered.seek, 0)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
