#!/bin/bash

# First, do single-machine test
echo 'Testing single-machine mode...'
nosetests -v *.py

# Second, test when there is no mpi
echo 'Simulating no mpi case...'
PYTHONPATH_SAV=$PYTHONPATH
PYTHONPATH=$PWD/nompi:$PYTHONPATH
nosetests -v *.py
PYTHONPATH=$PYTHONPATH_SAV

# third, test when mpi exists.
echo 'Testing mpi cases...'
for i in {1..5}
do
    echo "Testing mpi with $i nodes..."
    mpirun -n $i nosetests *.py
done


