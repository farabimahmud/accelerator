## Progress

### Finished:
- filter-weight gradient compute added in scale-sim simulator
- transposed convolution (deconvolution) for input gradient in order to propagate back to previous layer as its output gradient (in scale-sim)
- support an array of accelerator instances as a distributed training system
    - data-parallelism support: can be simulated using a single instance
- scale-sim and booksim integration using python-c++ binding (pybind11)
    - build booksim as a library
    - add APIs for communications between scale-sim simulator and booksim network interface

### On-going:
- DRAM BW reflection in scale-sim to support HMC
- scale-sim: add basic protocol message and sequencer/controller to support communication among HMCs

### Todo:
- message passing interfaces for programmability and expressiveness
- flow control support extension for bulk transfer
- active-routing support
    - processing elements in routers/NI and art tree construction
    - sequencer for flit/packet/message processing during transmission
    - tree scheduling
        - level by level from leaves to root
        - aggregate from children to parent for every node recursively until completion (binary tree like, 2x theoretical speedup)
- support an array of accelerator instances as a distributed training system
    - model-parallelism support 
