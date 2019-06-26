## Progress

###Finished:
- filter-weight gradient compute added in scale-sim simulator
- transposed convolution (deconvolution) for input gradient in order to propagate back to previous layer as its output gradient (in scale-sim)

###On-going:
- DRAM BW reflection in scale-sim to support HMC
- support an array of accelerator instances as a distributed training system

###Todo:
- scale-sim and booksim integration using python-c++ binding
     - build booksim as a library
     - add APIs for communications between scale-sim simulator and booksim network interface
- message passing interfaces for programmability and expressiveness
- flow control support extension and active-routing support