> FastTT Branch

### Prerequisites

* Make: GNU 'make'.
* Optional: Building with autotools requires autoconf/automake packages.

x86_64:
* Assembler: nasm v2.11.01 or later (nasm v2.13 or better suggested for building in AVX512 support)
  or yasm version 1.2.0 or later.
* Compiler: gcc, clang, icc or VC compiler.

aarch64:
* Assembler: gas v2.24 or later.
* Compiler: gcc v4.7 or later.

other:
* Compiler: Portable base functions are available that build with most C compilers.
* GF-Complete: https://github.com/ceph/gf-complete

### Autotools

docker start ceph_build
docker exec -it -w /ceph/cerasure ceph_build /bin/bash

To build and install the library with autotools it is usually sufficient to run:

    ./autogen.sh
    ./configure
    make LDFLAGS="-Wl,--allow-multiple-definition" CFLAGS="-mavx2 -O3"
    sudo make install

<!-- ### Makefile
To use a standard makefile run:

    make -f Makefile.unx -->

### Other make targets
Other targets include:
* `make ex`    : build examples

### test
    cd raid/
    ./xor_example -k 4 -p 2 -w 8 -s 512
