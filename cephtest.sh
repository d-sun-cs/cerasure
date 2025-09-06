# make ex
make ex CFLAGS="-mavx2 -O3 -w"

./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0

./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0 -e 1

./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0 -e 1 -e 2

# test packetsize