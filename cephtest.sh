make ex

# ./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0
# ./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 8

# ./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0 -e 1
# ./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0 -e 8
# ./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 8 -e 9

./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0 -e 1 -e 2
./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0 -e 1 -e 8
./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 0 -e 8 -e 9
./raid/xor_example -k 8 -p 3 -l 16 -n 2000 -e 8 -e 9 -e 10