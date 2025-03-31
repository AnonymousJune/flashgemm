sudo cpupower -c all frequency-set -u 3.8GHz
sudo cpupower -c all frequency-set -d 3.8GHz

find . -type f -name "*.txt" -exec rm -f {} +

cd ./flashgemm 
make clean 
make 
sh run.sh

cd ../mkl 
source /home/wangpy/zjw/intel/oneapi/setvars.sh
make
sh run.sh

cd ../openblas 
make clean 
make 
sh run.sh

cd ../blis 
make clean 
make 
sh run.sh

cd ../onednn 
make clean 
make 
sh run.sh

cd ..