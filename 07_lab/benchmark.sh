for size in 1 10 100 1000 10000 100000 1000000 10000000 100000000
do
  for i in $(seq 0 29)
  do 
    mpiexec -np 48 benchmark-latency.exe $i $size      
  done
done
