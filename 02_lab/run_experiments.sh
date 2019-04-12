make pi

for steps in 100 1000 10000
do
  for threads in 2 4 8 16 32
  do
    echo "Steps: $steps, threads: $threads"
    OMP_NUM_THREADS=$threads ./pi $steps
    echo "--------------------------------"
  done
done
