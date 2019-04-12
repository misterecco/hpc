for num_elems in 1000 1000000 1000000000
do
  for radius in 3 30 300 3000
  do
    echo "NUM_ELEMS: $num_elems, RADIUS: $radius"
    nvcc stencil.cu -DNUM_ELEMENTS=$num_elems -DRADIUS=$radius -o stencil
    ./stencil
    echo "--------------------------------"
  done
done
