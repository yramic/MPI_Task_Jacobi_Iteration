compile: functions.hpp solver.hpp mpijacobi.cpp
	mpic++ -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math mpijacobi.cpp -o mpijacobi

run:
	mpirun -n 4 ./mpijacobi 2D 125 500