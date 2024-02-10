#include <assert.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <mpi.h>

#include "solver.hpp"
// #include "functions.hpp"

int main(int argc, char **argv)
{

  std::string dimension;
  int resolution;
  int iterations;

  // parse command line arguments; else set default resolution and iterations
  if (argc != 4)
  {
    dimension = "1D";
    resolution = 125;
    iterations = 10000;
    std::cout << "Wrong number of arguments" << std::endl;
    std::cout << "Decomposition = 1D and resolution = 125 and iteration = 10000 (default)" << std::endl;
  }
  else
  {
    dimension = argv[1];
    resolution = std::stoi(argv[2]);
    iterations = std::stoi(argv[3]);
  }

  // Break if parsed arguments are unreasonable
  assert((resolution > 0) && "Negative resolution!");
  assert((iterations > 0) && "Negative iterations!");
  assert(((dimension == "1D") || (dimension == "2D")) && "Invalid dimension!");

  // initiate time-taking variables
  int sample_size = 1;
  int warm_up = 0;
  std::vector<double> runtimes(sample_size, 0);
  double mean_runtime[1]; // store mean runtime in array[1] for use in ALLREDUCE
  double start, stop;

  // initiate MPI
  MPI_Init(&argc, &argv);
  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Check conditions for 2D
  bool flag_2d = true;

  // Prime number check
  // ARE THERE OTHER CONDITIONS FOR 2D DECOMP?

  bool prime = true;
  for (int i = 2; i < num_proc / 2 + 1; ++i)
  { // ADDED +1 to avoid special case with 4
    if (num_proc % i == 0)
    {
      prime = false;
      break;
    }
  }

  if (prime)
  {
    flag_2d = false;
  }

  if ((flag_2d == false) && (dimension == "2D"))
  {
    dimension = "1D";
    if (rank == 0)
    {
      std::cout << "Conditions not fulfilled for 2D. Defaulting to 1D decomposition!" << std::endl;
    }
  }

  // init array to store norm to rank
  double norms[4];

  // execute jacobisolver and measure runtime
  if (dimension == "2D")
  {
    for (int i = 0; i < sample_size; i++)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

      start = MPI_Wtime();
      PoissonJacobiStencil2D(norms, resolution, iterations, num_proc, rank);
      stop = MPI_Wtime();
      runtimes[i] = stop - start;

      MPI_Barrier(MPI_COMM_WORLD); // just to be safe; best practice HPC lecture
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  else
  {
    for (int i = 0; i < sample_size; i++)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

      start = MPI_Wtime();
      PoissonJacobiStencil1D(norms, resolution, iterations, num_proc, rank);
      stop = MPI_Wtime();
      runtimes[i] = stop - start;

      MPI_Barrier(MPI_COMM_WORLD); // just to be safe; best practice HPC lecture
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  // Establish mean runtime for each rank (excluding first #warm-up runs)
  mean_runtime[0] = 1.0 * std::accumulate(runtimes.begin() + warm_up, runtimes.end(), 0.0) / (runtimes.size() - warm_up);

  // Allreduce runtimes with MAX operator (since runtime = MAX(runtime i) for i processes)
  MPI_Allreduce(MPI_IN_PLACE, mean_runtime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // TIME OUTPUT
  if (rank == 0)
    std::cout << "resolution: " << resolution << " iterations: " << iterations
              << std::scientific
              << " t/iteration: " << mean_runtime[0] / iterations
              << " |error|= " << norms[0] << " |errorMax|= " << norms[1]
              << " |residual|= " << norms[2] << " |residualMax|= " << norms[3]
              << std::endl;

  MPI_Finalize();
  return 0;
}