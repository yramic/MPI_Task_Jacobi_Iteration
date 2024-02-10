#pragma once

#ifndef THE_FILE_NAME_H
#define THE_FILE_NAME_H

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <string>
#include <assert.h>
#include <stdio.h>

#include "functions.hpp"

#endif



void PoissonJacobiStencil1D(double *norms, size_t resolution, size_t iterations, int num_proc, int rank)
{
  ///////////////////////////////////////////////////////////////////////////////////////
  // I SET UP COMMUNICATOR
  ///////////////////////////////////////////////////////////////////////////////////////

  double N = resolution;
  double h = 1.0 / (N - 1);
  int ndims = 1;

  // Set Cartesian communicator dimension
  std::array<int, 1> dims = {0};
  MPI_Dims_create(num_proc, ndims, std::data(dims));

  // Let MPI create a Cartesian communicator
  std::array<int, 1> periods = {false};
  int reorder = false;
  MPI_Comm comm_1d;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, std::data(dims), std::data(periods), reorder, &comm_1d);

  // Use MPI_Cart_shift to find neighboring ranks
  int displacement = 1;
  enum DIR : int
  {
    No = 0,
    So = 1
  };
  std::array<int, 2> nb = {-1, -1};
  MPI_Cart_shift(comm_1d, 0, displacement, &nb[DIR::No], &nb[DIR::So]);

  // Assign coordinates to rank
  std::array<int, 1> coord = {-1};
  MPI_Cart_coords(comm_1d, rank, ndims, std::data(coord));

  // Hesitate until all ranks are through MPI set up
  MPI_Barrier(MPI_COMM_WORLD);

  ///////////////////////////////////////////////////////////////////////////////////////
  // II ASSIGN SUB DOMAINS
  ///////////////////////////////////////////////////////////////////////////////////////

  // Determine the number of rows each processor treats (except last processor)
  // and the first row (of global domain) a rank operates on
  int block_height = floor(N / num_proc);
  int first_row = rank * block_height;

  // Generate vectors recvcount and displs for the MPI command MPI_Gatherv
  std::vector<int> recvcount(num_proc, block_height * N); // recvcount: stores number of elements each processor treats
  std::vector<int> displs(num_proc, 0);                   // displs: stores the global index of the first element one processor treats

  // the last processor treats the remainder
  recvcount[num_proc - 1] = (N - (num_proc - 1) * block_height) * N;

  // Find displacements for all ranks (e.g. rank X starts to care about X*blockheight*N#th entry in F)
  int counter = 0;
  for (int i = 0; i < num_proc; i++, counter += block_height * N)
    displs[i] = counter;

  // Store information in array for further processing
  int *displs_arr = &displs[0];
  int *recvcount_arr = &recvcount[0];

  // Highest rank deals with remainder
  if (rank == (num_proc - 1))
    block_height = N - rank * block_height;

  ///////////////////////////////////////////////////////////////////////////////////////
  // III SET UP SOLUTION VECTORS, RHS AND ANAL. SOLUTION
  ///////////////////////////////////////////////////////////////////////////////////////

  // Initialize vectors
  std::vector<double> u_old(N * block_height, 0);
  std::vector<double> u_new(N * block_height, 0);
  std::vector<double> u_temp(N * block_height, 0); // for residual norm
  std::vector<double> f(N * block_height);
  std::vector<double> u_an_block(N * block_height);
  std::vector<double> u_an(N * N);    // for root
  std::vector<double> u_final(N * N); // for root
  std::vector<double> u_pen(N * N);   // for root, for residual norm

  // Establish boundary conditions for lower end of global domain
  if (rank == num_proc - 1)
  {
    int j = N - 1;
    for (int i = 0; i < N; ++i)
    {
      double p = ParticularSolution(i * h, j * h);
      u_old[(j - first_row) * N + i] = p;
      u_new[(j - first_row) * N + i] = p;
    }
  }

  // Initialize RHS
  for (int j = first_row; (j < first_row + block_height); ++j)
    for (int i = 0; i < N; ++i)
    {
      double p = ParticularSolution(i * h, j * h) * 4 * M_PI * M_PI;
      f[(j - first_row) * N + i] = p;
    }

  // Establish analytical solution
  for (int j = first_row; (j < first_row + block_height); ++j)
    for (int i = 0; i < N; ++i)
    {
      double u_part = ParticularSolution(i * h, j * h);
      u_an_block[(j - first_row) * N + i] = u_part;
    }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // IV ITERATION
  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Initialize vectors to store Southern and Northern ghost layers
  std::vector<double> g_N(N, 0);
  std::vector<double> g_S(N, 0);

  for (size_t k = 0; k < iterations; k++)
  {
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // COMMUNICATION ROUND
    ///////////////////////////////////////////////////////////////////////////////////////////////

    if (rank == 0) // upmost slice
    {
      int index = (block_height - 1) * N; // index specifies the starting index of the last row
      MPI_Sendrecv(&u_new[index], N, MPI_DOUBLE, nb[1], 0, &g_S[0], N, MPI_DOUBLE, nb[1], 0, comm_1d, MPI_STATUS_IGNORE);
    }

    // most southern slice
    else if (rank == num_proc - 1)
      MPI_Sendrecv(&u_new[0], N, MPI_DOUBLE, nb[0], 0, &g_N[0], N, MPI_DOUBLE, nb[0], 0, comm_1d, MPI_STATUS_IGNORE);

    else // middle slices
    {
      MPI_Sendrecv(&u_new[0], N, MPI_DOUBLE, nb[0], 0, &g_N[0], N, MPI_DOUBLE, nb[0], 0, comm_1d, MPI_STATUS_IGNORE);
      int index = (block_height - 1) * N; // index specifies the starting index of the last row
      MPI_Sendrecv(&u_new[index], N, MPI_DOUBLE, nb[1], 0, &g_S[0], N, MPI_DOUBLE, nb[1], 0, comm_1d, MPI_STATUS_IGNORE);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // STENCIL UPDATE
    ///////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) // upmost slice
    {
      for (int y = 1; y < block_height - 1; y++) // all but ghost-layer dependent row
      {
        for (int x = 1; x < N - 1; x++)
        {
          double u_w = u_old[y * N + x - 1];
          double u_e = u_old[y * N + x + 1];
          double u_s = u_old[(y + 1) * N + x];
          double u_n = u_old[(y - 1) * N + x];
          double f_c = f[y * N + x];

          double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
          double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);

          u_new[y * N + x] = nominator / denominator;
        }
      }

      int y_last = block_height - 1;
      for (int x = 1; x < N - 1; x++) // ghost-layer dependent row
      {
        double u_w = u_old[y_last * N + x - 1];
        double u_e = u_old[y_last * N + x + 1];
        double u_s = g_S[x];
        double u_n = u_old[(y_last - 1) * N + x];
        double f_c = f[y_last * N + x];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);

        u_new[y_last * N + x] = nominator / denominator;
      }
    }

    else if (rank == num_proc - 1) // most southern slice
    {
      for (int y = 1; y < block_height - 1; y++) // all but ghost-layer dependent row
      {
        for (int x = 1; x < N - 1; x++)
        {
          double u_w = u_old[y * N + x - 1];
          double u_e = u_old[y * N + x + 1];
          double u_s = u_old[(y + 1) * N + x];
          double u_n = u_old[(y - 1) * N + x];
          double f_c = f[y * N + x];

          double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
          double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);

          u_new[y * N + x] = nominator / denominator;
        }
      }

      for (int x = 1; x < N - 1; x++) // ghost-layer dependent row
      {
        double u_w = u_old[x - 1];
        double u_e = u_old[x + 1];
        double u_s = u_old[x + N];
        double u_n = g_N[x];
        double f_c = f[x];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);
        u_new[x] = nominator / denominator;
      }
    }

    else // middle slices
    {
      for (int y = 1; y < block_height - 1; y++) // all but ghost-layer dependent rows
      {
        for (int x = 1; x < N - 1; x++)
        {
          double u_w = u_old[y * N + x - 1];
          double u_e = u_old[y * N + x + 1];
          double u_s = u_old[(y + 1) * N + x];
          double u_n = u_old[(y - 1) * N + x];
          double f_c = f[y * N + x];

          double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
          double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);
          u_new[y * N + x] = nominator / denominator;
        }
      }

      int y_last = block_height - 1;
      for (int x = 1; x < N - 1; x++) // ghost-layer dependent rows
      {
        // SOUTHERN LAYER
        double u_w = u_old[y_last * N + x - 1];
        double u_e = u_old[y_last * N + x + 1];
        double u_s = g_S[x];
        double u_n = u_old[(y_last - 1) * N + x];
        double f_c = f[y_last * N + x];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);
        u_new[y_last * N + x] = nominator / denominator;

        // NORTHERN LAYER
        u_w = u_old[x - 1];
        u_e = u_old[x + 1];
        u_s = u_old[x + N];
        u_n = g_N[x];
        f_c = f[x];

        nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);
        u_new[x] = nominator / denominator;
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SOLUTION UPDATE
    ///////////////////////////////////////////////////////////////////////////////////////////////

    u_temp = u_old;
    u_old = u_new;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // V GATHER INFORMATION TO ROOT
  ///////////////////////////////////////////////////////////////////////////////////////////////

  MPI_Gatherv(&u_an_block[0], N * block_height, MPI_DOUBLE, &u_an[0], recvcount_arr, displs_arr, MPI_DOUBLE, 0, comm_1d);
  MPI_Gatherv(&u_new[0], N * block_height, MPI_DOUBLE, &u_final[0], recvcount_arr, displs_arr, MPI_DOUBLE, 0, comm_1d);
  MPI_Gatherv(&u_temp[0], N * block_height, MPI_DOUBLE, &u_pen[0], recvcount_arr, displs_arr, MPI_DOUBLE, 0, comm_1d);

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // VI RESULT VAILDATION AND OUTPUT OPTIONS
  ///////////////////////////////////////////////////////////////////////////////////////////////

  // root does postprocessing
  if (rank == 0)
  {
    // Compute Norms
    std::vector<double> u_diff(N * N);
    std::transform(u_final.begin(), u_final.end(), u_an.begin(), u_diff.begin(), std::minus<double>());
    auto euc_errorNorm = NormL2(u_diff);
    auto max_errorNorm = NormInf(u_diff);

    std::vector<double> res(N * N);
    const double diag = (4 + 4 * h * h * M_PI * M_PI) / pow(h, 2);
    std::vector<double> A_diag(N * N, diag);
    A_diag[0] = 1 / pow(h, 2);
    A_diag[N * N - 1] = 1 / pow(h, 2);
    for (int i = 0; i < N * N; ++i)
      res[i] = A_diag[i] * (u_pen[i] - u_final[i]);

    auto euc_resNorm = NormL2(res);
    auto max_resNorm = NormInf(res);

    // SET NORMS
    norms[0] = euc_errorNorm;
    norms[1] = max_errorNorm;
    norms[2] = euc_resNorm;
    norms[3] = max_resNorm;

    // OUTPUT look at approx. solution
    // std::cout << "Analytical vs. Numerical Sol \n";
    // for (int i = 0; i < size(u_an); ++i) std::cout << "knot i: " << i << " [u_an, u_num]: " << u_an[i] << " " << u_final[i] <<  std::endl;
  }
}

void PoissonJacobiStencil2D(double *norms, size_t resolution, size_t iterations, int num_proc, int rank)
{
  ///////////////////////////////////////////////////////////////////////////////////////
  // I SET UP COMMUNICATOR
  ///////////////////////////////////////////////////////////////////////////////////////

  double N = resolution;
  double h = 1.0 / (N - 1);
  int ndims = 2;

  // Set Cartesian communicator dimension
  std::array<int, 2> grid_dim = {0, 0};
  MPI_Dims_create(num_proc, ndims, std::data(grid_dim));

  // Let MPI create a Cartesian communicator
  std::array<int, 2> periods = {false, false};
  int reorder = false;
  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, std::data(grid_dim), std::data(periods), reorder, &comm_2d);

  // Use MPI_Cart_shift to find neighboring ranks
  int displacement = 1;
  enum DIR : int
  {
    No = 0,
    So = 1,
    We = 2,
    Ea = 3
  };

  std::array<int, 4> nb = {-2, -2, -2, -2};
  MPI_Cart_shift(comm_2d, 0, displacement, &nb[DIR::No], &nb[DIR::So]);
  MPI_Cart_shift(comm_2d, 1, displacement, &nb[DIR::We], &nb[DIR::Ea]);

  // Assign coordinates to rank
  std::array<int, 2> coord = {-1, -1};
  MPI_Cart_coords(comm_2d, rank, ndims, std::data(coord));

  // Hesitate until all ranks are through MPI set up
  MPI_Barrier(MPI_COMM_WORLD);

  ///////////////////////////////////////////////////////////////////////////////////////
  // II ASSIGN SUB DOMAINS
  ///////////////////////////////////////////////////////////////////////////////////////

  // Determine the number of rows/cols a processor works on
  int block_dim_y = floor(N / grid_dim[0]);
  int block_dim_x = floor(N / grid_dim[1]);

  // We want remainder ranks to know the 'normal' blocks' dimensions
  int block_dim_x_global = block_dim_x;
  int block_dim_y_global = block_dim_y;

  // Remainder assignment
  if (nb[1] == -2)
    block_dim_y = block_dim_y + (int(N) % (grid_dim[0])); // Southern procesors get remainder in y direction
  if (nb[3] == -2)
    block_dim_x = block_dim_x + (int(N) % (grid_dim[1])); // Eastern procesors get remainder in x direction

  // Set up row major displacement and recvcount (solution gather step 1)
  std::vector<int> recvcount(grid_dim[0], 0); // recvcount: stores number of elements each ROW MAJOR treats
  std::vector<int> displs(grid_dim[0], 0);    // displs: stores the global index of the first element a ROW MAJOR treats
  int major_row_count = 0;
  for (int rank_i = 0; rank_i < num_proc; rank_i += grid_dim[1], major_row_count++)
  {
    std::array<int, 2> coord_i = {-1, -1};
    MPI_Cart_coords(comm_2d, rank_i, ndims, std::data(coord_i));
    displs[major_row_count] = coord_i[0] * N * block_dim_y_global;
    recvcount[major_row_count] = N * block_dim_y_global;
    if (rank_i + grid_dim[1] >= num_proc)
      recvcount[major_row_count] = N * N - N * block_dim_y_global * (grid_dim[0] - 1);
  }

  // Store in array data structure to make information usable for MPI
  int *displs_arr = &displs[0];
  int *recvcount_arr = &recvcount[0];

  ///////////////////////////////////////////////////////////////////////////////////////
  // III SET UP SOLUTION VECTORS, RHS AND ANAL. SOLUTION
  ///////////////////////////////////////////////////////////////////////////////////////

  // Initialize common vectors
  std::vector<double> u_old(block_dim_y * block_dim_x, 0), u_new(block_dim_y * block_dim_x, 0), u_temp(block_dim_y * block_dim_x, 0), f(block_dim_y * block_dim_x, 0), u_an_block(block_dim_y * block_dim_x, 0);

  // Initialize root vectors
  std::vector<double> u_an(N * N);
  std::vector<double> u_final(N * N);
  std::vector<double> u_pen(N * N);

  // Iitialize row major vectors
  int row_size_ex_r = block_dim_x * block_dim_y * (grid_dim[1] - 1);     // number of values in row excluding remainder
  int remainder_size = int(N) % grid_dim[1] * block_dim_y;               // number of remainder values
  int remainder_block_size = remainder_size + block_dim_x * block_dim_y; // number of values in block additionally dealing with remainder
  std::vector<double> u_a_row(row_size_ex_r + remainder_block_size, -1);
  std::vector<double> u_final_row(row_size_ex_r + remainder_block_size, -1);
  std::vector<double> u_temp_row(row_size_ex_r + remainder_block_size, -1);

  // Establish boundary conditions for lower end of global domain
  if (nb[1] == -2) // Southern neighbour non existent
  {
    int starting_index = block_dim_x * (block_dim_y - 1); // starting index in local u_old/u_new
    int global_i = coord[1] * block_dim_x_global;         // starting x coordinate in gloabl domain matrix CAM: SWAPPED FROM DIM_X TO DIM_X_GLOBAL
    int global_j = N - 1;                                 // y coordinate in global domain matrix
    for (int i = 0; i < block_dim_x; i++)                 // the whole row
    {
      double p = ParticularSolution(double(global_i + i) * h, double(global_j) * h);
      u_old[starting_index + i] = p;
      u_new[starting_index + i] = p;
    }
  }

  // Find global y-index
  int global_j = coord[0] * block_dim_y;
  if (nb[1] == -2)
    global_j = N - block_dim_y; // check for southern remainder

  // Initialize RHS; u_anal
  int k = 0;
  for (int j = 0; (j < block_dim_y); ++j)
  {
    int global_i = coord[1] * block_dim_x; // find global x-index
    if (nb[3] == -2)
      global_i = N - block_dim_x; // check for eastern remainder
    for (int i = 0; i < block_dim_x; ++i)
    {
      double u_part = ParticularSolution(global_i * h, global_j * h);
      u_an_block[k] = u_part;    // u_anal
      u_part *= 4 * M_PI * M_PI; // RHS
      f[k] = u_part;             // j*block_dimx_x+i
      global_i++;
      k++;
    }
    global_j++;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // IV ITERATION
  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Initialize vectors to store ghost layers
  std::vector<double> g_N(block_dim_x, -3); // change init value to zero in the end!
  std::vector<double> g_S(block_dim_x, -3); // change init value to zero in the end!
  std::vector<double> g_E(block_dim_y, -3); // change init value to zero in the end!
  std::vector<double> g_W(block_dim_y, -3); // change init value to zero in the end!

  //////////////////////////////////////////////////////////////////////////////////////
  //// DATA TYPE FOR HORIZONTAL GHOST LAYER COMMUNICATION
  //////////////////////////////////////////////////////////////////////////////////////

  MPI_Datatype ghost_horizontal;
  MPI_Type_vector(
      block_dim_y, // number of blocks
      1,           // number of meaningful values per block
      block_dim_x, // number of empty mem locations inbetween blocks
      MPI_DOUBLE,
      &ghost_horizontal);
  MPI_Type_commit(&ghost_horizontal);

  for (size_t k = 0; k < iterations; k++)
  {
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // COMMUNICATION ROUND
    ///////////////////////////////////////////////////////////////////////////////////////////////
    int index = (block_dim_y - 1) * block_dim_x; // index specifies the starting index of the last row

    MPI_Sendrecv(&u_new[0], block_dim_x, MPI_DOUBLE, nb[0], 0, &g_N[0], block_dim_x, MPI_DOUBLE, nb[0], 0, comm_2d, MPI_STATUS_IGNORE);     // N
    MPI_Sendrecv(&u_new[index], block_dim_x, MPI_DOUBLE, nb[1], 0, &g_S[0], block_dim_x, MPI_DOUBLE, nb[1], 0, comm_2d, MPI_STATUS_IGNORE); // S

    MPI_Sendrecv(&u_new[0], 1, ghost_horizontal, nb[2], 0, &g_W[0], block_dim_y, MPI_DOUBLE, nb[2], 0, comm_2d, MPI_STATUS_IGNORE);               // W
    MPI_Sendrecv(&u_new[block_dim_x - 1], 1, ghost_horizontal, nb[3], 0, &g_E[0], block_dim_y, MPI_DOUBLE, nb[3], 0, comm_2d, MPI_STATUS_IGNORE); // E

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // STENCIL UPDATE
    ///////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    // UPDATE OF INNER POINTS ONLY
    for (int y = 1; y < block_dim_y - 1; y++)
    {
      for (int x = 1; x < block_dim_x - 1; x++)
      {
        double u_w = u_old[y * block_dim_x + x - 1];
        double u_e = u_old[y * block_dim_x + x + 1];
        double u_s = u_old[(y + 1) * block_dim_x + x];
        double u_n = u_old[(y - 1) * block_dim_x + x];
        double f_c = f[y * block_dim_x + x];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);

        u_new[y * block_dim_x + x] = nominator / denominator;
      }
    }

    // IF THERE IS A PROCESSOR IN WESTERN DIRECTION
    if (nb[2] != -2)
    {
      for (int y = 1; y < block_dim_y - 1; y++)
      {
        double u_w = g_W[y];
        double u_e = u_old[y * block_dim_x + 1];
        double u_s = u_old[(y + 1) * block_dim_x];
        double u_n = u_old[(y - 1) * block_dim_x];
        double f_c = f[y * block_dim_x];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);
        u_new[y * block_dim_x] = nominator / denominator;
      }
    }

    // IF THERE IS A PROCESSOR IN EASTERN DIRECTION
    if (nb[3] != -2)
    {
      for (int y = 1; y < block_dim_y - 1; y++) // ghost-layer dependent row, -2
      {
        double u_w = u_old[(y + 1) * block_dim_x - 2];
        double u_e = g_E[y];
        double u_s = u_old[(y + 2) * block_dim_x - 1];
        double u_n = u_old[(y)*block_dim_x - 1];
        double f_c = f[(y + 1) * block_dim_x - 1];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);
        u_new[(y + 1) * block_dim_x - 1] = nominator / denominator;
      }
    }

    // IF THERE IS A PROCESSOR IN SOUTHERN DIRECTION
    if (nb[1] != -2)
    {
      int x_first = 1;
      int x_last = block_dim_x - 1;

      if (nb[3] != -2)
        x_last++; // PROCESS IN EASTERN DIRECTION
      if (nb[2] != -2)
        x_first--; // PROCESS IN WESTERN DIRECTION

      int y_last = block_dim_y - 1;
      for (int x = x_first; x < x_last; x++)
      {
        double u_w, u_e;
        u_e = u_old[y_last * block_dim_x + x + 1];
        u_w = u_old[y_last * block_dim_x + x - 1];
        double u_s = g_S[x];
        double u_n = u_old[(y_last - 1) * block_dim_x + x];
        double f_c = f[y_last * block_dim_x + x];

        // PROCESS IN SOUTHERN AND EASTERN DIRECTION
        if (nb[3] != -2 && x == block_dim_x - 1)
          u_e = g_E[block_dim_y - 1];

        // PROCESS IN SOUTHERN AND WESTERN DIRECTION
        if (nb[2] != -2 && x == x_first)
          u_w = g_W[block_dim_y - 1];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);

        u_new[y_last * block_dim_x + x] = nominator / denominator;
      }
    }

    // IF THERE IS A PROCESSOR IN NORTHERN DIRECTION
    if (nb[0] != -2)
    {
      int first_x = 1;
      int last_x = block_dim_x - 1;

      if (nb[3] != -2)
        last_x++; // PROCESS IN EASTERN DIRECTION
      if (nb[2] != -2)
        first_x--; // PROCESS IN WESTERN DIRECTION

      for (int x = first_x; x < last_x; x++)
      {
        double u_w, u_e;
        u_w = u_old[x - 1];
        u_e = u_old[x + 1];
        double u_s = u_old[x + block_dim_x];
        double u_n = g_N[x];
        double f_c = f[x];

        // PROCESS IN NORTHERN AND EASTERN DIRECTION
        if (nb[3] != -2 && x == block_dim_x - 1)
          u_e = g_E[0];

        // PROCESS IN NORTHERN AND WESTERN DIRECTION
        if (nb[2] != -2 && x == first_x)
          u_w = g_W[0];

        double nominator = u_w + u_s + u_n + u_e + pow(h, 2) * f_c;
        double denominator = 4 * (pow(M_PI, 2) * pow(h, 2) + 1);
        u_new[x] = nominator / denominator;
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SOLUTION UPDATE
    ///////////////////////////////////////////////////////////////////////////////////////////////
    u_temp = u_old;
    u_old = u_new;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // V GATHER INFORMATION TO ROOT
  ///////////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // V.I GATHER ANALYTICAL INFO TO ROOT
  ///////////////////////////////////////////////////////////////////////////////////////////////

  // STEP 1: Collect values for a whole row at row major (rank with coord [any,0])
  if (coord[1] == 0)
  {
    std::vector<double> u_prank_a(block_dim_x * block_dim_y);
    for (int prank = rank + 1; prank < rank + grid_dim[1] - 1; prank++)
    {
      MPI_Recv(u_prank_a.data(), block_dim_x * block_dim_y, MPI_DOUBLE, prank, 1, comm_2d, MPI_STATUS_IGNORE);

      // plug u_prank elements into u_a_row
      int start_idx_row_vec = (prank % grid_dim[1]) * block_dim_x - 1;
      int k = 0;
      for (int j = 0; j < block_dim_y; j++)
        for (int i = 0; i < block_dim_x; i++, k++)
          u_a_row[start_idx_row_vec + j * int(N) + i] = u_prank_a[k];
    }

    // deal with remainder and own contribution
    int prank_r= rank + grid_dim[1] - 1;
    std::vector<double> u_prank_a_r(remainder_block_size);

    MPI_Recv(u_prank_a_r.data(), remainder_block_size, MPI_DOUBLE, prank_r, 1, comm_2d, MPI_STATUS_IGNORE);

    // plug u_prank_X_r elements into u_X_row
    int start_idx_row_vec = (prank_r % grid_dim[1]) * block_dim_x;
    int k1 = 0;
    int k2 = 0;
    for (int j = 0; j < block_dim_y; j++)
    {
      for (int i = 0; i < remainder_block_size / block_dim_y; i++, k1++)
        u_a_row[start_idx_row_vec + j * int(N) + i] = u_prank_a_r[k1]; // remainder
      for (int i = 0; i < block_dim_x; i++, k2++)
        u_a_row[j * int(N) + i] = u_an_block[k2]; // own contribution
    }
  }

  else
    MPI_Send(&u_an_block[0], block_dim_x * block_dim_y, MPI_DOUBLE, coord[0] * grid_dim[1], 1, comm_2d); // sending ranks

  // Step 2a: Create 1D Communicator in order to Gather across row majors
  MPI_Comm comm_row_major;
  int color = MPI_UNDEFINED;
  if (coord[1] == 0)
    color = 1;
  MPI_Comm_split(comm_2d, color, coord[0], &comm_row_major); // creates sub communicator from existing one

  // Step 2b: Gather across row majors
  if (rank % grid_dim[1] == 0)
    MPI_Gatherv(&u_a_row[0], u_a_row.size(), MPI_DOUBLE, &u_an[0], recvcount_arr, displs_arr, MPI_DOUBLE, 0, comm_row_major);

  /////////////////////////////////////////////////////////////////////////////////
  // V.II GATHER NUMERICAL SOLUTION TO ROOT ==> SAME AS V.I
  /////////////////////////////////////////////////////////////////////////////////

  MPI_Barrier(MPI_COMM_WORLD);

  if (coord[1] == 0)
  {
    std::vector<double> u_prank_f(block_dim_x * block_dim_y);
    for (int prank = rank + 1; prank < rank + grid_dim[1] - 1; prank++)
    {
      MPI_Recv(u_prank_f.data(), block_dim_x * block_dim_y, MPI_DOUBLE, prank, 2, comm_2d, MPI_STATUS_IGNORE);
      int start_idx_row_vec = (prank % grid_dim[1]) * block_dim_x - 1;
      int k = 0;
      for (int j = 0; j < block_dim_y; j++)
        for (int i = 0; i < block_dim_x; i++, k++)
          u_final_row[start_idx_row_vec + j * int(N) + i] = u_prank_f[k];
    }
    int prank_r = rank + grid_dim[1] - 1;
    std::vector <double> u_prank_f_r(remainder_block_size);
    MPI_Recv(u_prank_f_r.data(), remainder_block_size, MPI_DOUBLE, prank_r, 2, comm_2d, MPI_STATUS_IGNORE);
    int start_idx_row_vec = (prank_r % grid_dim[1]) * block_dim_x;
    int k1 = 0;
    int k2 = 0;
    for (int j = 0; j < block_dim_y; j++)
    {
      for (int i = 0; i < remainder_block_size / block_dim_y; i++, k1++)
        u_final_row[start_idx_row_vec + j * int(N) + i] = u_prank_f_r[k1];
      for (int i = 0; i < block_dim_x; i++, k2++)
        u_final_row[j * int(N) + i] = u_new[k2];
    }
  }
  else
    MPI_Send(&u_new[0], block_dim_x * block_dim_y, MPI_DOUBLE, coord[0] * grid_dim[1], 2, comm_2d);
  if (rank % grid_dim[1] == 0)
    MPI_Gatherv(&u_final_row[0], u_final_row.size(), MPI_DOUBLE, &u_final[0], recvcount_arr, displs_arr, MPI_DOUBLE, 0, comm_row_major);

  /////////////////////////////////////////////////////////////////////////////////
  // V.III GATHER 'OLD' SOLUTION TO ROOT (for residual norm) ==> SAME AS V.I
  /////////////////////////////////////////////////////////////////////////////////

  MPI_Barrier(MPI_COMM_WORLD);

  if (coord[1] == 0)
  {
    std::vector <double> u_prank_t (block_dim_x * block_dim_y);
    for (int prank = rank + 1; prank < rank + grid_dim[1] - 1; prank++)
    {
      MPI_Recv(u_prank_t.data(), block_dim_x * block_dim_y, MPI_DOUBLE, prank, 2, comm_2d, MPI_STATUS_IGNORE);
      int start_idx_row_vec = (prank % grid_dim[1]) * block_dim_x - 1;
      int k = 0;
      for (int j = 0; j < block_dim_y; j++)
        for (int i = 0; i < block_dim_x; i++, k++)
          u_temp_row[start_idx_row_vec + j * int(N) + i] = u_prank_t[k];
    }
    int prank_r = rank + grid_dim[1] - 1;
    std::vector <double> u_prank_t_r(remainder_block_size);
    MPI_Recv(u_prank_t_r.data(), remainder_block_size, MPI_DOUBLE, prank_r, 2, comm_2d, MPI_STATUS_IGNORE);
    int start_idx_row_vec = (prank_r % grid_dim[1]) * block_dim_x;
    int k1 = 0;
    int k2 = 0;
    for (int j = 0; j < block_dim_y; j++)
    {
      for (int i = 0; i < remainder_block_size / block_dim_y; i++, k1++)
        u_temp_row[start_idx_row_vec + j * int(N) + i] = u_prank_t_r[k1];
      for (int i = 0; i < block_dim_x; i++, k2++)
        u_temp_row[j * int(N) + i] = u_temp[k2];
    }
  }
  else
    MPI_Send(&u_temp[0], block_dim_x * block_dim_y, MPI_DOUBLE, coord[0] * grid_dim[1], 2, comm_2d);
  if (rank % grid_dim[1] == 0)
    MPI_Gatherv(&u_temp_row[0], u_temp_row.size(), MPI_DOUBLE, &u_pen[0], recvcount_arr, displs_arr, MPI_DOUBLE, 0, comm_row_major);

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // VI RESULT VAILDATION AND OUTPUT OPTIONS
  ///////////////////////////////////////////////////////////////////////////////////////////////

  MPI_Barrier(MPI_COMM_WORLD);

  // root does postprocessing
  if (rank == 0)
  {
    // Error Norms
    std::vector<double> u_diff(N * N);
    std::transform(u_final.begin(), u_final.end(), u_an.begin(), u_diff.begin(), std::minus<double>());
    auto euc_errorNorm = NormL2(u_diff);
    auto max_errorNorm = NormInf(u_diff);

    // Res Norms
    std::vector<double> res(N * N, 0);
    const double diag = (4 + 4 * h * h * M_PI * M_PI) / pow(h, 2);
    std::vector<double> A_diag(N * N, diag);
    A_diag[0] = 1 / pow(h, 2);
    A_diag[N * N - 1] = 1 / pow(h, 2);
    for (int i = 0; i < N * N; ++i)
      res[i] = A_diag[i] * (u_pen[i] - u_final[i]);
    auto euc_resNorm = NormL2(res);
    auto max_resNorm = NormInf(res);

    // SET NORMS
    norms[0] = euc_errorNorm;
    norms[1] = max_errorNorm;
    norms[2] = euc_resNorm;
    norms[3] = max_resNorm;

    // OUTPUT look at approx. solution
    // std::cout << "Analytical vs. Numerical Sol \n";
    // for (int i = 0; i < size(u_an); ++i)
    // {
    //   std::cout << "knot i: " << i << " [u_an, u_num]: " << u_an[i] << " " << u_final[i] << std::endl;
    // }
  }
}