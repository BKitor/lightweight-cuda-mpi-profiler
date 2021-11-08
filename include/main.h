/* lightweight-cuda-mpi-profiler is a simple CUDA-Aware MPI profiler
 * Copyright (C) 2021 Yiltan Hassan Temucin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef MAIN_H
#define MAIN_H

#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
#include "cuda_helpers.h"
#include "mpi_helpers.h"

#define LARGE_BSIZE (1<<22) // 4MB

// MPI functions we plan to profile
int MPI_Init(int *argc, char ***argv);
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Finalize(void);

// Collectives
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int MPI_Bcast(void *buffer, int count,
                  MPI_Datatype datatype, int root, MPI_Comm comm);

// Metrics we will profile
#define COLL_COUNT_MAX 32 
typedef struct lwcmp_data_t {
  double msize_times[COLL_COUNT_MAX];
  int64_t msize_counts[COLL_COUNT_MAX];
} lwcmp_data_t;

lwcmp_data_t lwcmp_data;
lwcmp_data_t lwcmp_data_cuda;

#endif // MAIN_H
