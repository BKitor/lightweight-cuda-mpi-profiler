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
#define COLL_COUNT_MAX 32 //32MB
int ar_arr[COLL_COUNT_MAX];
int bc_arr[COLL_COUNT_MAX];

// Functions to count metrics
static inline void init_metrics() {
  for(int i = 0; i<COLL_COUNT_MAX; i++){
    ar_arr[i] = 0;
    bc_arr[i] = 0;
  }

  printf("initialized arrays of size %ld\n",sizeof(ar_arr)/sizeof(int));
  // Add other metrics
}
static inline void count_metric_bc(int count,
                                 MPI_Datatype datatype) {
  int buffer_size = get_MPI_message_size(datatype, count);

  // see: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  // bit criptic, but essentialy rounds down 
  //buffer_size--;
  buffer_size |= buffer_size>>1;
  buffer_size |= buffer_size>>2;
  buffer_size |= buffer_size>>4;
  buffer_size |= buffer_size>>8;
  buffer_size |= buffer_size>>16;
  buffer_size++;
  buffer_size>>=1;

  if(buffer_size == 0)
    bc_arr[0]++;
  else
    for(int i = 1; i<COLL_COUNT_MAX; i++)
      if(1<<i & buffer_size)bc_arr[i]++;


  // Add other metrics
}

static inline void count_metric_ar(int count,
                                 MPI_Datatype datatype) {
  int buffer_size = get_MPI_message_size(datatype, count);

  // see: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  // bit criptic, but essentialy rounds down 
  //buffer_size--;
  buffer_size |= buffer_size>>1;
  buffer_size |= buffer_size>>2;
  buffer_size |= buffer_size>>4;
  buffer_size |= buffer_size>>8;
  buffer_size |= buffer_size>>16;
  buffer_size++;
  buffer_size>>=1;

  if(buffer_size == 0)
    ar_arr[0]++;
  else
    for(int i = 1; i<COLL_COUNT_MAX; i++)
      if(1<<i & buffer_size)ar_arr[i]++;


  // Add other metrics
}

static inline void print_metrics() {
  printf("Allreudce sizes:\n");
  for(int i = 0; i<COLL_COUNT_MAX; i++)
    printf("%d\t%d\n",1<<(i-1), ar_arr[i]);

  printf("Broadcast sizes:\n");
  for(int i = 0; i<COLL_COUNT_MAX; i++)
    printf("%d\t%d\n",1<<(i-1), bc_arr[i]);
  

  // Add other metrics
}

#endif // MAIN_H
