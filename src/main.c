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

#include "main.h"

static inline void init_metrics() {
  for(int32_t i = 0; i<COLL_COUNT_MAX; i++){
    lwcmp_data.msize_counts[i] = 0;
    lwcmp_data.msize_times[i] = 0.0;
    lwcmp_data_cuda.msize_counts[i] = 0;
    lwcmp_data_cuda.msize_times[i] = 0.0;
  }
}

static inline void count_metric_ar(int count, MPI_Datatype datatype, double time, lwcmp_data_t *data) {
  int32_t msize = get_MPI_message_size(datatype, count);
  
  int idx = __builtin_clz(msize);
  idx = 32 - idx;
  data->msize_counts[idx]++;
  data->msize_times[idx]+= time;
}

static inline void print_metrics(lwcmp_data_t * d, int32_t size) {
  char *suffix[] = {"B", "KB", "MB", "GB", "TB"};
  char length = sizeof(suffix) / sizeof(suffix[0]);
  static char output[256];

  for(int64_t msize = 0; msize<COLL_COUNT_MAX; msize++){
    int32_t suf_idx = 0;
    int64_t bytes = ((int64_t)1)<<msize;
    bytes >>= 1;
    int64_t dblBytes = bytes;

    if (bytes > 1024) {
      for (suf_idx = 0; (bytes / 1024) > 0 && suf_idx<length-1; suf_idx++, bytes /= 1024)
        dblBytes = bytes / 1024.0;
    }

    printf("%ld%s\tcount:%-*ld time:%.2f us\n", 
      dblBytes, suffix[suf_idx],
      10, d->msize_counts[msize]/size, 
      d->msize_times[msize]*1e6/(double)size);
  }
}

static inline void cleanup_metrics(double f_time){
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank != 0){
    PMPI_Reduce(lwcmp_data.msize_times, NULL, 32, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    PMPI_Reduce(lwcmp_data.msize_counts, NULL, 32, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    PMPI_Reduce(lwcmp_data_cuda.msize_times, NULL, 32, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    PMPI_Reduce(lwcmp_data_cuda.msize_counts, NULL, 32, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  }else{
    PMPI_Reduce(MPI_IN_PLACE, lwcmp_data.msize_times, 32, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    PMPI_Reduce(MPI_IN_PLACE, lwcmp_data.msize_counts, 32, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    PMPI_Reduce(MPI_IN_PLACE, lwcmp_data_cuda.msize_times, 32, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    PMPI_Reduce(MPI_IN_PLACE, lwcmp_data_cuda.msize_counts, 32, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    printf("LWMP CPU METRICS:\n");
    print_metrics(&lwcmp_data, size);
    printf("LWMP GPU METRICS:\n");
    print_metrics(&lwcmp_data_cuda, size);
  }
  
}

int MPI_Init(int *argc, char ***argv) {
  init_metrics();
  return PMPI_Init(argc, argv);
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
  init_metrics();
  return PMPI_Init_thread(argc, argv, required, provided);
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  double s_time, e_time;
  int ret;
  lwcmp_data_t *d = &lwcmp_data;

  s_time = MPI_Wtime();
  ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  e_time = MPI_Wtime();
  
  if(is_device_pointer(recvbuf))
    d = &lwcmp_data_cuda;

  count_metric_ar(count, datatype, e_time - s_time, d);

  return ret;
}

int MPI_Finalize(void) {
  double f_time = MPI_Wtime();
  cleanup_metrics(f_time);
  return PMPI_Finalize();
}