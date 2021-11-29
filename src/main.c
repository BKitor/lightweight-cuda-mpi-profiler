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
#include <sys/stat.h>

int lwcmpp_pool_init() {
  lwcmpp_pool_head = calloc(1, sizeof(lwcmpp_pool_t));
  if (NULL == lwcmpp_pool_head) {
    return -1;
  }

  lwcmpp_pool_tail = lwcmpp_pool_head;
  lwcmpp_pool_head->next = NULL;
  lwcmpp_pool_head->size = LWCMPP_POOL_SIZE;
  lwcmpp_pool_head->idx = 0;
  lwcmpp_pool_head->pool = calloc(LWCMPP_POOL_SIZE, sizeof(lwcmpp_data_node_t));
  if (NULL == lwcmpp_pool_head->pool) {
    return -1;
  }

  return 0;
}

void lwcmpp_pool_destroy() {
  lwcmpp_pool_t* pool = lwcmpp_pool_head;
  lwcmpp_pool_t* tmp = NULL;
  while (NULL != pool) {
    free(pool->pool);
    tmp = pool->next;
    free(pool);
    pool = tmp;
  }
}

static inline void init_metrics() {
  // Should probably handle llwcmpp_pool_init's return value
  lwcmpp_pool_init();

  lwcmpp_data_node_t* node_cpu = lwcmpp_pool_alloc_node();
  node_cpu->arrival_time = -1;
  node_cpu->exit_time = -1;
  node_cpu->msize = -1;
  lwcmpp_data_head = node_cpu;
  lwcmpp_data_tail = node_cpu;

  lwcmpp_data_node_t* node_gpu = lwcmpp_pool_alloc_node();
  node_gpu->arrival_time = -1;
  node_gpu->exit_time = -1;
  node_gpu->msize = -1;
  lwcmpp_data_head_gpu = node_gpu;
  lwcmpp_data_tail_gpu = node_gpu;
}

static inline void count_metric_ar(int count, MPI_Datatype datatype, double ar_time, double ex_time, lwcmpp_data_node_t** tail) {
  int32_t msize = get_MPI_message_size(datatype, count);

  lwcmpp_data_node_t* node = lwcmpp_pool_alloc_node();
  (*tail)->next = node;
  node->arrival_time = ar_time;
  node->exit_time = ex_time;
  node->msize = msize;
  (*tail) = node;
}

static inline void cleanup_metrics(double end_time) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  lwcmpp_data_node_t* n = lwcmpp_data_head->next;
  lwcmpp_data_node_t* n_gpu = lwcmpp_data_head_gpu->next;
  
  if(rank==0){
    struct stat st = {0};
    if(stat(LWCMPP_OUT_DIR, &st) == -1){
      mkdir(LWCMPP_OUT_DIR, 0777);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  char f_out_name[256];
  sprintf(f_out_name, LWCMPP_OUT_FILE, rank);
  FILE *f_out = fopen(f_out_name, "w");
  if (NULL == f_out){
    fprintf(stderr, "rank %d could't open file %s\n", rank, f_out_name);
    goto lwcmpp_cleanup_abort;
  }

  fprintf(f_out,"CPU Allareduce times:\n");
  fflush(f_out);
  int count = 0;
  while (n != NULL) {
    count++;
    fprintf(f_out,"%d\t%.6f\t%.6f\t%d\n", rank, n->arrival_time, n->exit_time, n->msize);

    n = n->next;
  }

  fprintf(f_out,"GPU Allareduce times:\n");
  fflush(f_out);
  int count_gpu = 0;
  while (n_gpu != NULL) {
    count_gpu++;
    fprintf(f_out,"GPU: %d\t%.6f\t%.6f\t%d\n", rank, n_gpu->arrival_time, n_gpu->exit_time, n_gpu->msize);
    n_gpu = n_gpu->next;
  }
  
  fflush(stdout);
  PMPI_Barrier(MPI_COMM_WORLD);

  fprintf(stdout, "MPI_Rank %d, of %d, ran for: %.6f counted %d ar, %d gpu ar\n", rank, size, end_time, count, count_gpu);
  fprintf(f_out, "MPI_Rank %d, of %d, ran for: %.6f counted %d ar, %d gpu ar\n", rank, size, end_time, count, count_gpu);

  fclose(f_out);

  lwcmpp_cleanup_abort:
  lwcmpp_pool_destroy();
}

int MPI_Init(int* argc, char*** argv) {
  init_metrics();
  int ret = PMPI_Init(argc, argv);
  lwcmpp_start_time = MPI_Wtime();
  return ret;
}

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
  init_metrics();
  int ret = PMPI_Init_thread(argc, argv, required, provided);
  lwcmpp_start_time = MPI_Wtime();
  return ret;
}

int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  double s_time, e_time;
  int ret;
  lwcmpp_data_node_t** d = &lwcmpp_data_tail;

  s_time = MPI_Wtime();
  ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  e_time = MPI_Wtime();

  if (is_device_pointer(recvbuf))
    d = &lwcmpp_data_tail_gpu;

  count_metric_ar(count, datatype, s_time, e_time, d);

  return ret;
}

int MPI_Finalize(void) {
  double f_time = MPI_Wtime();
  cleanup_metrics(f_time);
  return PMPI_Finalize();
}
