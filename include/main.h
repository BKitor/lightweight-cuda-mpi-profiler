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

 // should be around 30MB x 4proc/node for 1 million entries
#define LWCMPP_POOL_SIZE (1 << 20)
#define LWCMPP_OUT_DIR "./lwcmpp_output"
#define LWCMPP_OUT_FILE "./lwcmpp_output/lwcmpp_out_rank%d.txt"

// MPI functions we plan to profile
int MPI_Init(int* argc, char*** argv);
int MPI_Init_thread(int* argc, char*** argv, int required, int* provided);
int MPI_Finalize(void);

// Collectives
int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
	MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int MPI_Bcast(void* buffer, int count,
	MPI_Datatype datatype, int root, MPI_Comm comm);

double lwcmpp_start_time = -1;
// sizeof = 28B
typedef struct lwcmpp_data_node_t {
	struct lwcmpp_data_node_t* next;
	double arrival_time;
	double exit_time;
	int32_t msize;
} lwcmpp_data_node_t;

lwcmpp_data_node_t* lwcmpp_data_head = NULL;
lwcmpp_data_node_t* lwcmpp_data_tail = NULL;
lwcmpp_data_node_t* lwcmpp_data_head_gpu = NULL;
lwcmpp_data_node_t* lwcmpp_data_tail_gpu = NULL;

typedef struct  lwcmpp_pool_t {
	struct lwcmpp_pool_t* next;
	lwcmpp_data_node_t* pool;
	int32_t size;
	int32_t idx;
} lwcmpp_pool_t;

lwcmpp_pool_t* lwcmpp_pool_head = NULL;
lwcmpp_pool_t* lwcmpp_pool_tail = NULL;
int lwcmpp_pool_init();
void lwcmpp_pool_destroy();

static inline void lwcmpp_pool_extend() {
	lwcmpp_pool_t *p = calloc(1, sizeof(lwcmpp_pool_t));
	p->pool = calloc(LWCMPP_POOL_SIZE, sizeof(lwcmpp_data_node_t));
	p->next = NULL;
	p->size = LWCMPP_POOL_SIZE;
	p->idx = 0;
	lwcmpp_pool_tail->next = p;
	lwcmpp_pool_tail = p;
    printf("alloc pool with size %d and idx %d\n", p->size, p->idx);
	fflush(stdout);
}

static inline lwcmpp_data_node_t* lwcmpp_pool_alloc_node() {
	if (__builtin_expect(lwcmpp_pool_tail->idx > lwcmpp_pool_tail->size, 0)) {
		lwcmpp_pool_extend();
	}

	lwcmpp_data_node_t* ret = &lwcmpp_pool_tail->pool[lwcmpp_pool_tail->idx];
	lwcmpp_pool_tail->idx++;
	return ret;
}

#endif // MAIN_H
