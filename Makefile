#lightweight-cuda-mpi-profiler is a simple CUDA-Aware MPI profiler
#Copyright (C) 2021 Yiltan Hassan Temucin
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

CC=mpicc

ROOT_DIR := ${CURDIR}
LDLIBS=-L$(ROOT_DIR)/build/lib
LDFLAGS=-llwcmp

all: build/obj/.d build/lib/.d
	$(CC) -Iinclude/ -c src/main.c -o build/obj/main.o
	$(CC) -shared -o build/lib/liblwcmp.so build/obj/main.o

test: all build/bin/.d build/lib/liblwcmp.so
	$(CC) -Iinclude/ $(LDLIBS) $(LDFLAGS) \
		test/test.c -o build/bin/test

check: test
	LD_LIBRARY_PATH=$(ROOT_DIR)/build/lib:$(LD_LIBRARY_PATH) \
	mpirun -np 4 $(ROOT_DIR)/build/bin/test

clean:
	rm -rf build/*

%/.d:
	mkdir -p $(@D)
	touch $@
