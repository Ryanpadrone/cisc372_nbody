FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

GCC  = gcc
NVCC = nvcc

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(GCC) $(FLAGS) -c $< 

compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -x cu -c $< 

cpu_nbody: nbody_cpu.o compute_cpu.o
	$(GCC) $(FLAGS) $^ -o $@ $(LIBS)

nbody_cpu.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(GCC) $(FLAGS) -c $< -o nbody_cpu.o

compute_cpu.o: compute_cpu.c config.h vector.h $(ALWAYS_REBUILD)
	$(GCC) $(FLAGS) -c $<




clean:
	rm -f *.o nbody
