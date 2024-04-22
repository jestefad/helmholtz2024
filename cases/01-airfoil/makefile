target := vdf.x

ifndef MPI_ENABLE
MPI_ENABLE := 1
endif

ifndef fmath
fmath := 1
endif

ifndef cuda
cuda := 1
endif

ifeq (${MPI_ENABLE},1)
cc_h := $(shell which mpicxx)
else
cc_h := $(shell which g++)
endif

compflags :=
compflags += -DMPI_ENABLE=${MPI_ENABLE}

hflags :=
hflags += -fconcepts-diagnostics-depth=3 -fmax-errors=1 -Werror=return-type -Wcpp
ifeq (${sanny},1)
hflags += -fsanitize=undefined,address -fstack-protector-all
endif
ifeq (${gee},1)
hflags += -g
endif

hflags += -std=c++20 -O3
hflags += ${compflags}
flags = ${hflags}

dflags :=
dflags += -x cu
dflags += -std=c++20 -O3 #-prec-div=false
dflags += -ccbin=${cc_h}
dflags += --extended-lambda
#dflags += -Xptxas -v
#dflags += --maxrregcount 96
dflags += -w

ifeq (${gee},1)
dflags += -lineinfo
endif

ifeq (${fmath},1)
dflags += --use_fast_math
endif

#dflags += --expt-relaxed-constexpr
#dflags += --verbose

cc_d = $(shell which nvcc)

cc := ${cc_h}
flags := ${hflags}
ifeq (${cuda},1)
cc = ${cc_d}
flags := ${dflags} -Xcompiler "${hflags}"
endif
main:
	${cc} ${flags} -Ilocal -I${SCIDF}/src -I${SPADE}/src main.cc -o ${target}

run: main
	./${target}

clean:
	rm -f ${target}
