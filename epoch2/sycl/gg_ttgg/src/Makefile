INCFLAGS = -I.
OPTFLAGS = -O3
CXXFLAGS = $(OPTFLAGS) -std=c++11 $(INCFLAGS) $(USE_NVTX) -Wall -Wshadow $(MGONGPU_CONFIG)
LIBDIR   = ../lib
LIBFLAGS = -L$(LIBDIR) -l$(MODELLIB)
CXX     ?= g++


# Assuming uname is available, detect if architecture is power
UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),ppc64le)
    CUFLAGS+= -Xcompiler -mno-float128
endif

target=$(LIBDIR)/libmodel_sm.a
cxx_objects=Parameters_sm.o read_slha.o rambo.o 
cu_objects= # NB grambo.cu must be included by gcheck.cu (no rdc)

all: $(target)

debug: OPTFLAGS = -g -O0 -DDEBUG2
debug: CUFLAGS := $(filter-out -lineinfo,$(CUFLAGS))
debug: CUFLAGS += -G
debug: $(target)

# NB: cuda includes are needed in the C++ code for curand.h
%.o : %.cc *.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CUINC) -c $< -o $@

#%.o : %.cu *.h
#	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c $< -o $@

$(target): $(cxx_objects) $(cu_objects)
	if [ ! -d $(LIBDIR) ]; then mkdir $(LIBDIR); fi
	$(AR) cru $@ $(cxx_objects) $(cu_objects)
	ranlib $(target)

.PHONY: clean

clean:
	rm -f $(target)
	rm -f $(cxx_objects) $(cu_objects)
