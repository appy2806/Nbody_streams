# Makefile — nbody_streams/tree_gpu  (GPU Barnes-Hut tree code, shared-lib)
# Targets:
#   make          → build libtreeGPU.so
#   make clean    → remove objects
#   make clean_all→ remove objects + library
#
# Optional flags:
#   FAST_MATH=1   → add -use_fast_math  (default: off for better accuracy)
#   VERBOSE=1     → echo compile commands

# ---------------------------------------------------------------------------
# Toolchain detection
# ---------------------------------------------------------------------------
CXX      ?= g++
FAST_MATH ?= 1

NVCC_PATH := $(shell command -v nvcc 2>/dev/null || true)
ifeq ($(NVCC_PATH),)
  $(error nvcc not found in PATH. Install CUDA or add nvcc to PATH.)
endif
NVCC    := $(NVCC_PATH)
CUDA_TK := $(shell dirname $$(dirname $(NVCC)))

# ---------------------------------------------------------------------------
# GPU architecture detection
# ---------------------------------------------------------------------------
# Detect GPU compute capability:
#   1. nvidia-smi (preferred — works on bare-metal and most clusters)
#   2. python3 + CuPy (fallback for WSL2 / environments where nvidia-smi is absent)
#   3. Hard-coded sm_80 (Ampere) as last resort — supported by CUDA 11.1+
_RAW_CC  := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' 2>/dev/null)
ifeq ($(_RAW_CC),)
  _RAW_CC := $(shell python3 -c "import cupy as cp; print(cp.cuda.Device(0).compute_capability)" 2>/dev/null | tr -d '.')
endif
GPU_CC   := $(if $(_RAW_CC),$(_RAW_CC),80)

# Build gencode flags for each unique compute capability found on the system.
# Also always emit PTX for forward compatibility with future GPUs.
_ALL_CCS := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u)
ifeq ($(_ALL_CCS),)
  _ALL_CCS := $(shell python3 -c "import cupy as cp; [print(cp.cuda.Device(i).compute_capability) for i in range(cp.cuda.runtime.getDeviceCount())]" 2>/dev/null | tr -d '.' | sort -u)
endif
ifeq ($(_ALL_CCS),)
  _ALL_CCS := 80
endif

GENCODE_FLAGS := $(foreach cc,$(_ALL_CCS),-gencode=arch=compute_$(cc),code=sm_$(cc))
# PTX for forward compat (uses the highest CC found)
GENCODE_FLAGS += -gencode=arch=compute_$(GPU_CC),code=compute_$(GPU_CC)

$(info )
$(info === tree_gpu build ===)
$(info CUDA toolkit : $(CUDA_TK))
$(info GPU(s) found : $(_ALL_CCS))
$(info Primary arch : sm_$(GPU_CC))
$(info ===========================)
$(info )

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------
CPP_STD  ?= -std=c++17
OFLAGS   ?= -O3 -g
CXXFLAGS ?= $(OFLAGS) $(CPP_STD) -fPIC -Wall -Wstrict-aliasing=2

NVCCFLAGS ?= -Xptxas=-v \
             $(GENCODE_FLAGS) \
             -Xcompiler -fPIC \
             --expt-relaxed-constexpr \
             $(CPP_STD) \
             -Wno-deprecated-declarations

ifeq ($(FAST_MATH),1)
  NVCCFLAGS += -use_fast_math
  $(info FAST_MATH enabled)
endif

NVCCLDFLAGS ?= -lcudart -lstdc++

# ---------------------------------------------------------------------------
# Sources  (no treecode.cu — this is a library, not an executable)
# ---------------------------------------------------------------------------
CUDASRC = buildTree.cu \
          computeMultipoles.cu \
          makeGroups.cu \
          computeForces.cu \
          treeGPU_interface.cu

CUOBJ   = $(CUDASRC:%.cu=%.o)
LIB     = libtreeGPU.so

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all lib clean clean_all
all: lib

lib: $(LIB)

$(LIB): $(CUOBJ)
	@echo "Linking $@"
	$(NVCC) -shared $(NVCCFLAGS) -o $@ $^ $(NVCCLDFLAGS) \
	  -Xlinker -rpath -Xlinker $(CUDA_TK)/lib64

# Per-file compile rules
%.o: %.cu
	@echo "Compiling $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# computeForces gets flush-to-zero for better performance on subnormals
computeForces.o: computeForces.cu
	@echo "Compiling $< (ftz=true)"
	$(NVCC) $(NVCCFLAGS) -ftz=true -c $< -o $@

clean:
	$(RM) $(CUOBJ)

clean_all: clean
	$(RM) $(LIB)

$(info )
