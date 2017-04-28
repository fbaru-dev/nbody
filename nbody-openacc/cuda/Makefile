CXX = pgc++ 
#icpc
OPTFLAGS = -fast 
DEBUGFLAGS = -g
ACCFLAGS = -acc -ta=tesla:cc60 -Minfo=accel 
INCLUDES = 

CXXFLAGS = $(OPTFLAGS) $(DEBUGFLAGS) $(ACCFLAGS)

SOURCES = main.cpp 
SOURCE_ACC = GSimulation.cpp
SOURCE_CUDA = GSimulation-cuda.cu 
.SUFFIXES: .o .cpp .cu

##########################################
OBJSC = $(SOURCES:.cpp=.o)
OBJSACC = $(SOURCE_ACC:.cpp=.o)
OBJSCUDA = $(SOURCE_CUDA:.cu=.o)
##########################################

EXEC=nbody.x

all: cuda

%.o: %.cu
	nvcc -arch sm_35 -c $< -o $@ 

%.o: %.cpp
	$(info )
	$(info Compiling the object file for GPU: )
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

cuda: $(OBJSC) $(OBJSCUDA)
	$(info )
	$(info Linking the GPU executable:)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Mcuda -o $(EXEC) $(OBJSC) $(OBJSCUDA)

acc: $(OBJSC) $(OBJSACC)
	$(info )
	$(info Linking the GPU executable:)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC) $(OBJSACC)
	
run: 
	$(info )
	$(info Run the default test case on GPU: )
	./nbody.x 
	
clean: 
	rm -f *.o nbody.x
