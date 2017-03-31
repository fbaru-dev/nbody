CXX = pgc++ 
#icpc
OPTFLAGS = -fast 
DEBUGFLAGS = -g
ACCFLAGS = -acc -ta=tesla:cc60 -Minfo=accel
INCLUDES = 

CXXFLAGS = $(OPTFLAGS) $(DEBUGFLAGS) $(ACCFLAGS)

SOURCES = GSimulation.cpp main.cpp
.SUFFIXES: .o .cpp

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=nbody.x

all: gpu

%.o: %.cpp
	$(info )
	$(info Compiling the object file for GPU: )
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

gpu: $(OBJSC)
	$(info )
	$(info Linking the GPU executable:)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC)
	
run: 
	$(info )
	$(info Run the default test case on GPU: )
	./nbody.x 
	
clean: 
	rm -f $(OBJSC) nbody.x 
