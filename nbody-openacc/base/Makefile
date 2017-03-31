CXX = pgc++ 
#icpc
OPTFLAGS = -fast 
DEBUGFLAGS = -g
ACCFLAGS = -acc tesla:cc60 -Minfo=accel
INCLUDES = 

CXXFLAGS = $(OPTFLAGS) $(DEBUGFLAGS) $(ACCFLAGS)

SOURCES = GSimulation.cpp main.cpp
#SOURCES = sim.cpp struct.cpp
.SUFFIXES: .o .cpp

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=nbody.x

all: gpu

%.o: %.cpp
	$(info )
	$(info Compiling the object file for CPU: )
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

gpu: $(OBJSC)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC)
	
run: 
	$(info )
	$(info Run the default test case on CPU: )
	./nbody.x 
	
clean: 
	rm -f $(OBJSC) nbody.x *.optrpt
