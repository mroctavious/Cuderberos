TARGET=cuderberos
CC=nvcc
CFLAGS=-std=c++11 -O3
LIBS=`pkg-config --libs opencv`
INCLUDE_DIRS=-I./include -I/usr/local/cuda/include `pkg-config --cflags opencv`
SOURCE_DIR=./src
OBJDIR=./build
OBJECTS=Image.o main.o
BINDIR=./bin
ARCH=-DCUDA=1 -DDEBUG=1
#######################################################################
#					DO NOT TOUCH BEYOND THIS LINE					  #
#######################################################################
all: $(TARGET)

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

%.o: $(SOURCE_DIR)/%.cxx
	$(CC) $(ARCH) -x cu $(CFLAGS) $(INCLUDE_DIRS) $(LIBS) -c $< -o $(OBJDIR)/$@

cuderberos: directories $(OBJECTS)
	$(CC) $(ARCH) -o $(BINDIR)/$(TARGET) $(OBJDIR)/* $(CFLAGS) $(INCLUDE_DIRS) $(LIBS) 
clean:
	rm -rf $(OBJDIR) $(BINDIR) 2> /dev/null