# $Id $

# Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

LEX = flex
YACC   = bison -y
DEFINE = 
INCPATH = -I. -Iarbiters -Iallocators -Irouters -Inetworks -Ipower -Idsent
CPPFLAGS += -Wall $(INCPATH) $(DEFINE)
CPPFLAGS += -O3
CPPFLAGS += -g
LFLAGS +=

OBJDIR := obj
PROG := booksim

########## DSENT ###########
# Define the directories that will be compiled
DSENT_DIRS_TO_COMPILE := util tech io \
												 model model/timing_graph \
												 model/std_cells \
												 model/electrical \
												 model/electrical/router \
												 model/optical \
												 model/optical_graph \
												 model/network \
												 model/network/ATAC

DSENT_DIRS = $(patsubst %,dsent/%,$(DSENT_DIRS_TO_COMPILE))
DSENT_SRCS = $(foreach dir, $(DSENT_DIRS), $(wildcard $(dir)/*.cc))
DSENT_OBJS = $(DSENT_SRCS:%.cc=%.o)

DEF_FLAGS =
INCLUDE_FLAGS = -Idsent
OPT_FLAGS = -O2 -g
WARN_FLAGS = -pedantic -Wall -W #-Wextra -Werror -Wno-write-strings
CXXFLAGS = $(OPT_FLAGS) $(WARN_FLAGS) $(INCLUDE_FLAGS) $(DEF_FLAGS)

# link libraries
LD_LIBS += -lutil
LD_FLAGS += -Ldsent/libutil

# Other libraries used
LIB_UTIL = dsent/libutil/libutil.a

########### DSENT ##############

# simulator source files
CPP_SRCS = $(wildcard *.cpp) $(wildcard */*.cpp)
CPP_HDRS = $(wildcard *.hpp) $(wildcard */*.hpp)
CPP_DEPS = $(addprefix ${OBJDIR}/,$(notdir $(CPP_SRCS:.cpp=.d)))
CPP_OBJS = $(addprefix ${OBJDIR}/,$(notdir $(CPP_SRCS:.cpp=.o)))

LEX_SRCS = lex.yy.c
LEX_OBJS = ${OBJDIR}/lex.yy.o

YACC_SRCS = y.tab.c
YACC_HDRS = y.tab.h
YACC_OBJS = ${OBJDIR}/y.tab.o

OBJS := $(DSENT_OBJS) $(CPP_OBJS) $(LEX_OBJS) $(YACC_OBJS)

.PHONY: clean

all: $(LIB_UTIL) $(PROG)

$(PROG): $(OBJS) dsent/DSENT.o
	 $(CXX) $(LFLAGS) $(LD_FLAGS) $^ -o $@ $(LD_LIBS)

$(LIB_UTIL):
	$(MAKE) -C dsent/libutil

dsent/DSENT.o: dsent/DSENT.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LEX_SRCS): config.l
	$(LEX) $<

$(YACC_SRCS) $(YACC_HDRS): config.y
	$(YACC) -d $<

$(LEX_OBJS): $(LEX_SRCS) $(YACC_HDRS)
	$(CC) $(CPPFLAGS) -c $< -o $@

$(YACC_OBJS): $(YACC_SRCS)
	$(CC) $(CPPFLAGS) -c $< -o $@

.SECONDEXPANSION:

${OBJDIR}/%.o: %.cpp | $$(@D)
	$(CXX) $(CPPFLAGS) -MMD -c $< -o $@

# rules to compile networks
${OBJDIR}/%.o: networks/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile arbiters
${OBJDIR}/%.o: arbiters/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile allocators
${OBJDIR}/%.o: allocators/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile routers
${OBJDIR}/%.o: routers/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile power classes
${OBJDIR}/%.o: power/%.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR)
	rm -f $(YACC_SRCS) $(YACC_HDRS)
	rm -f $(LEX_SRCS)
	rm -f $(CPP_DEPS)
	rm -f $(OBJS)
	rm -f $(PROG)
	make clean -C dsent

distclean: clean
	rm -rf $(OBJDIR)
	rm -f *~ */*~
	rm -f *.o */*.o
	rm -f *.d */*.d

-include $(CPP_DEPS)
