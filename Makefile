# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robbie/Documents/GEANT/Photon-Processes

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robbie/Documents/GEANT/Photon-Processes

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/robbie/Documents/GEANT/Photon-Processes/CMakeFiles /home/robbie/Documents/GEANT/Photon-Processes/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/robbie/Documents/GEANT/Photon-Processes/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named PhotonProcess

# Build rule for target.
PhotonProcess: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 PhotonProcess
.PHONY : PhotonProcess

# fast build rule for target.
PhotonProcess/fast:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/build
.PHONY : PhotonProcess/fast

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/build
.PHONY : test/fast

Source/Fields/BlackBody.o: Source/Fields/BlackBody.cpp.o

.PHONY : Source/Fields/BlackBody.o

# target to build an object file
Source/Fields/BlackBody.cpp.o:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Fields/BlackBody.cpp.o
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Fields/BlackBody.cpp.o
.PHONY : Source/Fields/BlackBody.cpp.o

Source/Fields/BlackBody.i: Source/Fields/BlackBody.cpp.i

.PHONY : Source/Fields/BlackBody.i

# target to preprocess a source file
Source/Fields/BlackBody.cpp.i:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Fields/BlackBody.cpp.i
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Fields/BlackBody.cpp.i
.PHONY : Source/Fields/BlackBody.cpp.i

Source/Fields/BlackBody.s: Source/Fields/BlackBody.cpp.s

.PHONY : Source/Fields/BlackBody.s

# target to generate assembly for a file
Source/Fields/BlackBody.cpp.s:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Fields/BlackBody.cpp.s
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Fields/BlackBody.cpp.s
.PHONY : Source/Fields/BlackBody.cpp.s

Source/GaussianProcess/GaussianProcess.o: Source/GaussianProcess/GaussianProcess.cpp.o

.PHONY : Source/GaussianProcess/GaussianProcess.o

# target to build an object file
Source/GaussianProcess/GaussianProcess.cpp.o:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/GaussianProcess/GaussianProcess.cpp.o
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/GaussianProcess/GaussianProcess.cpp.o
.PHONY : Source/GaussianProcess/GaussianProcess.cpp.o

Source/GaussianProcess/GaussianProcess.i: Source/GaussianProcess/GaussianProcess.cpp.i

.PHONY : Source/GaussianProcess/GaussianProcess.i

# target to preprocess a source file
Source/GaussianProcess/GaussianProcess.cpp.i:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/GaussianProcess/GaussianProcess.cpp.i
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/GaussianProcess/GaussianProcess.cpp.i
.PHONY : Source/GaussianProcess/GaussianProcess.cpp.i

Source/GaussianProcess/GaussianProcess.s: Source/GaussianProcess/GaussianProcess.cpp.s

.PHONY : Source/GaussianProcess/GaussianProcess.s

# target to generate assembly for a file
Source/GaussianProcess/GaussianProcess.cpp.s:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/GaussianProcess/GaussianProcess.cpp.s
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/GaussianProcess/GaussianProcess.cpp.s
.PHONY : Source/GaussianProcess/GaussianProcess.cpp.s

Source/Processes/BreitWheeler.o: Source/Processes/BreitWheeler.cpp.o

.PHONY : Source/Processes/BreitWheeler.o

# target to build an object file
Source/Processes/BreitWheeler.cpp.o:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/BreitWheeler.cpp.o
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/BreitWheeler.cpp.o
.PHONY : Source/Processes/BreitWheeler.cpp.o

Source/Processes/BreitWheeler.i: Source/Processes/BreitWheeler.cpp.i

.PHONY : Source/Processes/BreitWheeler.i

# target to preprocess a source file
Source/Processes/BreitWheeler.cpp.i:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/BreitWheeler.cpp.i
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/BreitWheeler.cpp.i
.PHONY : Source/Processes/BreitWheeler.cpp.i

Source/Processes/BreitWheeler.s: Source/Processes/BreitWheeler.cpp.s

.PHONY : Source/Processes/BreitWheeler.s

# target to generate assembly for a file
Source/Processes/BreitWheeler.cpp.s:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/BreitWheeler.cpp.s
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/BreitWheeler.cpp.s
.PHONY : Source/Processes/BreitWheeler.cpp.s

Source/Processes/ComptonScatter.o: Source/Processes/ComptonScatter.cpp.o

.PHONY : Source/Processes/ComptonScatter.o

# target to build an object file
Source/Processes/ComptonScatter.cpp.o:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/ComptonScatter.cpp.o
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/ComptonScatter.cpp.o
.PHONY : Source/Processes/ComptonScatter.cpp.o

Source/Processes/ComptonScatter.i: Source/Processes/ComptonScatter.cpp.i

.PHONY : Source/Processes/ComptonScatter.i

# target to preprocess a source file
Source/Processes/ComptonScatter.cpp.i:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/ComptonScatter.cpp.i
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/ComptonScatter.cpp.i
.PHONY : Source/Processes/ComptonScatter.cpp.i

Source/Processes/ComptonScatter.s: Source/Processes/ComptonScatter.cpp.s

.PHONY : Source/Processes/ComptonScatter.s

# target to generate assembly for a file
Source/Processes/ComptonScatter.cpp.s:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/ComptonScatter.cpp.s
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/ComptonScatter.cpp.s
.PHONY : Source/Processes/ComptonScatter.cpp.s

Source/Processes/PhotonProcess.o: Source/Processes/PhotonProcess.cpp.o

.PHONY : Source/Processes/PhotonProcess.o

# target to build an object file
Source/Processes/PhotonProcess.cpp.o:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/PhotonProcess.cpp.o
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/PhotonProcess.cpp.o
.PHONY : Source/Processes/PhotonProcess.cpp.o

Source/Processes/PhotonProcess.i: Source/Processes/PhotonProcess.cpp.i

.PHONY : Source/Processes/PhotonProcess.i

# target to preprocess a source file
Source/Processes/PhotonProcess.cpp.i:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/PhotonProcess.cpp.i
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/PhotonProcess.cpp.i
.PHONY : Source/Processes/PhotonProcess.cpp.i

Source/Processes/PhotonProcess.s: Source/Processes/PhotonProcess.cpp.s

.PHONY : Source/Processes/PhotonProcess.s

# target to generate assembly for a file
Source/Processes/PhotonProcess.cpp.s:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/PhotonProcess.cpp.s
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/PhotonProcess.cpp.s
.PHONY : Source/Processes/PhotonProcess.cpp.s

Source/Processes/PhotonScatter.o: Source/Processes/PhotonScatter.cpp.o

.PHONY : Source/Processes/PhotonScatter.o

# target to build an object file
Source/Processes/PhotonScatter.cpp.o:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/PhotonScatter.cpp.o
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/PhotonScatter.cpp.o
.PHONY : Source/Processes/PhotonScatter.cpp.o

Source/Processes/PhotonScatter.i: Source/Processes/PhotonScatter.cpp.i

.PHONY : Source/Processes/PhotonScatter.i

# target to preprocess a source file
Source/Processes/PhotonScatter.cpp.i:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/PhotonScatter.cpp.i
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/PhotonScatter.cpp.i
.PHONY : Source/Processes/PhotonScatter.cpp.i

Source/Processes/PhotonScatter.s: Source/Processes/PhotonScatter.cpp.s

.PHONY : Source/Processes/PhotonScatter.s

# target to generate assembly for a file
Source/Processes/PhotonScatter.cpp.s:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Processes/PhotonScatter.cpp.s
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Processes/PhotonScatter.cpp.s
.PHONY : Source/Processes/PhotonScatter.cpp.s

Source/Tools/Numerics.o: Source/Tools/Numerics.cpp.o

.PHONY : Source/Tools/Numerics.o

# target to build an object file
Source/Tools/Numerics.cpp.o:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Tools/Numerics.cpp.o
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Tools/Numerics.cpp.o
.PHONY : Source/Tools/Numerics.cpp.o

Source/Tools/Numerics.i: Source/Tools/Numerics.cpp.i

.PHONY : Source/Tools/Numerics.i

# target to preprocess a source file
Source/Tools/Numerics.cpp.i:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Tools/Numerics.cpp.i
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Tools/Numerics.cpp.i
.PHONY : Source/Tools/Numerics.cpp.i

Source/Tools/Numerics.s: Source/Tools/Numerics.cpp.s

.PHONY : Source/Tools/Numerics.s

# target to generate assembly for a file
Source/Tools/Numerics.cpp.s:
	$(MAKE) -f CMakeFiles/PhotonProcess.dir/build.make CMakeFiles/PhotonProcess.dir/Source/Tools/Numerics.cpp.s
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/Source/Tools/Numerics.cpp.s
.PHONY : Source/Tools/Numerics.cpp.s

test.o: test.cpp.o

.PHONY : test.o

# target to build an object file
test.cpp.o:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/test.cpp.o
.PHONY : test.cpp.o

test.i: test.cpp.i

.PHONY : test.i

# target to preprocess a source file
test.cpp.i:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/test.cpp.i
.PHONY : test.cpp.i

test.s: test.cpp.s

.PHONY : test.s

# target to generate assembly for a file
test.cpp.s:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/test.cpp.s
.PHONY : test.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... PhotonProcess"
	@echo "... test"
	@echo "... Source/Fields/BlackBody.o"
	@echo "... Source/Fields/BlackBody.i"
	@echo "... Source/Fields/BlackBody.s"
	@echo "... Source/GaussianProcess/GaussianProcess.o"
	@echo "... Source/GaussianProcess/GaussianProcess.i"
	@echo "... Source/GaussianProcess/GaussianProcess.s"
	@echo "... Source/Processes/BreitWheeler.o"
	@echo "... Source/Processes/BreitWheeler.i"
	@echo "... Source/Processes/BreitWheeler.s"
	@echo "... Source/Processes/ComptonScatter.o"
	@echo "... Source/Processes/ComptonScatter.i"
	@echo "... Source/Processes/ComptonScatter.s"
	@echo "... Source/Processes/PhotonProcess.o"
	@echo "... Source/Processes/PhotonProcess.i"
	@echo "... Source/Processes/PhotonProcess.s"
	@echo "... Source/Processes/PhotonScatter.o"
	@echo "... Source/Processes/PhotonScatter.i"
	@echo "... Source/Processes/PhotonScatter.s"
	@echo "... Source/Tools/Numerics.o"
	@echo "... Source/Tools/Numerics.i"
	@echo "... Source/Tools/Numerics.s"
	@echo "... test.o"
	@echo "... test.i"
	@echo "... test.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

