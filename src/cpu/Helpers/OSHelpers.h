/*
 * OSHelpers.hxx
 *
 * This header contains some useful functions that are intended to be cross-platform compatible.
 *
 * Code to get the physical and virtual memory used adapted from an answer on StackOverflow (CC BY-SA 4.0)
 * https://stackoverflow.com/a/64166/1185
 */

#ifndef OS_HXX
#define OS_HXX

#include <fstream>

class OsHelpers {
  private:
    static const int ERROR = -1;              // An error occurred during the operation
    static const int NOT_SUPPORTED = -2;      // The operation is not supported by this OS

    // Parse a line from /proc/self/status to get the integer value
    static int parseProcLine(char * line);

    // Return the defined value in /proc/self/status
    static int getProcValue(const char * name, size_t length);

  public:
    // Check to see if the file indicated exists.
    static bool file_exists(const std::string &name);

    // Get the physical memory that is currently used by this process, negative value indicates an error.
    static int getPhysicalMemoryUsed();

    // Get the virtual memory that is currently used by this process, negative value indicates an error.
    static int getVirtualMemoryUsed();
};

#endif
