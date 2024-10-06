/*
 * OSHelpers.cpp
 *
 * Implement the defined helper functions. Note that -1 is  our default error code while -2 indicates the operation was
 * not supported.
 */
#include "OSHelpers.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

bool OsHelpers::file_exists(const std::string &name) {
  const std::ifstream f(name.c_str());
  return f.good();
}

int OsHelpers::getPhysicalMemoryUsed() {
  int result = NOT_SUPPORTED;
  #if __linux__
  result = getProcValue("VmRSS:", 6);
  #endif
  return result;
}

int OsHelpers::getVirtualMemoryUsed() {
  int result = NOT_SUPPORTED;
  #if __linux__
  result = getProcValue("VmSize:", 7);
  #endif
  return result;
}

int OsHelpers::parseProcLine(char *line) {
  // This assumes that a digit will be found and the line ends with " Kb"
  int value = static_cast<int>(strlen(line));
  const char* ch = line;
  while (*ch < '0' || *ch > '9') { ch++; }
  line[value - 3] = '\0';
  value = atoi(ch);
  return value;
}

int OsHelpers::getProcValue(const char *name, size_t length) {
  FILE* file = fopen("/proc/self/status", "r");
  int result = ERROR;
  char line[128];

  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, name, length) == 0) {
      result = parseProcLine(line);
      break;
    }
  }
  fclose(file);
  return result;
}
