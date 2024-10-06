/*
 * AscFile.cpp
 * 
 * Implementation of defined functions.
 */
#include "AscFile.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

AscFile::~AscFile() {
    if (data == nullptr) { return; } 

    // Free the allocated memory
    for (auto ndx = 0; ndx < NROWS; ndx++) {
        delete [] data[ndx];
    }
    delete [] data;
}

// Check that the contents fo the ASC file are correct. Returns TRUE if any errors are found, which 
// are enumerated in the string provided.
bool AscFileManager::checkAscFile(AscFile* file, std::string* errors) {
    
    // Check values that must be set
    if (file->NCOLS == AscFile::NOT_SET) { *errors += "number of columns is not set;"; }
    if (file->NROWS == AscFile::NOT_SET) { *errors += "number of rows is not set;"; }
    if (file->CELLSIZE == AscFile::NOT_SET) { *errors += "cell size is not set;"; }

    // The coordinate to fix the raster should either be the lower-left, or the center
    if (file->XLLCENTER == AscFile::NOT_SET && file->YLLCENTER == AscFile::NOT_SET &&
        file->XLLCORNER == AscFile::NOT_SET && file->YLLCORNER == AscFile::NOT_SET) {
        *errors += "no location provided for raster coordinate;";
    }
    if (file->XLLCENTER != AscFile::NOT_SET && file->YLLCENTER != AscFile::NOT_SET &&
        file->XLLCORNER != AscFile::NOT_SET && file->YLLCORNER != AscFile::NOT_SET) {
        *errors += "conflicting raster coordinates;";
    }

    // Return true if errors were found
    return (!errors->empty());
}

// Read the indicated file from disk, caller is responsible for checking if 
// data is integer or floating point.
AscFile* AscFileManager::read(const std::string& fileName) {
    // Treat the struct as POD
    auto* results = new AscFile();

    // Open the file and verify it
    std::string field, value;
    std::ifstream in(fileName);
    if (!in.good()) {
        throw std::runtime_error("Error opening ASC file: " + fileName);
    }
    if (in.peek() == std::ifstream::traits_type::eof()) {
	throw std::runtime_error("EOF encountered at start of the file: " + fileName);
    } 
    
    // Read the first six lines of the header
    for (auto ndx = 0; ndx < 6; ndx++) {

        // Read the field and value, cast to upper case
        in >> field >> value;
        std::transform(field.begin(), field.end(), field.begin(), ::toupper);

        // Store the values
        if (field == "NCOLS") {
            results->NCOLS = std::stoi(value);
        } else if (field == "NROWS") {
            results->NROWS = std::stoi(value);
        } else if (field == "XLLCENTER") {
            results->XLLCENTER = std::stod(value);
        } else if (field == "YLLCENTER") {
            results->YLLCENTER = std::stod(value);
        } else if (field == "XLLCORNER") {
            results->XLLCORNER = std::stod(value);
        } else if (field == "YLLCORNER") {
            results->YLLCORNER = std::stod(value);
        } else if (field == "CELLSIZE") {
            results->CELLSIZE = std::stod(value);
        } else if (field == "NODATA_VALUE") {
            results->NODATA_VALUE = std::stod(value);
        }
    }

    // Check the header to make sure it is valid
    auto* errors = new std::string();
    if (checkAscFile(results, errors)) {
        throw std::runtime_error(*errors);
    }
    delete errors;

    // Allocate the memory and read the remainder of the actual raster data
    results->data = new float*[results->NROWS];
    for (auto ndx = 0; ndx < results->NROWS; ndx++) {
        results->data[ndx] = new float[results->NCOLS];
    }

    // Remainder of the file is the actual raster data
    for (auto ndx = 0; ndx < results->NROWS; ndx++) {
        for (auto ndy = 0; ndy < results->NCOLS; ndy++) {
            // If the file is malformed then we may encounter the EOF before reading all the data
            if (in.eof()) {
                throw std::runtime_error("EOF encountered while reading data.");
            }
            in >> value;
            results->data[ndx][ndy] = std::stof(value);
        }
    }

    // Clean-up and return
    in.close();
    return results;
}

// Write the contents of the AscFile to disk.
void AscFileManager::write(AscFile* file, const std::string& fileName) {

    // Open the file for writing
    std::ofstream out(fileName);

    // Write the header
    out << std::setprecision(16) << std::left
        << std::setw(HEADER_WIDTH) << "ncols" << file->NCOLS << AscFile::CRLF
        << std::setw(HEADER_WIDTH) << "nrows" << file->NROWS << AscFile::CRLF
        << std::setw(HEADER_WIDTH) << "xllcorner" << file->XLLCORNER << AscFile::CRLF
        << std::setw(HEADER_WIDTH) << "yllcorner" << file->YLLCORNER << AscFile::CRLF
        << std::setw(HEADER_WIDTH) << "cellsize" << file->CELLSIZE << AscFile::CRLF
        << std::setw(HEADER_WIDTH) << "NODATA_value" << file->NODATA_VALUE << AscFile::CRLF;

    // Write the raster data
    for (auto ndx = 0; ndx < file->NROWS; ndx++) {
        for (auto ndy = 0; ndy < file->NCOLS; ndy++) {
            out << std::setprecision(8) << file->data[ndx][ndy] << " ";
        }
        out << AscFile::CRLF;
    }

    // Clean-up
    out.close();
}
