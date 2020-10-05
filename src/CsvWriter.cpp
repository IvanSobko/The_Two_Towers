#include "CsvWriter.h"

#include <fstream>
#include <vector>


CSVWriter::CSVWriter(std::string filename, std::string delm) :
        mFileName(filename), mDelimeter(delm), mLinesCount(0) {}

void CSVWriter::addDataInRow(std::vector<std::string>::iterator first, std::vector<std::string>::iterator last) {
    std::fstream file;
    file.open(mFileName, std::ios::out | (mLinesCount ? std::ios::app : std::ios::trunc));
    for (;first != last;) {
        file << *first;
        if (++first != last) {
            file << mDelimeter;
        }
    }
    file << "\n";
    mLinesCount++;
    file.close();
}