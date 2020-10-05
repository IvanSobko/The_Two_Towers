#ifndef THE_TWO_TOWERS_CSVWRITER_H
#define THE_TWO_TOWERS_CSVWRITER_H

#include <iostream>
#include <vector>

class CSVWriter {
public:
    CSVWriter(std::string filename, std::string delm = ",");

    void addDataInRow(std::vector<std::string>::iterator first, std::vector<std::string>::iterator last);

private:
    std::string mFileName;
    std::string mDelimeter;
    int mLinesCount;
};


#endif //THE_TWO_TOWERS_CSVWRITER_H
