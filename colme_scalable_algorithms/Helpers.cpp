#include "Helpers.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <iterator>
#include <cmath>
#include <limits>
#include <fstream>
#include <filesystem>

namespace Helpers {

    // given an integer number counts the number of digits it is made of
    int count_digits(int number) {
        if (number == 0) { return 1; } // special caso for 0
        return static_cast<int>(std::log10(std::abs(number))) + 1;
    }


    // add zeros to the left so that the string corresponding to 'number' has '
    std::string str_left_zeros(int number, int str_length) {
        std::string number_str = std::to_string(number);
        int numZeros = str_length - static_cast<int>(number_str.length());
        if (numZeros > 0) {
            std::string padding(numZeros, '0');
            number_str = padding + number_str;
        }
        return number_str;
    }


    std::vector<int> changed_indices(const std::vector<bool>& first, const std::vector<bool>& second) {

        std::vector<int> removed_indices{}; // empty vector

        for (int i = 0; i < static_cast<int>(first.size()); ++i) {
            if (first[i] && !second[i]) // when corresponding elements are true in 'first' and false in 'second'
                removed_indices.push_back(i);
        }
        return removed_indices;
    }


    float safe_read(const std::vector<float>& v, size_t index) {
        if (index < v.size())
            return v[index];
        else
            return std::numeric_limits<float>::quiet_NaN(); // Nan if v has no element at 'index'
    }

    int safe_read(const std::vector<int>& v, size_t index) {
        if (index < v.size())
            return v[index];
        else
            return -1; // TODO: problem if the values may be negative
    }


    int print_progress(int cyle_count, int prev_prog, int tot_cycles, std::string pre) {
        
        int prog = (int)floor(((double)cyle_count / (double)tot_cycles) * 100.0);
        if ((prog % 10 == 0) && ((prog / 10) != prev_prog)) {
            std::cout << pre << prog << " % done" << std::endl; // print the progress percentage
            return prog / 10;
        }
        else
            return prev_prog;
    }


    bool are_equal(float a, float b, double tolerance) {
        return std::abs(a - b) < tolerance;
    }


    std::string get_last_line(std::filesystem::path filename) {

        std::ifstream file(filename);

        if (!file) {
            std::cout << "Unable to open file.\n";
            exit(-1);
        }
        
        std::string line;
        while (file >> std::ws && std::getline(file, line)) // skip empty lines
            ;
        
        return line;
    }


    void write_log(std::string filename, std::string line_to_add) {

        std::filesystem::path log_filename = filename; // portable

        if (!std::filesystem::exists(log_filename)) { // check if log already exists, and create it if it does not
            std::ofstream log_file(log_filename);
            if (!log_file) {
                std::cerr << "ERROR: not possible to create log file." << std::endl;
                exit(-1);
            }
            log_file << line_to_add << std::endl; // always add the line if the file had to be created
            log_file.close();
        }

        std::string last_line = get_last_line(log_filename);
        // std::cout << "\t*** " << last_line << std::endl;

        std::fstream log_file(log_filename, std::ios::app); // open in append
        if (!log_file) {
            std::cerr << "Error opening log file." << std::endl;
            exit(-1);
        }

        if (last_line != line_to_add) 
            log_file << line_to_add << std::endl; // add only if different content

        log_file.close();

    }

}
