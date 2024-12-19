#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <filesystem>

namespace Helpers {

    // given an integer number counts the number of digits it is made of
    int count_digits(int number);


    // add zeros to the left so that the string corresponding to 'number' has '
    std::string str_left_zeros(int number, int str_length);


    /* Given two boolean vectors check which indices have changed from true (in the first vector) to false
    *  (in the second vector). This indicates the corresponding elements has been removed.
    * PARAMETERS:
    *  - first : vector at previous time instant
    *  - second: updated vector
    */
    std::vector<int> changed_indices(const std::vector<bool>& first, const std::vector<bool>& second);

    
    /* Returns the value at index 'index' otherwise returns Nan (std::numeric_limits<float>::quiet_NaN())
    * PARAMETERS:
    *  - v    : vector of elements of template type T passed by reference
    *  - index: the index of the element we would like to access
    */
    float safe_read(const std::vector<float>& v, size_t index);


    int safe_read(const std::vector<int>& v, size_t index);


    // returns the value (integer division) with wich the current progress needs to be updated
    int print_progress(int cyle_count, int prev_prog, int tot_cycles, std::string pre="\t");

    // float "==" comparison
    bool are_equal(float a, float b, double tolerance = 1e-9);


    void write_log(std::string filename, std::string line_to_add);

    // inefficient for very long log files (as it scans the entire file)
    std::string get_last_line(std::filesystem::path filename);


    /********************************************** Template Functions ************************************************/

    // Overload of the operator << to easily print std::vector, needs to be declared here
    template <typename T>
    std::ostream& operator<< (std::ostream& out, const std::vector<T>& v){
        if (!v.empty())
        {
            out << '[';
            std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
            out << "]";
        }
        return out;
    }


    /* Search 'element' in v, returns the index if it finds it and -1 otherwise (ok as index are positive)
    * PARAMETERS:
    *  - v   : vector of elements of template type T passed by reference
    * - node: if present as value in v gets removed (return 0) if element not found (return -1)
    *
    * COMPLEXITY: O(n), with n size of the vector v passed
    */
    template <typename T>
    int vector_find(const std::vector<T>& v, const T& element) {
        auto it = std::find(v.begin(), v.end(), element);
        if (it != v.end()) { // element found, return the index
            return static_cast<int>(std::distance(v.begin(), it));
        }
        else
            return -1; // element not found
    }


    /* Deletes 'node' from 'v' vector passed by reference, returns the index of the (original) 'node' element
    * PARAMETERS:
    *  - v   : vector of elements of template type T passed by reference
    * - node: if present as value in v gets removed (return 0) if element not found (return -1)
    *
    * COMPLEXITY: O(n), with n size of the vector v passed
    */
    template<typename T>
    int swap_delete(std::vector<T>& v, T node) {

        auto it = std::find(v.begin(), v.end(), node); // search for an element in std::vector

        if (it != v.end()) {
            int index = static_cast<int>(std::distance(v.begin(), it));  // index where 'node' is located in v

            // swap is used in order to avoid the moving of all the elemnts down (would take O(N))
            // This can be done only if order is NOT important and if the elements are unique
            std::swap(*it, v.back());
            v.pop_back();

            return index;
        }
        else // element not present
            return -1;
    }


    /* Deletes the element at 'index'. IMPORTANT: uses swap, to be used only if order is not important
    * terminates execution if the vector is empty or the index is invalid, returns the elimintaed element
    * PARAMETERS:
    *  - v   : vector of elements of template type T passed by reference
    * - index: index in the vector
    *
    * COMPLEXITY: O(1), thanks to the swapping procedure and as it is not needed to search for the element
    */
    template <typename T>
    T swap_delete_by_index(std::vector<T>& v, int index) { 
        if (!v.empty() && index >= 0 && index < static_cast<int>(v.size())) { // check if the index is valid

            typename std::vector<T>::iterator it = v.begin() + index; // obtain the iterator from the index

            std::iter_swap(it, v.end() - 1); // swap the element at index with the last element
            T removed_element = v.back();
            v.pop_back();                    // remove the last element (which was at index) from the vector
            return removed_element;
        }
        else { // invalid index or empty vector
            std::cout << "ERROR[c++helpers]: invalid index or empty vector" << std::endl;
            exit(-1);
        }
    }

    /* Finds and returns the index of the (first occurence) of the maximum in the vector 'v'
    * PARAMETERS:
    *  - v   : vector of elements of template type T passed by reference
    */
    template <typename T>
    int find_index_max(const std::vector<T>& v) {

        if (v.empty()) {
            std::cout << "WARNING: cannot find the index of the max, vector is empty" << std::endl;
            return -1;
        }
        auto max_e_iter = std::max_element(v.begin(), v.end());
        return static_cast<int>(std::distance(v.begin(), max_e_iter));
    }


    /* 
      in a template vector find the index associated with the second biggest element in the vector
    */
    template <typename T>
    int find_index_second_max(const std::vector<T>& v) {
        if (v.size() < 2) {
            std::cout << "WARNING: cannot find the index of the second max, vector has less than two elements" << std::endl;
            return -1;
        }

        // index of the maximum element
        int max_index = find_index_max(v);

        std::vector<T> v_without_max = v; // copy the vector
        v_without_max.erase(v_without_max.begin() + max_index);    // remove the maximum

        // find the index of the second maximum element in the modified vector
        auto second_max_iter = std::max_element(v_without_max.begin(), v_without_max.end());
        int second_max_index = static_cast<int>(std::distance(v_without_max.begin(), second_max_iter));

        if (second_max_index >= max_index) { // adjust to account for the removal of the max
            second_max_index++;
        }

        return second_max_index;
    }


    /* Returns (by value) a std::vector with only the elements who have a 'true' in the 'mask'
    * PARAMETERS:
    *  - v   : vector of elements of template type T passed by reference
    *  - mask: boolean mask for the vector v, true: keep the value, false: drop it
    * COMPLEXITY: O(n), I have to iterate over the (mask) vector
    */
    template <typename T>
    std::vector<T> filter_by_mask(const std::vector<T>& v, const std::vector<bool>& mask) {
        std::vector<T> filtered_v;

        if (v.size() != mask.size()) { // ensure sizes of v and mask are the same
            std::cout << "ERROR[c++helpers] size of v and mask do not match" << std::endl;
            exit(-1);
        }

        for (size_t i = 0; i < mask.size(); ++i) { // add elements according to mask
            if (mask[i])
                filtered_v.push_back(v[i]);
        }
        return filtered_v;
    }


    // similar as above but returns the index of the 'true' flags
    template <typename T>
    std::vector<int> index_by_mask(const std::vector<T>& v, const std::vector<bool>& mask) {
        std::vector<int> filtered_idx;

        if (v.size() != mask.size()) { // ensure sizes of v and mask are the same
            std::cout << "ERROR[c++helpers] size of v and mask do not match" << std::endl;
            exit(-1);
        }

        for (int i = 0; i < static_cast<int>(mask.size()); ++i) { // add elements according to mask
            if (mask[i])
                filtered_idx.push_back(i);
        }
        return filtered_idx;
    }


    /* Adds (push_back) an element into v or updates the value v[k] if it is populated
    * PARAMETERS:
    *  - v    : vector of elements of template type T passed by reference
    *  - index: index in the vector at which the element is intended to be added (if index>v.size() v is zero-padded)
    *  - value: value to add
    */
    template <typename T>
    void safe_add_or_update(std::vector<T>& v, int index, const T value) {
        if (index >= 0) {
            if (static_cast<size_t>(index) >= v.size()) {
                v.resize(index + 1); // Resize to ensure space for the element, 0-pad the vector
            }
            v[index] = value; // add or update the element at index
        }
    }


    /* In a bidimensional vector (std::vector<std::vector<>>) swaps vector at 'index' with an empty vector
    * PARAMETERS:
    *  - matrix: bidimenional vector of elements of template type T passed by reference
    *  - index : index in the vector at which the element is intended to be added (if index>v.size() v is zero-padded)
    */
    template <typename T>
    void replace_with_empty(std::vector<std::vector<T>>& matrix, size_t index) {
        if (index < matrix.size()) {
            matrix[index] = std::vector<T>();
        }
        else {
            std::cout << "ERROR[c++helpers]: index eceeds matrix (first) size" << std::endl;
            exit(-1);
        }
    }


}
