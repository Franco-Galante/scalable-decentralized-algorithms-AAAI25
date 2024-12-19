#include "Graph.hpp"
#include <string>
#include <chrono>
#include <fstream>


std::chrono::nanoseconds get_time();

void save_info_to_csv(const std::string& filename, const std::string& header, const std::string& data);

void save_w(std::vector<std::vector<float>> matrix, int seed_p, int n_nodes_p);

std::string clust_header(int n_nodes);



int main() {

    std::string csv_header = "alg,graph,seed,n,int_avg_deg,sigma,alpha,up_to,epsilon,delta,init_s,alg_s";

    
    std::vector<int> seed_list = {111, 222, 333, 444, 555, 666, 777, 888, 999, 101,
                                  202, 303, 404, 505, 606, 707, 808, 909, 123, 456};

    // std::string noise_distribution = "unif"; // "none" to have the "perfect" case
    std::string noise_distribution = "none";

    // constant values across all the experiments
    int n_nodes    = 1000;
    int n_max_iter = 2000;
    
    std::vector<float> sigma_vec = {2.0f};

    float epsilon = 0.1f;
    float delta   = 0.1f;

    std::vector<std::string> alg_vec = {"belief_propagation_v1", "consensus", "colme", "colme_recompute"};
    
    std::string g = "gnr";

    // specifics of the underlying collaborative graph
    std::vector<float> p_vec = {0.01f};

    // belief propagation
    std::vector<int> tab_ut_vec = {5};

    // consensus 
    std::vector<float> alpha_vec = {0.5f};

    int avg_case_alg = 1; // lenght of the vector params of the algorithms

    int tot_cases = static_cast<int>(alg_vec.size() * sigma_vec.size() * p_vec.size() * avg_case_alg * seed_list.size());
    int prev_prog = -1;
    int cases = 0;
    int diameter_guess = 10; // set quite arbitrarily (for memory allocation efficiency purpose)

    for (std::string algorithm : alg_vec){

        for (float sigma : sigma_vec) {

            for (float p : p_vec) {

                int avg_deg_ub = static_cast<int>(std::ceil(n_nodes * p)); // upper bound

                float alpha = 0.5f; // default
                int tab_ut  = 10;   // default
                
                if (algorithm == "belief_propagation_v1") {
                    
                    for (int i=0; i < static_cast<int>(tab_ut_vec.size()); i++) {

                        for (int seed : seed_list) {
                            
                            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                            Graph net = Graph(algorithm, noise_distribution, epsilon, delta, n_nodes, avg_deg_ub, p, seed,
                                                 diameter_guess, {0.0, 1.0}, sigma, alpha, tab_ut_vec[i]);
                            if (g=="gnp")
                                net.gnp_generator();

                            else if (g=="tree")
                                net.regular_tree_generator(avg_deg_ub);

                            else if (g=="gnr") {
                                try {
                                    int r_val = avg_deg_ub;
                                    net.gnr_generator2(r_val);
                                }
                                catch (const std::runtime_error& e) {
                                    std::cerr << "Exception caught (runtime): " << e.what() << std::endl;
                                    std::cout << "\tskip this seed and try to generate with a different seed!" << std::endl;
                                    break; // skip this seed and go to the next one
                                }
                            }
                            else {
                                std::cout << "ERROR: unsupported graph type" << std::endl;
                                exit(1);
                            }

                            net.save_graph();

                            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                            std::chrono::milliseconds init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                            double init_s_time = static_cast<double>(init_duration.count()) / 1000.0; // convert ms to s

                            start = std::chrono::steady_clock::now();
                            net.run_estimation_agorithm(n_max_iter, {0.0f, 1.0f}, sigma, true, seed, avg_deg_ub);
                            end = std::chrono::steady_clock::now();
                            std::chrono::milliseconds alg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                            double alg_s_time = static_cast<double>(alg_duration.count()) / 1000.0; // convert ms to s

                            cases++; // update program progress
                            prev_prog = Helpers::print_progress(cases, prev_prog, tot_cases);

                            std::string csv_line_com = algorithm + "," + g + "," + std::to_string(seed) + "," + std::to_string(n_nodes) + "," + std::to_string(avg_deg_ub) +
                                                    "," + std::to_string(sigma) + "," + std::to_string(alpha) + "," + std::to_string(tab_ut_vec[i]) +
                                                    "," + std::to_string(epsilon) + "," + std::to_string(delta);

                            // I group in the same filename the networks that have the same number of nodes
                            // so at lest the file has the same number of columns (maybe I can do smth better)
                            std::string n_filename = "N_" + std::to_string(n_nodes) + "_clusters";
                            std::string csv_clust_header = "alg,graph,seed,n,int_avg_deg,sigma,alpha,up_to,epsilon,delta," + clust_header(n_nodes);
                            std::string csv_clust_line = csv_line_com; // copy
                            std::vector<int> clust_id = net.read_clusters();
                            for (int c_idx=0; c_idx < static_cast<int>(clust_id.size()); c_idx++) {
                                csv_clust_line += ("," + std::to_string(clust_id[c_idx]));
                            }

                            save_info_to_csv("time.csv", csv_header, csv_line_com + "," + std::to_string(init_s_time) + "," + std::to_string(alg_s_time));
                            save_info_to_csv(n_filename + ".csv", csv_clust_header, csv_clust_line);
                        }
                    }
                }
                else if (algorithm == "consensus") {

                    for (int i=0; i < static_cast<int>(alpha_vec.size()); i++) {

                        for (int seed : seed_list) {

                            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                            Graph net = Graph(algorithm, noise_distribution, epsilon, delta, n_nodes, avg_deg_ub, p, seed, diameter_guess, {0.0, 1.0}, sigma, alpha_vec[i], tab_ut);
                            
                            if (g=="gnp")
                                net.gnp_generator();

                            else if (g=="tree")
                                net.regular_tree_generator(avg_deg_ub);

                            else if (g=="gnr") {
                                try {
                                    int r_val = avg_deg_ub;
                                    net.gnr_generator2(r_val);
                                }
                                catch (const std::runtime_error& e) {
                                    std::cerr << "Exception caught (runtime): " << e.what() << std::endl;
                                    std::cout << "\tskip this seed and try to generate with a different seed!" << std::endl;
                                    break; // skip this seed and go to the next one
                                }
                            }
                            else {
                                std::cout << "ERROR: unsupported graph type" << std::endl;
                                exit(1);
                            }          

                            net.save_graph();                  
                            
                            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                            std::chrono::milliseconds init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                            double init_s_time = static_cast<double>(init_duration.count()) / 1000.0; // convert ms to s
                            
                            start = std::chrono::steady_clock::now();
                            net.run_estimation_agorithm(n_max_iter, {0.0f, 1.0f}, sigma, true, seed, avg_deg_ub);
                            end = std::chrono::steady_clock::now();
                            std::chrono::milliseconds alg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                            double alg_s_time = static_cast<double>(alg_duration.count()) / 1000.0; // convert ms to s

                            cases++; // update program progress
                            prev_prog = Helpers::print_progress(cases, prev_prog, tot_cases);

                            std::string csv_line_com = algorithm + "," + g + "," + std::to_string(seed) + "," + std::to_string(n_nodes) + "," + std::to_string(avg_deg_ub) +
                                                    "," + std::to_string(sigma) + "," + std::to_string(alpha_vec[i]) + "," + std::to_string(tab_ut) +
                                                    "," + std::to_string(epsilon) + "," + std::to_string(delta);
                            
                            
                            std::string n_filename = "N_" + std::to_string(n_nodes) + "_clusters";
                            std::string csv_clust_header = "alg,graph,seed,n,int_avg_deg,sigma,alpha,up_to,epsilon,delta," + clust_header(n_nodes);
                            std::string csv_clust_line = csv_line_com; // copy
                            std::vector<int> clust_id = net.read_clusters();
                            for (int c_idx=0; c_idx < static_cast<int>(clust_id.size()); c_idx++) {
                                csv_clust_line += ("," + std::to_string(clust_id[c_idx]));
                            }

                            save_info_to_csv("time.csv", csv_header, csv_line_com + "," + std::to_string(init_s_time) + "," + std::to_string(alg_s_time));
                            save_info_to_csv(n_filename + ".csv", csv_clust_header, csv_clust_line); // this should take care of putting everything in the right file and create it if necessary
                        }
                    }
                }
                else if ((algorithm == "colme") || (algorithm == "colme_recompute")) {

                    for (int seed : seed_list) {

                        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                        Graph net = Graph(algorithm, noise_distribution, epsilon, delta, n_nodes, avg_deg_ub, p, seed, diameter_guess, {0.0, 1.0}, sigma, alpha, tab_ut);
                        net.gnp_generator(); // these case do not have an underlying graph (all generators implement a full mesh for them)
                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                        std::chrono::milliseconds init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                        double init_s_time = static_cast<double>(init_duration.count()) / 1000.0; // convert ms to s

                        start = std::chrono::steady_clock::now();
                        net.run_estimation_agorithm(n_max_iter, {0.0f, 1.0f}, sigma, true, seed, avg_deg_ub);
                        end = std::chrono::steady_clock::now();
                        std::chrono::milliseconds alg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                        double alg_s_time = static_cast<double>(alg_duration.count()) / 1000.0; // convert ms to s


                        cases++; // update program progress
                        prev_prog = Helpers::print_progress(cases, prev_prog, tot_cases);

                        std::string csv_line_com = algorithm + "," + g + "," + std::to_string(seed) + "," + std::to_string(n_nodes) + "," + std::to_string(avg_deg_ub) +
                                                    "," + std::to_string(sigma) + "," + std::to_string(alpha) + "," + std::to_string(tab_ut) +
                                                    "," + std::to_string(epsilon) + "," + std::to_string(delta);

                        std::string n_filename = "N_" + std::to_string(n_nodes) + "_clusters";
                        std::string csv_clust_header = "alg,graph,seed,n,int_avg_deg,sigma,alpha,up_to,epsilon,delta," + clust_header(n_nodes);
                        std::string csv_clust_line = csv_line_com; // copy
                        std::vector<int> clust_id = net.read_clusters();
                        for (int c_idx=0; c_idx < static_cast<int>(clust_id.size()); c_idx++) {
                            csv_clust_line += ("," + std::to_string(clust_id[c_idx]));
                        }

                        save_info_to_csv("time.csv", csv_header, csv_line_com + "," + std::to_string(init_s_time) + "," + std::to_string(alg_s_time));
                        save_info_to_csv(n_filename + ".csv", csv_clust_header, csv_clust_line);
                    }
                }
                else {
                    std::cout << "FATAL ERROR: unrecognized algorithm" << std::endl;
                    exit(-1);
                }
            }
        }
    }
    return 0;
}


// Some helpers functions

std::chrono::nanoseconds get_time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch());
}


// save all info (in particular the empirical probabilities)
void save_info_to_csv(const std::string& filename, const std::string& header, const std::string& data) {

    std::ofstream file;

    // check if the file exists, and if not, create it
    std::ifstream check_file(filename);
    bool file_exists = check_file.good();
    check_file.close();

    if (!file_exists) {
        file.open(filename); // create the file
        if (!file.is_open()) {
            std::cout << "ERROR: Cannot create file " << filename << std::endl;
            exit(-1);
        }
        file << header << "\n"; // write header
    } else {
        file.open(filename, std::ios::app); // open the file in append mode

        if (!file.is_open()) {
            std::cout << "ERROR: Cannot open file " << filename << std::endl;
            exit(-1);
        }
    }

    file << data << "\n"; // write data
    file.close();
}


void save_w(std::vector<std::vector<float>> matrix, int seed_p, int n_nodes_p) {

    std::string filename = "matrix_w_data_" + std::to_string(seed_p) + "_" + std::to_string(n_nodes_p) + ".csv";
    std::ofstream file(filename);

    if (file.is_open()) { // Check if the file is open

        for (const auto& row : matrix) {
            for (const auto& value : row) {
                file << value << ','; // Separate values by commas
            }
            file << '\n'; // Move to the next line for the next row
        }

        file.close();
        std::cout << "\tMatrix data saved successfully.\n";

    } else {
        std::cerr << "Unable to open the file.\n";
    }

}


std::string clust_header(int n_nodes) {

    std::string ret_v = "";
    for (int node=0; node < (n_nodes-1); node++) {
        ret_v += ("node_" + std::to_string(node) + ",");
    }
    ret_v += ("node_" + std::to_string(n_nodes-1));

    return ret_v;
}
