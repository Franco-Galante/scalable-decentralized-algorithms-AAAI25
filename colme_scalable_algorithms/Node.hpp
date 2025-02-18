#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>    // needed for isnan
#include <limits>   // for infty, and nan (c++17 needed)
#include <string>
#include <algorithm>
#include <utility>
#include "Rand.hpp"
#include "Helpers.hpp"


class Node
{
protected:
    int   node_class;  // true distribution class of the node. i s.t. D_i = D_j \in {D_j} (so far 2 Gaussian classes)
    float new_beta;    // updated parametrized confidence interval, depends on i) # sample and ii) 'effective' neighbors
    float curr_beta;   // value used by the other nodes (as the samples depend on the 'non-updated' estimate)
    float estimate;    // estimate computed using the info and the rules according to the algorithm chosen
    float oracle;      // estimate computed with same rule as the algorithm but, knowing the nodes of the same class

    float class_noise; // for the "imperfect" case, it is possible that a node has \mu + \epsilon as its distrib's mean

    // In current implementation these structures don't change in size once constructed (Graph()) and init (by generators)
    std::vector<int>  neighbors;       // adjacency list, list of neighbors of the Node (no self-loops, no multiple nodes)
    std::vector<bool> flag_neigh_same; // (flags) is the neighbor at index idx same class as Node? (less memory occupation)
    std::vector<bool> curr_neigh_opt;  // (flags) is the neighbor deemed of the same class according to the optimistic dist
    std::vector<bool> new_neigh_opt;   // updated version of the previous vector
                                       // it says, "at the previous step did I use the neighbor or not to compute mu_k?"

    /* Parametrization of confidence intervals as for Maillard 2019
     *
     * PARAMETERS:
     *  - std_p     : standard deviation of underlying distribution or bound on it (sub-gaussianity)
     *  - n_p       : number of samples from the distribution used to compute the mean
     *  - delta_p   : precision, i.e., Pr{|true_mu-estimate|>epsilon}<delta_p
     *  - n_collab_p: numb of 'collaborating' nodes, colud be N (total) or N_n (neighbourhood), up to calling func (DEN)
     */
    float param_confidence_interval(float std_p, int n_p, float delta_p, int n_collab_p) {

        if (n_p < 2) {
            return std::numeric_limits<float>::infinity();
        } else {
            float gamma = delta_p / static_cast<float>(n_collab_p); // all the constants should be in the provided var
            float ln_v = std::log(std::sqrt(static_cast<float>(n_p) + 1.0f) / gamma);
            return std_p * std::sqrt(2.0f * (1.0f / static_cast<float>(n_p)) * (1.0f + (1.0f / static_cast<float>(n_p))) * ln_v);
        }
    }

public:
    /* Constructor 1:
     *
     * PARAMETERS:
     *  - n_c    : class (random) of the Node, 1-1 correpondence with the available distributions
     *  - avg_deg: average degree of the collaborative network, estimated size of the adjacency list
     * 
     * NOTE: if we use N for the c.i. beta we can avoid to keep it as it depends only on the # of samples.
     */
    Node(int n_c, int avg_deg) :  node_class(n_c), new_beta(std::numeric_limits<float>::infinity()), curr_beta(new_beta), 
                                    estimate(-1.0f), oracle(-1.0f), class_noise(std::numeric_limits<float>::quiet_NaN()) {
        size_t v = static_cast<size_t>(avg_deg);
        neighbors.reserve(v);       // not precise bu the best guess we can make at this point
        flag_neigh_same.reserve(v);
        curr_neigh_opt.reserve(v);
        new_neigh_opt.reserve(v);
    }

    // Constructor 2: specify the noise in the node mean estimate
    Node(int n_c, int avg_deg, float noise) :  node_class(n_c), new_beta(std::numeric_limits<float>::infinity()), curr_beta(new_beta), 
                                    estimate(-1.0f), oracle(-1.0f), class_noise(noise) {
        neighbors.reserve(avg_deg);       // not precise bu the best guess we can make at this point
        flag_neigh_same.reserve(avg_deg);
        curr_neigh_opt.reserve(avg_deg);
        new_neigh_opt.reserve(avg_deg);
    }

    // NOTE: if the class has virtual functions I have to define also a virtual distructor
    virtual ~Node() {} // I don't have heap object (I should be able to define an empty destructor)


    /* Add a neighbor node to the adjacency list avoiding duplicates. Updates the vector indicating which neighbors are
     * actually from the same (distribution) class. Needed to compute the oracle, irregardless of the algorithm chosen.
     * IMPORTANT NOTE: the method assumes the network is constructed before the dynamics, so n=0 and beta=+\infty, thus
     *                 all the neighbors are initially considered in the same class as Node through the optimistic dist.
     */
    void add_neighbor(int node_id, int neigh_id, int neigh_class, float neigh_noise, std::vector<float> mu_values_p) {
        // a set would guarantee uniqueness of elements and has better search performances O(log(n)) but i) graphs will
        // be sufficiently sparse and ii) I keep a boolean mask for the 'true' neighbors with a 1-1 relation with idxs
        if (Helpers::vector_find(neighbors, neigh_id) == -1) { // neighbor not yet present
            neighbors.push_back(neigh_id);
            if (std::isnan(class_noise)) { 
                flag_neigh_same.push_back(node_class == neigh_class);
            }
            else { 
                float mu_node  = mu_values_p[node_class] + class_noise;
                float mu_neigh = mu_values_p[neigh_class] + neigh_noise;
                bool check = Helpers::are_equal(mu_node, mu_neigh);
                flag_neigh_same.push_back(check);
            }
            curr_neigh_opt.push_back(true); // init all in the optimistic neighbor class
            new_neigh_opt.push_back(true);
        }
        else if(neigh_id == node_id) { // avoid self loops in the network 
            throw std::runtime_error("ERROR[c++node]: attempting to generate a self loop");
        }
        else {
            throw std::runtime_error("ERROR[c++node]: attempting to add the same node twice");
        }
    }


    /* Remove a neighbor from the adjacency list, exits with an error if the node is not present.
     * NOTE: it updates both 'curr' and 'new' versions of the optimistic neighbors, could be needed to updated only the
     *       'new' version
     */
    void remove_neighbor(int neigh_id) {

        // no need to perform a 'find' as the swap_delete method already performs it
        int idx_to_delete = Helpers::swap_delete(neighbors, neigh_id);
        if (idx_to_delete == -1) {
            std::cout << "ERROR[c++node]: attempt to delete a neighbor not in the adjacency list" << std::endl;
            exit(-1);
        }
        else { // the neighbor is in the adjacency list and has been removed, the other records have to be updated
            Helpers::swap_delete_by_index(flag_neigh_same, idx_to_delete);
            Helpers::swap_delete_by_index(curr_neigh_opt, idx_to_delete);
            Helpers::swap_delete_by_index(new_neigh_opt, idx_to_delete);
        }
    }


    void erase_neighbors() {
        neighbors.clear();
        flag_neigh_same.clear();
        curr_neigh_opt.clear();
        new_neigh_opt.clear();
    }


    // (read-only reference) to the adjacency list of the Node
    const std::vector<int>& neighbors_list() { return neighbors; }


    // returns the number of neighbors for the Node
    int num_neighbors() { return static_cast<int>(neighbors.size()); }


    // the vector is generated in the method, I return it by value. ID of only the neighbors of the same class
    std::vector<int> true_same_class_neighbors() { return Helpers::filter_by_mask(neighbors, flag_neigh_same); }

    std::vector<int> true_same_class_index() { return Helpers::index_by_mask(neighbors, flag_neigh_same); }


    // analogous as above, returns the (previous) list of 'optimistic' neighbors
    std::vector<int> opt_same_class_neighbors() { return Helpers::filter_by_mask(neighbors, curr_neigh_opt); }


    // updates the flag of the neighbor according to optimistic distance computed by the calling method
    // used when the calling method loops over the 'neighbors', therefore the index of the neighbor is known
    // RETURN: it returns 'true' if the old value is different from the current one, 'false' otherwise
    bool update_neighbor_flag(int idx_neigh, bool flag_p) {
        bool is_updated = (flag_p != new_neigh_opt[idx_neigh]);
        new_neigh_opt[idx_neigh] = flag_p;
        return is_updated;
    }


    // check which neighbors have been removed from the optimistic set, after the 'new' vector has been updated
    // returns the INDICES of the neighbors removed for the Node object
    // NOTE: those that have been removed and may "re-appear" as optimistic neighbors do not need care, the update
    //       method would just populate the row in the table again
    std::vector<int> idx_removed_neighbors() { return Helpers::changed_indices(curr_neigh_opt, new_neigh_opt); }


    // method used to check if Node "sees" 'node_p', i.e., if Node has used the info from 'node_p' in its metrics
    // this is used to elimiate the info sent by Node (which may include the ridoundant info from 'node_p') when
    // updating the mu_k structure of 'node_p'. (NOTE: use the 'current' version of the flag vector)
    bool isin_opt_class(int node_p) {
        // first search for the element in the Node adjacency list, the graph is undirected so the element must be present
        int idx_node_p = Helpers::vector_find(neighbors, node_p);
        if (idx_node_p != -1) // element found
            return curr_neigh_opt[idx_node_p]; 
        else { // element not found (not possible -> FATAL ERROR)
            std::cout << "FATAL ERROR[c++node]: directed link present in the network" << std::endl;
            exit(-1);
        }
    }


    // returns the nuumber of neighbors considered to compute the 'current' metric; needed to RENORMALIZE the values
    // i.e., to remove the redundant info sent by the Node
    int num_opt_neighbors() { return static_cast<int>(std::count(curr_neigh_opt.begin(), curr_neigh_opt.end(), true)); }


    // returns the (updated) value of 'optimistic neighbors', to be used after nodes have been updated
    int new_num_opt_neighbors() { return static_cast<int>(std::count(new_neigh_opt.begin(), new_neigh_opt.end(), true)); }


    int num_true_neighbors() { return static_cast<int>(std::count(flag_neigh_same.begin(), flag_neigh_same.end(), true)); }

    float noise() { return class_noise; }


    // returns the neighbor at 'index' and nan if the index is out of bound
    int read_neighbor_at(int idx) { return Helpers::safe_read(neighbors, idx); }


    // searches for 'n' in Node's adjacency list, returns -1 if 'n' is not a neighbor of Node and the index in
    // the adjacency list otherwise
    int idx_of_neigh(int n) { return Helpers::vector_find(neighbors, n); }


    // returns true if (in previous step) the neighbors at idx 
    bool is_neighbor_at_opt(int idx) { return curr_neigh_opt[idx]; }


    // check if the Node is isolated, i.e., no
    bool is_isolated() { return neighbors.size() == 0; }


    // updates 'current' version of the neighbors flag, to be used at the end of cycle, similar to what is done for beta
    void refresh_neigh_flag() { curr_neigh_opt.assign(new_neigh_opt.begin(), new_neigh_opt.end()); }


    // counts the number of "wrong" (directed) connections between the nodes: Returns a pair, the first value is the 
    // number of "lost neighbors" those that are of the same class as Node but are considered not such, in the 
    // 'irreversible' case this value is weakly increasing, in the 'non-irreversible' the value will evetually reach 0.
    // The second value in the pair is the number of the "wrong neighbors", those that are considered as neighbors but
    // actually are not of the same class (and so in the transient make the estimate bad). This value will eventually
    // go to zero, when all the nodes have EXCLUDED their "wrong" neighbors
    std::pair<int, int> wrong_links() {
        // for the comparison uses 'flag_neigh_same' and 'new_neigh_opt' (which is kept updated also in colme-like algs)
        int lost_neigh  = 0;
        int wrong_neigh = 0;

        for (int i=0; i< static_cast<int>(new_neigh_opt.size()); i++) {
            if ((flag_neigh_same[i]) && (!new_neigh_opt[i]))
                lost_neigh++;
            else if ((!flag_neigh_same[i]) && (new_neigh_opt[i])) {
                wrong_neigh++;
            }
        }
        return std::make_pair(lost_neigh, wrong_neigh);
    }


    /***************************************************************************************************************/

    // update the value of beta at the given iteration (new_beta), according to the iteration, returns the updated value
    float update_beta(float std_p, int iter_p, float delta_p, int n_collab_p) {
        int n_p = iter_p + 1; // number of samples
        new_beta = param_confidence_interval(std_p, n_p, delta_p, n_collab_p);
        return new_beta;
    }


    // refreshes the 'curr_beta' with the new value whenever all the nodes have updated their values
    void refresh_beta() { curr_beta = new_beta; }


    // returns the current (t) non updated value of the parametrized confidence interval beta
    float read_beta() { return curr_beta; }


    // gives access to the class (identifying the distribution) for the given node
    int node_true_class() { return node_class; }


    // virtual functions, whose specific implementation is provided in each specialized class (NodeX)
    virtual float update_mean(int n_p, std::vector<float>& mu_values_p, float std_p, int seed_p) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    virtual float local_mean() { return std::numeric_limits<float>::quiet_NaN(); }


    /******************************************************************************************************************/

    // initializes the internal structure specific to the subclass NodeX
    virtual void init_internal_struct(std::vector<float>& mu_values_p, float std_p, int seed_p, int add1_p = -1, int add2_p = -1) {
        std::cout << "ERROR[c++node]: no general implementation of init_internal_struct is possible" << std::endl;
        exit(-1);
    }

    virtual void update_internal_struct(float f, int i = -1, int j = -1, int k = -1) {
        std::cout << "ERROR[c++node]: no general implementation of manage_tables is possible" << std::endl;
        exit(-1);
    }

    virtual void update_oracle_struct(float f, int i = -1, int j = -1, int k = -1) {
        std::cout << "ERROR[c++node]: no general implementation of manage_tables is possible" << std::endl;
        exit(-1);
    }

    virtual float read_struct_at(int idx=-1, int neigh_idx = -1, bool flag=false) { return std::numeric_limits<float>::quiet_NaN(); }

    virtual float read_oracle_at(int idx=-1, int neigh_idx = -1, bool flag=false) { return std::numeric_limits<float>::quiet_NaN(); }

    virtual void print_internal_struct() {
        std::cout << "ERROR[c++node]: no general implementation of refresh_internal is possible" << std::endl;
        exit(-1);
    }

    virtual float read_old_struct_at(int idx) { return std::numeric_limits<float>::quiet_NaN(); }

    virtual float read_old_oracle_at(int idx) { return std::numeric_limits<float>::quiet_NaN(); }

    virtual void refresh_internal_struct() {
        std::cout << "ERROR[c++node]: no general implementation of refresh_internal is possible" << std::endl;
        exit(-1);
    }

    virtual void eliminate_obsolete(int k_p, int other_idx = -1) {
        std::cout << "ERROR[c++node]: no general implementation of eliminate_obsolete is possible" << std::endl;
        exit(-1);
    }

    virtual int read_additional(bool oracle_flag) { return -1; }

    virtual void write_additional(int v, bool oracle_flag) { }

    virtual float estimate_mean(int iter, float std_p=-1.0f, float delta_p=-1.0f) { return std::numeric_limits<float>::quiet_NaN(); }

    float read_estimate() { return estimate; }

    virtual float oracle_estimate(int iter, float std_p=-1.0f, float delta_p=-1.0f) { return std::numeric_limits<float>::quiet_NaN(); }

    float read_oracle() { return oracle; }


    /********************************* to be able to use colme ********************************************************/
    
    // I need both the actual neighbors (now also to compute the wrong and lost neighbors, before I used it for the oracle
    // ) and the optimal neighbors for metrics on the learning and neighbors exclusion. No need for current and new it is
    // enough to hvae one of the two
    void flag_reserve(int r) {
        flag_neigh_same.reserve(r);
        new_neigh_opt.reserve(r);
    }

    void set_flag_at(bool v, int idx) { Helpers::safe_add_or_update(flag_neigh_same, idx, v); }

    void set_opt_flag_at(bool v, int idx) {  Helpers::safe_add_or_update(new_neigh_opt, idx, v); }

    bool read_flag_at(int idx) { return flag_neigh_same[idx]; }

    bool read_opt_flag_at(int idx) { return new_neigh_opt[idx]; }

};



/*********************************************************** INRIA Node **********************************************/

class NodeA : public Node // specific to the 'Inria' approach to which we compare ourselves
{
private:
    float curr_mean; // this does not need an oracle (no neighoìbors queried)
    float new_mean;
    int last_updated;
    int oracle_last_updated;
    std::vector<std::pair<float, int>> others_mean; // this is for exclusive use of Node (no other nodes looks at it)
    std::vector<std::pair<float, int>> oracle_others_mean; 

public:
    NodeA(int n_c, int n_nodes) : Node(n_c, 0), curr_mean(0.0f), new_mean(0.0f), last_updated(0), oracle_last_updated(0) { 
        others_mean.reserve(n_nodes);
        oracle_others_mean.reserve(n_nodes);
    }


    NodeA(int n_c, int n_nodes, float noise_p) : Node(n_c, 0, noise_p), curr_mean(0.0f), new_mean(0.0f), last_updated(0), oracle_last_updated(0) { 
        others_mean.reserve(n_nodes);
        oracle_others_mean.reserve(n_nodes);
    }

    void init_internal_struct(std::vector<float>& mu_values_p, float std_p, int seed_p, int n_nodes, int node_p) override {

        if (std::isnan(class_noise))
            new_mean = Rand::gaussian_rv(mu_values_p[node_class], std_p, seed_p);
        else
            new_mean = Rand::gaussian_rv(mu_values_p[node_class]+class_noise, std_p, seed_p);
        curr_mean = new_mean; // as if it was already 'refreshed'

        // init the internal structure with nan in the float filed, this will also signal a non-optimistic neighbor
        // at the beginning beta=\infty therefore all neighbors will be 'optimistic' neighbors, all nodes are possible neighbors
        for (int i = 0; i < n_nodes; i++)
            others_mean.emplace_back(std::numeric_limits<float>::quiet_NaN(), 0);
        others_mean[node_p] = std::make_pair(std::numeric_limits<float>::quiet_NaN(), -1); // idx of node itself (iter>0 by def)
        
        // init also the 'oracle structure'
        for (int i = 0; i < n_nodes; i++) {
            if (flag_neigh_same[i])
                oracle_others_mean.emplace_back(std::numeric_limits<float>::quiet_NaN(), 0);
            else // can already place inf, this node will never be actually considered by the 'oracle'
                oracle_others_mean.emplace_back(std::numeric_limits<float>::infinity(), 0);
        }
        oracle_others_mean[node_p] = std::make_pair(std::numeric_limits<float>::quiet_NaN(), -1); // idx associated to node itself
    }


    float update_mean(int n_p, std::vector<float>& mu_values_p, float std_p, int seed_p) override {
        int n_samples = n_p + 1; // at n=0 all nodes have 1 sample

        // first element (mean) always has to be updated with the new sample, irregardless of the other neighbors
        float noise_to_add = 0.0f; // if no noise I do not add anything
        if (! std::isnan(class_noise))
            noise_to_add = class_noise;

        float new_sample = Rand::gaussian_rv(mu_values_p[node_class]+noise_to_add, std_p, seed_p);
        new_mean = curr_mean * ((n_samples-1.0f) / n_samples) + new_sample * (1.0f / n_samples);

        return new_mean;
    }


    float local_mean() override { return curr_mean; }


    // if inf the corresponding neighbor is not considered an 'optimistic' neighbor anymore
    float read_struct_at(int neigh_idx, int element, bool dummy=false) override { return read_pair_vec(others_mean, neigh_idx, element); }


    float read_oracle_at(int neigh_idx, int element, bool dummy=false) override { return read_pair_vec(oracle_others_mean, neigh_idx, element); }


    // the third value here is the sample at which the record has been collected
    void update_internal_struct(float f, int neigh_idx, int sample, int dummy=-1) override {
        others_mean[neigh_idx] = std::make_pair(f, sample);
    }

    void update_oracle_struct(float f, int neigh_idx, int sample, int dummy=-1) override {
        oracle_others_mean[neigh_idx] = std::make_pair(f, sample);
    }


    void refresh_internal_struct() override { curr_mean = new_mean; }


    /* Prints the table with all the k-hops information for the given node (row by row)
     */
    void print_internal_struct() override {
        std::cout << "All records: ";
        for (int i=0; i<static_cast<int>(others_mean.size()); i++) {
            std::cout << "(" << others_mean[i].first << ", " << others_mean[i].second << "), ";
        }
    }


    int read_additional(bool oracle_flag) override {
        if (oracle_flag) 
            return last_updated;
        else
            return oracle_last_updated;
    }

    void write_additional(int v, bool oracle_flag) override {
        if (oracle_flag)
            last_updated = v;
        else
            oracle_last_updated = v;
    }

    // if the additional parametrs (std_p and delta_p) are left blank (-> <0) then I know that the standard version of the 
    // colme algorithm needs to be used, otherwise I have to use the version that recomputes the optimistic neighborhood
    float estimate_mean(int iter, float std_p=-1.0f, float delta_p=-1.0f) override { // there is always at least the 'local mean'
        if ((std_p<0) || (delta_p<0)) {
            estimate = weighted_vec_mean(others_mean, iter);
            return estimate;
        }
        else { // recompute the optimistic set at each time
            estimate = weighted_vec_recompute(others_mean, iter, std_p, delta_p, false);
            return estimate;
        }
    }

    float oracle_estimate(int iter, float std_p=-1.0f, float delta_p=-1.0f) override {
        if ((std_p<0) || (delta_p<0)) {
            oracle = weighted_vec_mean(oracle_others_mean, iter);
            return oracle;
        }
        else { // recompute the optimistic set at each time
            oracle = weighted_vec_recompute(oracle_others_mean, iter, std_p, delta_p, true);
            return oracle;
        }
    }

private:
    float read_pair_vec(std::vector<std::pair<float,int>>& vec, int idx, int e) {
        if (e == 0) // read first element in pair (the float value)
            return vec[idx].first;
        else
            return static_cast<float>(vec[idx].second); // I need to do a cast in the calling method
    }

    float weighted_vec_mean(std::vector<std::pair<float,int>>& vec, int iter) {

        int norm_fact = iter+1; // the mean (all possible sample, with the update)
        float tmp_est = new_mean * norm_fact; // new mean same for 'opt' and 'oracle'

        for (int i=0; i<static_cast<int>(vec.size()); i++) {
            if ((!std::isinf(vec[i].first)) && (!std::isnan(vec[i].first)) && vec[i].second != -1) {
                tmp_est += (vec[i].first * vec[i].second);
                norm_fact += vec[i].second;
            }
        }
        return (1.0f / norm_fact) * tmp_est;
    }

    // method similar to the one above, with the difference that at each iteration it recomputes the set of 'optimistic' nodes
    // NOTE: here all nodes have (potential) access to all other nodes, so the beta parametrization is always done with respect
    //       to the total number of nodes 'n_nodes'. Then I need some additional info, i.e., the standard deviation and delta
    float weighted_vec_recompute(std::vector<std::pair<float,int>>& vec, int iter, float std_p, float delta_p, bool is_oracle) {

        int n_nodes_v = static_cast<int>(vec.size()); // the structure has a record for each node (also the node itself)

        int norm_fact = iter+1;
        float tmp_est = new_mean * norm_fact;

        for (int i=0; i<static_cast<int>(vec.size()); i++) {

            if (vec[i].second != -1) { // exclude the 'node' location itself
                
                int iter_neigh   = vec[i].second;
                float neigh_mean = vec[i].first;
                float saved_neigh_beta = param_confidence_interval(std_p, iter_neigh+1, delta_p, n_nodes_v); // compute beta parametrized
                
                float opt_dist;
                if (std::isnan(neigh_mean)) 
                    opt_dist = -1.0f * std::numeric_limits<float>::infinity(); // neigh not even seen yet
                else if (std::isinf(neigh_mean))
                    opt_dist = std::numeric_limits<float>::infinity(); // the distance should be POSITIVE -> we don't consider it
                else
                    // recompute the optimistic distance with the new values of mu and beta for the given node
                    opt_dist = std::abs(new_mean - neigh_mean) - new_beta - saved_neigh_beta;

                if (! is_oracle)
                    new_neigh_opt[i] = (opt_dist <= 0.0f); // update the optimistic state of the neighbor

                if (opt_dist <= 0.0f) {

                    if ((!std::isinf(vec[i].first)) && (!std::isnan(vec[i].first))) {
                        tmp_est += (vec[i].first * vec[i].second);
                        norm_fact += vec[i].second;
                    }
                }
            }

        }
        return (1.0f / norm_fact) * tmp_est;
    }

};


/************************************************* Beflief agorithm impl. 1 *******************************************/

class NodeB : public Node // extends the class node (specific for belief propagation alg)
{
private:
    // table contains all k-hop (whenever available) info from all Node's optimistic neighbors
    // each row is associated with an optimistic neighbor with all the metrics it passes to Node (variable lenght)
    std::vector<std::vector<float>> curr_table;   // table at t
    std::vector<std::vector<float>> new_table;    // table at t+1
    std::vector<std::vector<int>> curr_comp_over; // number of nodes over which tab[neigh, k] record has been computed over
    std::vector<std::vector<int>> new_comp_over;
    float curr_mean;                            // local mean values, for convenience, outside of the 'table' structure
    float new_mean;                             // updated mean value

    std::vector<std::vector<float>> oracle_curr_table;
    std::vector<std::vector<float>> oracle_new_table;
    std::vector<std::vector<int>> oracle_curr_comp_over; 
    std::vector<std::vector<int>> oracle_new_comp_over;

public:
    /* Constructor 1:
     *
     * PARAMETERS:
     *  - m_c        : class (random) of the Node, 1-1 correpondence with the available distributions
     *  - avg_deg    : average degree of the collaborative network, estimated size of the adjacency list
     */
    NodeB(int m_c, int avg_deg) : Node(m_c, 0), curr_mean(0.0f), new_mean(0.0f) { 
        reserve_struct(avg_deg);
        reserve_oracle(avg_deg);
    }


    NodeB(int m_c, int avg_deg, float noise_p) : Node(m_c, 0, noise_p), curr_mean(0.0f), new_mean(0.0f) {
        reserve_struct(avg_deg);
        reserve_oracle(avg_deg);
    }


    /* Initialize the table with the values once the collaborative network G is known (so that we know the # neighbors)
     * IMPORTANT NOTE: no changes in the network are admitted after the internal structure has been initialized
     */
    void init_internal_struct(std::vector<float>& mu_values_p, float std_p, int seed_p, int diam, int dummy=-1) {

        if (std::isnan(class_noise))
            new_mean = Rand::gaussian_rv(mu_values_p[node_class], std_p, seed_p);
        else
            new_mean = Rand::gaussian_rv(mu_values_p[node_class]+class_noise, std_p, seed_p);

        curr_mean = new_mean; // as if it was already 'refreshed'

        int n_neigh = static_cast<int>(neighbors.size());

        // at the beginning beta=\infty therefore all neighbors will be 'optimistic' neighbors: create empty vectors
        if (neighbors.size() > 0) { // not an isolated node

            // I init these structures with an empty vector (table line) for each neighbor
            curr_table.resize(n_neigh);   // make n_neigh rows
            new_table.resize(n_neigh);
            curr_comp_over.resize(n_neigh);
            new_comp_over.resize(n_neigh);

            // oracle init
            oracle_curr_table.resize(n_neigh);
            oracle_new_table.resize(n_neigh);
            oracle_curr_comp_over.resize(n_neigh);
            oracle_new_comp_over.resize(n_neigh);

            for (size_t i=0; i < neighbors.size(); i++) {

                curr_table[i].reserve(diam);
                curr_table[i] = {}; // init empty vector for each neighbor

                new_table[i].reserve(diam);
                new_table[i]  = {};

                curr_comp_over[i].reserve(diam);
                curr_comp_over[i] = {}; // init empty vector for each neighbor

                new_comp_over[i].reserve(diam);
                new_comp_over[i]  = {};


                // oracle part
                oracle_curr_table[i].reserve(diam);
                oracle_curr_table[i] = {};

                oracle_new_table[i].reserve(diam);
                oracle_new_table[i]  = {};

                oracle_curr_comp_over[i].reserve(diam);
                oracle_curr_comp_over[i] = {}; // init empty vector for each neighbor

                oracle_new_comp_over[i].reserve(diam);
                oracle_new_comp_over[i]  = {};
            }
        }
    }

    /* Computes the updated version of the mean considering the new sample that arrives at the node
     */
    float update_mean(int n_p, std::vector<float>& mu_values_p, float std_p, int seed_p) override {
        int n_samples = n_p + 1; // at n=0 all nodes have 1 sample

        // first element (mean) always has to be updated with the new sample, irregardless of the other neighbors
        float noise_to_add = 0.0f; // if no noise I do not add anything
        if (! std::isnan(class_noise))
            noise_to_add = class_noise;
        
        float new_sample = Rand::gaussian_rv(mu_values_p[node_class]+noise_to_add, std_p, seed_p);
        new_mean = curr_mean * ((n_samples-1.0f) / n_samples) + new_sample * (1.0f / n_samples);

        return new_mean;
    }


    // returns the local mean of the Node (first element in the mu_k structure), used to check the mean of a neighbor
    float local_mean() override { return curr_mean; }


    // depth the k, related to the k-info, from which we need to erase
    void eliminate_obsolete(int table_idx, int depth = -1) override {
        if (depth == -1) { // remove the row in the table
            Helpers::replace_with_empty(new_table, table_idx);
            Helpers::replace_with_empty(new_comp_over, table_idx);
        }
        else { // remove a particular record in the table, at row 'table_idx' and at 'depth'
            if (static_cast<int>(curr_table[table_idx].size()) > depth) {
                auto start_itr = new_table[table_idx].begin() + depth;
                new_table[table_idx].erase(start_itr, new_table[table_idx].end());
            }
            // empty also the cout structure
            if (static_cast<int>(curr_comp_over[table_idx].size()) > depth) {
                auto start_itr = new_comp_over[table_idx].begin() + depth;
                new_comp_over[table_idx].erase(start_itr, new_comp_over[table_idx].end());
            }
        }
    }

    float read_struct_at(int neigh_idx, int depth, bool is_comp) override {
        if(is_comp)
            return static_cast<float>(Helpers::safe_read(curr_comp_over[neigh_idx], depth)); 
        else
            return Helpers::safe_read(curr_table[neigh_idx], depth);
    } // nan if no element

    float read_oracle_at(int neigh_idx, int depth, bool is_comp) override {
        if(is_comp)
            return static_cast<float>(Helpers::safe_read(oracle_curr_comp_over[neigh_idx], depth)); 
        else
            return Helpers::safe_read(oracle_curr_table[neigh_idx], depth); 
    }


    // neigh_idx: row corresponding to that neighbor in the table (can be nan)
    // depth    : depth of the info, (depth+1)-hop info
    void update_internal_struct(float f, int neigh_idx, int depth, int n_k) override {
        // NOTE: it is guaranteed that the 'neigh_idx' record exists (init make one record gor each neighbor)
        Helpers::safe_add_or_update(new_table[neigh_idx], depth, f);
        Helpers::safe_add_or_update(new_comp_over[neigh_idx], depth, n_k);
    }

    void update_oracle_struct(float f, int neigh_idx, int depth, int n_k) override {
        Helpers::safe_add_or_update(oracle_new_table[neigh_idx], depth, f);
        Helpers::safe_add_or_update(oracle_new_comp_over[neigh_idx], depth, n_k);
    }


    void refresh_internal_struct() override {

        curr_mean = new_mean;

        curr_table.clear();  // Clear the existing content of curr_table
        for (const auto& inner_vec : new_table) {
            curr_table.push_back(inner_vec);  // Perform a deep copy of each inner vector
        }
        curr_comp_over.clear();  // Clear the existing content of curr_table
        for (const auto& inner_vec : new_comp_over) {
            curr_comp_over.push_back(inner_vec);  // Perform a deep copy of each inner vector
        }

        // refresh also the oracle structure
        oracle_curr_table.clear();  // Clear the existing content of curr_table
        for (const auto& inner_vec : oracle_new_table) {
            oracle_curr_table.push_back(inner_vec);  // Perform a deep copy of each inner vector
        }
        oracle_curr_comp_over.clear();  // Clear the existing content of curr_table
        for (const auto& inner_vec : oracle_new_comp_over) {
            oracle_curr_comp_over.push_back(inner_vec);  // Perform a deep copy of each inner vector
        }
    }


    /* Prints the table with all the k-hops information for the given node (row by row)
     */
    void print_internal_struct() override {
        int max_digits = Helpers::count_digits(static_cast<int>(new_table.size()));
        std::cout << std::endl; // formatting
        for (size_t i=0; i<new_table.size(); i++) {
            if (! new_table[i].empty()) { // the 'non-optimistic' nodes may have an empty record
                std::cout << "\t\tNeigh " + Helpers::str_left_zeros(neighbors[i], max_digits) + ": ";
                Helpers::operator<<(std::cout, new_table[i]); // I can plot the row
                std::cout << std::endl; // formatting
            }
            else { // empty record
                std::cout << "\t\tNeigh " + Helpers::str_left_zeros(neighbors[i], max_digits) + ": empty" << std::endl;
            }

            // do the same but also for the 'computed-over' data structure
            if (! new_comp_over[i].empty()) { // the 'non-optimistic' nodes may have an empty record
                std::cout << "\t\tNeigh " + Helpers::str_left_zeros(neighbors[i], max_digits) + ": ";
                Helpers::operator<<(std::cout, new_comp_over[i]); // I can plot the row
                std::cout << std::endl; // formatting
            }
            else { // empty record
                std::cout << "\t\tNeigh " + Helpers::str_left_zeros(neighbors[i], max_digits) + ": empty" << std::endl;
            }
        }
        std::cout << std::endl; // formatting
    }

    float estimate_mean(int iter, float dummy1=-1.0f, float dummy2=-1.0f) override {
        estimate = compute_table_estimate(iter, new_table, new_comp_over);
        return estimate;
    }

    float oracle_estimate(int iter, float dummy1=-1.0f, float dummy2=-1.0f) override {
        oracle = compute_table_estimate(iter, oracle_new_table, oracle_new_comp_over);
        return oracle;
    }

private:
    float compute_table_estimate(int iter, std::vector<std::vector<float>>& tab, std::vector<std::vector<int>>& count) {

        int norm_coeff = iter+1; // num of samples are iter +1 , and computed over just one node
        float tmp_v = (static_cast<float>(iter)+1.0f) * new_mean;

        for (int ni=0; ni < static_cast<int>(tab.size()); ni++) { // tab and count structure have exactly the same structure

            for (int k=0; k < static_cast<int>(tab[ni].size()); k++) {
                
                tmp_v += (tab[ni][k] * static_cast<float>(iter-k)); // k here is (k'+1)-hop info -> iter+1-k' = iter+1-1-k = iter+1-(k+1)=iter-k
                norm_coeff += (count[ni][k] * (iter-k));
            }
        }

        return tmp_v / norm_coeff; // should cast as first el is float
    }

    void reserve_struct(int r) {
        curr_table.reserve(r); // for i=0 -> \beta=inf thus there are as many elements as the neighbors
        new_table.reserve(r);
        curr_comp_over.reserve(r);
        new_comp_over.reserve(r);
    }

    void reserve_oracle(int r) {
        oracle_curr_table.reserve(r);
        oracle_new_table.reserve(r);
        oracle_curr_comp_over.reserve(r);
        oracle_new_comp_over.reserve(r);
    }

};



/**************************************************** Consensus agorithm **********************************************/

class NodeC : public Node // extends the class node (specific for consensus alg)
{
private:
    float alpha;

    float curr_mean;
    float new_mean;

    float curr_cons;
    float new_cons;
    float oracle_curr_cons;
    float oracle_new_cons;

public:
    NodeC(int n_c, int avg_deg, float alpha_p) : Node(n_c, avg_deg), alpha(alpha_p), curr_mean(0.0f), new_mean(0.0f), 
                        curr_cons(0.0f), new_cons(0.0f), oracle_curr_cons(0.0f), oracle_new_cons(0.0f) {}

    NodeC(int n_c, int avg_deg, float noise_p, float alpha_p) : Node(n_c, avg_deg, noise_p), alpha(alpha_p), curr_mean(0.0f), new_mean(0.0f), 
                        curr_cons(0.0f), new_cons(0.0f), oracle_curr_cons(0.0f), oracle_new_cons(0.0f) {}


    void init_internal_struct(std::vector<float>& mu_values_p, float std_p, int seed_p, int add1 = -1, int add2 = -1) override {

        float first_sample = 0.0f;
        if (std::isnan(class_noise))
            first_sample = Rand::gaussian_rv(mu_values_p[node_class], std_p, seed_p);
        else
            first_sample = Rand::gaussian_rv(mu_values_p[node_class]+class_noise, std_p, seed_p);

        curr_mean = new_mean = first_sample; 

        curr_cons = new_cons = oracle_curr_cons = oracle_new_cons = first_sample;
    }

    float update_mean(int n_p, std::vector<float>& mu_values_p, float std_p, int seed_p) override {
        int n_samples = n_p + 1; // at n=0 all nodes have 1 sample

        // first element (mean) always has to be updated with the new sample, irregardless of the other neighbors
        float noise_to_add = 0.0f; // if no noise I do not add anything
        if (! std::isnan(class_noise))
            noise_to_add = class_noise;

        float new_sample = Rand::gaussian_rv(mu_values_p[node_class]+noise_to_add, std_p, seed_p);
        new_mean = curr_mean * ((n_samples-1.0f) / n_samples) + new_sample * (1.0f / n_samples); // constructed in init
        return new_mean;
    }

    float local_mean() override { return curr_mean; }

    void print_internal_struct() override {
        std::cout << "x: " << std::fixed << std::setprecision(4) << curr_mean;
        std::cout << "\ty: " << std::fixed << std::setprecision(4) << curr_cons;
    }

    float read_struct_at(int dummy1=-1, int dummy2=-1, bool dummy=false) override {return curr_cons; }

    void update_internal_struct(float f, int dummy1=-1, int dummy2=-1, int dummy3=-1) override { new_cons = f; }

    float read_oracle_at(int dummy1=-1, int dummy2=-1, bool dummy=false) override {return oracle_curr_cons; }

    void update_oracle_struct(float f, int dummy1=-1, int dummy2=-1, int dummy3=-1) override { oracle_new_cons = f; }


    // refresh mean and all consistency vals
    void refresh_internal_struct() override {
        curr_mean = new_mean;

        curr_cons = new_cons;
        oracle_curr_cons = oracle_new_cons;
    }

    float estimate_mean(int dummy, float dummy1=-1.0f, float dummy2=-1.0f) override { 
        estimate = new_cons;
        return new_cons;
    }

    float oracle_estimate(int dummy, float dummy1=-1.0f, float dummy2=-1.0f) override {
        oracle = oracle_new_cons;
        return oracle_new_cons;
    }
    
};

