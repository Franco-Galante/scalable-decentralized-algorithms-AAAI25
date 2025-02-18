#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iterator> 
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <queue>
#include <stack>
#include <numeric> // iota
#include "Node.hpp"
#include "Rand.hpp"
#include "Helpers.hpp"
#include "Eigen.hpp"


class Graph {
public:

	std::string alg;
	std::string noise_distrib;
	float epsilon;
	float delta;
	const int n_nodes;
	const int int_avg_deg;
	const float p;
	const int seed;    
	std::vector<float> mu_values;   
	float sigma;                   
	
	std::vector<std::unique_ptr<Node>> g;
	std::vector<int> cluster_id;   
	std::vector<int> cluster_size;
	int diameter;  // diameter of the network
	float alpha;
	int table_up_to;     

	int id_clust_0; // id of largest connected comp (group)
	int id_clust_1; // id of second largest connected comp 


	bool graph_learnt; // becomes true whne the number of "wrong_links" goes to zero
	bool graph_change; // becomes true when a node cuts a link, reset to false at the next iteration
	                   // (has a delay 1 by construction, by updating at the end of the alg loop)
	/* Contructor1:
	 * It creates an empty graph with N nodes and randmoly assignes a class in [0,mu_values.size()] uniformly
	 * The graph is represented though adjecency lists for each one of the node.
	 * 
	 */
	Graph(std::string alg_p, std::string noise_distrib_p, float epsilon_p, float delta_p, int n_nodes_p, int int_avg_deg_p,
			float p_p, int seed_p, int diam_guess_p, const std::vector<float>& mu_values_p, float sigma_p, float alpha_p, int up_to = -1) 
				: alg(alg_p), noise_distrib(noise_distrib_p), epsilon(epsilon_p), delta(delta_p), n_nodes(n_nodes_p),
					int_avg_deg(int_avg_deg_p), p(p_p), seed(seed_p), mu_values(mu_values_p), sigma(sigma_p), cluster_id(n_nodes_p, -1), diameter(1), 
			  		alpha(alpha_p), table_up_to(up_to != -1), id_clust_0(-1), id_clust_1(-1), graph_learnt(false), graph_change(false) {
		
		g.reserve(n_nodes);
		
		int n_classes = static_cast<int>(mu_values.size()) -1;

		if (noise_distrib == "none") {

			/* Creation of an EMPTY graph */
			for (int i = 0; i < n_nodes; i++) {
				/*
				* A vector is the basic data structure representing the graph. Each index represents
				* the node's label, values are pointers to a Node object (where adj list is stored)
				*/
				int rnd_class = Rand::int_uniform_rv(0, n_classes, seed);
				int avg_degree = static_cast<int>(static_cast<float>(n_nodes)*p);

				// initialize the node (the adjecency list will be specified later)
				if (alg == "belief_propagation_v1") {
					table_up_to = up_to;
					g.emplace_back(std::make_unique<NodeB>(rnd_class, avg_degree));
				}
				else if (alg == "consensus") {
					g.emplace_back(std::make_unique<NodeC>(rnd_class, avg_degree, alpha));
				}
				else if ((alg == "colme") || (alg == "colme_recompute")) {
					g.emplace_back(std::make_unique<NodeA>(rnd_class, avg_degree));
				}
				else {
					std::cout << "FATAL ERROR[c++:graph]: unsupported algorithm specified" << std::endl;
				}
			}
		}
		else if (noise_distrib == "unif") {

			/* Creation of an EMPTY graph */

			for (int i = 0; i < n_nodes; i++) {

				int rnd_class   = Rand::int_uniform_rv(0, n_classes, seed);
				float noise_add = Rand::uniform_rv(-epsilon, +epsilon, seed);
				int avg_degree = static_cast<int>(static_cast<float>(n_nodes)*p);

				if (alg == "belief_propagation_v1") {
					table_up_to = up_to;
					g.emplace_back(std::make_unique<NodeB>(rnd_class, avg_degree, noise_add));
				}
				else if (alg == "consensus") {
					g.emplace_back(std::make_unique<NodeC>(rnd_class, avg_degree, noise_add, alpha));
				}
				else if ((alg == "colme") || (alg == "colme_recompute")) {
					g.emplace_back(std::make_unique<NodeA>(rnd_class, avg_degree, noise_add));
				}
				else {
					std::cout << "FATAL ERROR[c++:graph]: unsupported algorithm specified" << std::endl;
				}
			}
		}
		else {
			std::cout << "FATAL ERROR: unsupported noise distribution" << std::endl;
			exit(-1);
		}
	}



	/* (undirected) Erdos Renyi graph generator with the geometric method
	 * This method generates an Erdos Renyi graph taking as parameters the number of nodes n_nodes and average degree
	 * avgD Remind: p = avgD/(N-1), p is the probab to connect two randomly chosen nodes in the graph.
	 *
	 * COMPLEXITY: O(n+m), where m is the number of links in the graph. This complexity is optimal for this problem.
	 */
	void gnp_generator() { // the parameters have already been set in the constructor for the Graph

		if ((alg != "colme") && (alg != "colme_recompute")) {

			int v   =  1;
			int u   = -1;
			float r = 0.0f;

			while (v < n_nodes) {

				r = Rand::uniform_rv(0.0, 1.0, seed); // draw a random value r uniformly in [0,1)

				// log (ln) or log10 should be the same, the logarithm appear as a fraction
				u = u + 1 + static_cast<int>(std::floor(((std::log10(1.0 - r)) / (std::log10(1.0 - p)))));

				while (u >= v && v < n_nodes) {
					u = u - v;
					v++;
				}

				if (v < n_nodes && v != u) { // v != u to avoid SELF LOOPS

					// add the edge (v,u) -> add u to the adj list of v, and add v to the adj of u
					g[v]->add_neighbor(v, u, g[u]->node_true_class(), g[u]->noise(), mu_values);
					g[u]->add_neighbor(u, v, g[v]->node_true_class(), g[v]->noise(), mu_values);
				}
			}

			set_diameter();
			find_clusters(true);
		}
		else {
			mesh_generator(); // all nodes communicate with all other nodes (only 'flag_neighbor_same' is used)
		}
	}


	/* (undirected) Erdos Renyi graph generator with Edge-skipping technique
	 * This method generates an Erdos Renyi with n_nodes and link probability with the edge-skipping technique. This is
	 * an adaptation of the later introduced (more general) EdgeSkipping algorithm. NOTE: Node ID start from 0.
	 *
	 * COMPLEXITY: O(n+m), where m is the number of links in the graph.
	 * IMPLEMENTATION NOTE: in this version we have to represent a number (integer) that identifies an edge, when the
	 * number of nodes is large (order of 100'000), this procedure would overflow the 'int' numbers -> use long long
	 */

	void gnp_generator_2() {

		if ((alg != "colme") && (alg != "colme_recompute")) {

			long long start = 1;
			long long end = (n_nodes * (n_nodes - 1)) / 2;
			long long x = start - 1;

			float r = 0.0f;
			long long l = 0;

			long long u = 0;
			long long v = 0;

			while (x < end) { // strictly less, if the potential edge is the last one we do not enter in the loop
			
				r = Rand::uniform_rv(0.0f, 1.0f, seed); // draw a random value r uniformly in [0,1)

				l = static_cast<long long>(std::floor(std::log10(r) / (std::log10(1 - p)))); // potential edge to be skipped
				x = x + l + 1;

				if (x <= end) { // checking not to have exhausted all the potential edges
				
					// x is the identifier of the edge (edges ordered in lexicografic order and labelled with the integer positive numbers)
					// we need to retrieve the 2-tuple representing the node
					u = static_cast<long long>(std::ceil(((-1.0f + std::sqrt(1 + 8 * x))) / 2.0f));
					v = x - ((u * (u - 1)) / 2) - 1;

					// add the edge (u,v)
					g[v]->add_neighbor(static_cast<int>(v), static_cast<int>(u), g[u]->node_true_class(), g[u]->noise(), mu_values);
					g[u]->add_neighbor(static_cast<int>(u), static_cast<int>(v), g[v]->node_true_class(), g[v]->noise(), mu_values);
				}
			}

			set_diameter();
			find_clusters(true);
		}
		else {
			mesh_generator(); // all nodes communicate with all other nodes (only 'flag_neighbor_same' is used)
		}
	}


	void gnr_generator(int r) { // regular random graph, each node with degree = r

		if ((alg != "colme") && (alg != "colme_recompute")) {

			int trial_count = 0;
			int max_count   = 100;

			bool generated = false;
			while (!generated) {

				if (trial_count < max_count) {
					
					std::vector<int> stabs_id(n_nodes * r);           // int number for each N*d stabs of the nodes
					std::iota(stabs_id.begin(), stabs_id.end(), 0);
					
					if (static_cast<int>(stabs_id.size()) % 2 != 0) { // we want an even number of stabs (to match)
						std::cout << "ERROR[c++graph]: specify a even N*d value for gnr generation" << std::endl;
						exit(-1);
					}

					while(stabs_id.size() > 0) { // until all the stabs have been matched (or a self-loop or a multi path is )

						int idx_u  = Rand::int_uniform_rv(0, static_cast<int>(stabs_id.size()-1), seed);
						int stab_u = stabs_id[idx_u]; // id of the stub, (v) / d tells the node the stabs originated 
													  // I need to save the value because I immediately delete this value
						Helpers::swap_delete_by_index(stabs_id, idx_u);

						int idx_v = Rand::int_uniform_rv(0, static_cast<int>(stabs_id.size()-1), seed);
						int stab_v = stabs_id[idx_v];
						Helpers::swap_delete_by_index(stabs_id, idx_v);

						int u = stab_u / r;
						int v = stab_v / r;

						try {
							g[v]->add_neighbor(v, u, g[u]->node_true_class(), g[u]->noise(), mu_values);
							g[u]->add_neighbor(u, v, g[v]->node_true_class(), g[v]->noise(), mu_values);
						}
						catch (const std::exception& e) {
							// if it does a multi-link or a self-loop I need to start again the geneartion
							std::cout << "exception caught " << e.what() << std::endl;

							// I also need to delete all the links I have already added
							for (int node=0; node < static_cast<int>(g.size()); node++) {
								g[node]->erase_neighbors();
							}

							trial_count++;
							break; // the graph would be an hypergraph, so I need to redraw it
						}

						if (stabs_id.empty()) // I matched all the stabs without self-loops or multiple links (simple graph)
							generated = true;
					}
				}
				else {
					throw std::runtime_error("FATAL ERROR: could not generate a gnr within " + std::to_string(max_count) + " trials");
				}
			}

			set_diameter();
			find_clusters(true);
		}
		else {
			mesh_generator(); // all nodes communicate with all other nodes (only 'flag_neighbor_same' is used)
		}
	}


	// if an arc is invalid it just redraws it (prints the number of redrawn edges)
	void gnr_generator2(int r) { // regular random graph, each node with degree = r

		if ((alg != "colme") && (alg != "colme_recompute")) {

			int trial_count = 0;
			int max_count   = 5; // different meaning as above, here error occurs at the last pair (need to restart)

			bool generated = false;
			while (!generated) {

				if (trial_count < max_count) {

					int redrawn_edges = 0;
					
					std::vector<int> stabs_id(n_nodes * r);           // int number for each N*d stabs of the nodes
					std::iota(stabs_id.begin(), stabs_id.end(), 0);
					
					if (static_cast<int>(stabs_id.size()) % 2 != 0) { // we want an even number of stabs (to match)
						std::cout << "ERROR[c++graph]: specify a even N*d value for gnr generation" << std::endl;
						exit(-1);
					}

					while(stabs_id.size() > 0) { // until all the stabs have been matched (or a self-loop or a multi path is )

						int idx_u  = Rand::int_uniform_rv(0, static_cast<int>(stabs_id.size()-1), seed);
						int stab_u = stabs_id[idx_u];

						Helpers::swap_delete_by_index(stabs_id, idx_u);

						int idx_v = Rand::int_uniform_rv(0, static_cast<int>(stabs_id.size()-1), seed);
						int stab_v = stabs_id[idx_v];
						Helpers::swap_delete_by_index(stabs_id, idx_v);

						int u = stab_u / r;
						int v = stab_v / r;

						try {
							g[v]->add_neighbor(v, u, g[u]->node_true_class(), g[u]->noise(), mu_values);
							g[u]->add_neighbor(u, v, g[v]->node_true_class(), g[v]->noise(), mu_values);
						}
						catch (const std::exception& e) {

							redrawn_edges++;

							// don't do anything and continue to construct the gnr network by extracting a new edge

							if (stabs_id.size() == 2) { // last pair (could be problematic)
								trial_count++;
								break;
							}
						}

						if (stabs_id.empty()) { // I matched all the stabs without self-loops or multiple links (simple graph)
							generated = true;
							std::cout << "gnr generated with " << redrawn_edges << " excluded edges" << std::endl;
						}
					}
				}
				else {
					throw std::runtime_error("FATAL ERROR: could not generate a gnr within " + std::to_string(max_count) + " trials");
				}
			}

			set_diameter();
			find_clusters(true);
		}
		else {
			mesh_generator(); // all nodes communicate with all other nodes (only 'flag_neighbor_same' is used)
		}
	}



	/* Removes the DIRECTED link from 'node_i' to 'node_j' this is done by eliminating 'node_j' from nodes'i adj list
	 *
	 * PARAMETERS:
	 * - node_i, one end of the link to be removed
	 * - node_j, the other end of the link.
	 *
	 * COMPLEXITY: O(n), due to the search in the adjacency vector
	 */
	void remove_directed_link(int node_i, int node_j) { g[node_i]->remove_neighbor(node_j); }


	/* Removes the UNDIRECTED link from 'node_i' to 'node_j' and viceversa
	 *
	 * PARAMETERS:
	 * - node_i, one end of the link to be removed
	 * - node_j, the other end of the link.
	 *
	 * COMPLEXITY: O(n), due to the search in the adjacency vector
	 */
	void remove_link(int node_i, int node_j) {
		remove_directed_link(node_i, node_j);
		remove_directed_link(node_j, node_i);
	}


	/* Add a new DIRECTED link from node_i to node_j (parameters)
	 *
	 * COMPLEXITY: O(n), as we need to avoid duplicates
	 */
	void insert_directed_link(int node_i, int node_j) { 
		g[node_i]->add_neighbor(node_i, node_j, g[node_j]->node_true_class(), g[node_j]->noise(), mu_values);
	}


	/* Add a new UNDIRECTED link from node_i to node_j (parameters)
	 *
	 * COMPLEXITY: O(n), as we need to avoid duplicates
	 */
	void insert_link(int node_i, int node_j) {
		insert_directed_link(node_i, node_j);
		insert_directed_link(node_j, node_i);
	}


	/* generates a regular tree where each node has 'o' offsprings
	 * PARAMETERS:
	 *  - o: number of offsprings of each non-leaf node, the parent will have degree o, the "internal" nodes degree o+1
	 *       and the leafs, clearly 1. The leafs are in the order of o^{l-1}, where l are the levels of the tree
	 */
	void regular_tree_generator(int o) {

		if ((alg != "colme") && (alg != "colme_recompute")) {

			int max_level = find_max_level(o); // note that max level checks if n_nodes > 0
			int parent_level = 0;              // indicates the level of the father we are considering (her eparent exists)
			int all_ancestors = 1;             // the parent node only so far

			while(parent_level < max_level) {

				int num_children = static_cast<int>(std::pow(o, parent_level+1)); // as these are effectively the nodes at the next level
				if ((n_nodes-all_ancestors) < num_children)     // not enough to fill a level with children
					num_children = n_nodes - all_ancestors;     // consider only the residual
				
				int num_parents = static_cast<int>(std::pow(o, parent_level));

				int child_idx  = all_ancestors;
				int parent_idx = (all_ancestors-num_parents);

				int residual_children = num_children;
				for (int p_offset=0; (p_offset<num_parents) && (residual_children>0); p_offset++) {
					for (int c=0; (c<o) && (residual_children>0); c++) {
						insert_link(parent_idx+p_offset, child_idx);
						residual_children--;
						child_idx++;
					}
				}

				all_ancestors += num_children; // the children in this generation become parents in the next one
				parent_level++;
			}

			set_diameter();
			find_clusters(true);
		}
		else { // init in the specific way for colme, where all nodes can communicate with every other node

			mesh_generator(); // all nodes communicate with all other nodes (only 'flag_neighbor_same' is used)
		}
	}


	// Method which retruns the number of DIRECTED links in the graph g, divide by 2 to obtain the undirected value
	int num_directed_links() {
		int e = 0;
		for (int node=0; node < static_cast<int>(g.size()); node++) {
			e += g[node]->num_neighbors();
		}
		return e;
	}


	/* Depth-first-search - recursive method
	 * To have the recursive calls to function properly it is needed to pass the values by reference.
	 * 
	 */
	void depth_first_search(int& node, std::vector<bool>& visited, int& id_c, int& size, int& curr_class) {

		visited[node] = true;     // mark node as visited, in any case irregardless of the node class

		cluster_id[node] = id_c;  // assign the ID of the corresponding cluster
		size++;                   // keep track of cluster size

		if (curr_class < 0) { // clustering irregardless of the classes 

			for (int neighbor : g[node]->neighbors_list()) { // loop over the adjacency matrix of the node

				if (!visited[neighbor]) // visit node if NOT visited yet
					depth_first_search(neighbor, visited, id_c, size, curr_class);
			}
		}
		else {

			for (int neighbor : g[node]->neighbors_list()) { // loop over the adjacency matrix of the node

				if (!visited[neighbor] && (g[neighbor]->node_true_class() == curr_class)) // visit node if NOT visited yet
																						  // and of the same class
					depth_first_search(neighbor, visited, id_c, size, curr_class);
			}
		}
	}


	/* if the number of nodes grows large the above recursive call leads to fill up the available stack memory, here we 
	 * make a iterative version of dfs
	   NOTE: with respect to the other here the first element has a slighlty different semantic
	         becuase here the first element is the staring node and no recursive call is needed
	*/
	void depth_first_search_iterative(int start_node, std::vector<bool>& visited, int& id_c, int& size, int& curr_class) {
    
		std::stack<int> stack;  // using a stack makes you take the last added element and thus going in depth into the graph
		stack.push(start_node);

		while (!stack.empty()) {
			int node = stack.top();
			stack.pop();

			if (!visited[node]) {
				visited[node] = true;
				cluster_id[node] = id_c;
				size++;

				for (int neighbor : g[node]->neighbors_list()) {

					if (curr_class < 0) {
						if (!visited[neighbor]) {
							stack.push(neighbor);
						}
					}
					else { // consider the mu classes for the clusters
						if (!visited[neighbor] && (g[neighbor]->node_true_class() == curr_class)) {
							stack.push(neighbor);
						}
					}
				}
			}
		}
	}



	/* FindClusters
	 *
	 * Finds clusters given the network g (member variable) and assigns the values to the member 'cluster_id'. Moreover,
	 * for each cluster identified by an integer it provides the size of it in the 'cluster_size' member variable
	 * 
	 */
	void find_clusters(bool class_level) {

		std::cout << "\t\t ...finding clusters..." << std::endl;

		if (class_level) {

			for (int c=0; c < static_cast<int>(mu_values.size()); c++) { // we need to find class-specific clusters
				
				int id_cluster = *(std::max_element(cluster_id.begin(), cluster_id.end())); // start from the first available id
				std::vector<bool> visited = std::vector<bool>(n_nodes, false);

				for (int i = 0; i < n_nodes; i++) { // loop over all nodes

					if (g[i]->node_true_class() == c) { // the node needs to be of the class we are interested in

						if (!visited[i]) { // node not visited yet

							int size = 0;  // will be used only for this cluster by the recursive calls
							id_cluster++;

							// depth_first_search(i, visited, id_cluster, size, c);
							depth_first_search_iterative(i, visited, id_cluster, size, c);
							cluster_size.push_back(size);
						}
					}
					else {
						visited[i] = true; // we can consider as visited, and no recursive call needs to be done on it
					}
				}
			}

			// in this case (clust for classes) we can find (note the graph considering the correct class relationships is tatic)
			// the IDs of the largest and second largest connected components (that should be corresponding to the two classes)
			id_clust_0 = id_giant_component();
			id_clust_1 = id_second_largest();

		}
		else {
			int id_cluster = -1; // clusters start from 0
			int dummy_curr = -1; // signals that clusters will be evaluated irregardless of the 'class' of the node
			                     // to be passed to the breath-first search algorithm

			std::vector<bool> visited = std::vector<bool>(n_nodes, false);

			for (int i = 0; i < n_nodes; i++) { // loop over all nodes of the network (we want all components)

				if (!visited[i]) { // node not visited yet

					int size = 0;  // will be used only for this cluster by the recursive calls
					id_cluster++;

					depth_first_search(i, visited, id_cluster, size, dummy_curr);
					cluster_size.push_back(size);
				}
			}
		}
	}


	void print_clusters() {
		std::cout << "---------------- INFO ON CLUSTERS ----------------" << std::endl;
		for (int node=0; node < static_cast<int>(g.size()); node++) {
			std::cout << "\tNode " << node << "\tcl_id=" << cluster_id[node] << std::endl;
		}
	}


	std::vector<int>& read_clusters() { return cluster_id;	}


	/* 
	 * Utility function: return the number of clusters
	 */
	int cluster_number() { return static_cast<int>(cluster_size.size()); }


	/* 
	 * Utility function: returns the id (integer) of the cluster of the giant component
	 * If there are ties it gives the if of the first occurence
	 */
	/* RECALL: cluster size contains the number of nodes for all the clusters ID introduced with the bfs
	 *         therefore the index represents the ID, the largest connected component has id equal to
	 *         the index where the largest value in 'cluster_size' is
	 */
	int id_giant_component() { return Helpers::find_index_max(cluster_size); }


	// similarly, returns the id of the second largest component in the graph (the other large
	// connected component in the case of two classes of nodes)
	int id_second_largest() { return Helpers::find_index_second_max(cluster_size); }



	/************************************** Construct W (only for giant) **********************************************/

	std::vector<std::vector<float>> w_giant() { // can be optimized and rendered faster

		int cluster_id_giant = id_giant_component();
		int giant_size = *std::max_element(cluster_size.begin(), cluster_size.end());

		std::cout << "\tsize of giant component: " << giant_size << std::endl;

		// translating unit, for each idx (node id) saves 0 if the node is not in the component
		// and an integer corresponding to the row/column of that node in the w matrix
		std::vector<int> row_id_node(n_nodes, -1);
		int counter_row = 0;
		for (int node_idx = 0; node_idx < static_cast<int>(g.size()); node_idx++) {
			if (cluster_id[node_idx] == cluster_id_giant) {
				row_id_node[node_idx] = counter_row;
				counter_row++;
			}
		}

		std::vector<std::vector<float>> w_matrix(giant_size, std::vector<float>(giant_size, 0.0f));

		for (int node_idx = 0; node_idx < static_cast<int>(g.size()); node_idx++) {

			int c_id = cluster_id[node_idx];
			if (c_id == cluster_id_giant) { // nodes of the same (giant) cc

				std::vector<float> w_row(giant_size, 0.0f);

				/* Basically do the same procedure you would do for the consensus */
				std::vector<int> true_neigh = g[node_idx]->true_same_class_neighbors();
				std::vector<int> max_den{};
				max_den.reserve(g[node_idx]->num_true_neighbors());

				int self_ca = g[node_idx]->num_true_neighbors();
				float self_coeff = 1.0f;

				for (int i=0; i<static_cast<int>(true_neigh.size()); i++) {
					max_den.push_back(std::max(self_ca, g[true_neigh[i]]->num_true_neighbors()));

					int neigh_idx = true_neigh[i];

					int idx_row = row_id_node[neigh_idx];
					if (idx_row == -1) {
						std::cout << "FATAL ERROR: there appears to be a link to a non-giant node" << std::endl;
						exit(-1);
					}
					w_row[idx_row] = 1.0f / (1.0f + static_cast<float>(max_den[i]));

					self_coeff -= (1.0f / (1.0f + max_den[i]));
				}

				int row_column_curr_node = row_id_node[node_idx];
				w_row[row_column_curr_node] = self_coeff; // the self-loop (diagonal el)

				w_matrix[row_column_curr_node] = w_row;

			}
		}


		std::vector<float> eigen_w = Eigen::compute_eigenvalues(w_matrix);

		// Eigen::print_matrix(w_matrix); // debug

		std::sort(eigen_w.begin(), eigen_w.end(), std::greater<float>());
		std::cout << "********** Sorted eigenvalues of giant component W: **********" << std::endl;
		for (float value : eigen_w) {
			std::cout << value << " ";
		}
		std::cout << std::endl;
		

		return w_matrix;

	}


	/* 
	 * Prints info about all the nodes in the graph for debugging purposes
	 */
	void print_graph_info()
	{
		for (size_t i = 0; i < g.size(); i++) {
			if (std::isnan(g[i]->noise())) {
				std::cout << "Node " << i << " (" << g[i]->node_true_class() <<  ")" << " is connected to: ";
			}
			else {
				std::cout << "Node " << i << " (" << g[i]->node_true_class() <<  ")" << 
					"with noise: " << std::fixed << std::setprecision(4) << g[i]->noise() << ", is connected to: ";
			}
			Helpers::operator<<(std::cout, g[i]->neighbors_list()); // to call the overloaded << operator in Helpers
			std::cout << std::endl;
		}
	}

	void save_graph() {

		std::ofstream out("graph_" + std::to_string(seed) + ".csv");
		out << "node_id;node_class;neighbors_list" << std::endl;

		for (size_t i = 0; i < g.size(); i++) {
			out << i << ";" << g[i]->node_true_class() << ";";
			Helpers::operator<<(out, g[i]->neighbors_list()); // to call the overloaded << operator in Helpers
			out << std::endl;
		}
		
		out.close();
	
	}


	/* 
	 * performs a breath-first-search from 'node_idx_start' and returns the maximum distance (i.e., the 
	 * furthest node in the graph)
	 */
	int breadth_first_search(int node_idx_start, std::vector<bool>& visited) {

        std::queue<std::pair<int, int>> bfs_queue; // hold node ID and distance from the starting node

        bfs_queue.push({node_idx_start, 0});       // start node has distance 0 from itself
        visited[node_idx_start] = true;

		int max_dist = 0; // node at maximum distance from 'node'

        while (!bfs_queue.empty()) {
            auto current = bfs_queue.front(); // take first entry in the queue
            bfs_queue.pop();                  // remove and process it
            max_dist = std::max(max_dist, current.second); // needed then to compute the diameter

            for (int neigh_idx : g[current.first]->neighbors_list()) { // look at all the neighbors of 'current.first'
                if (!visited[neigh_idx]) {
                    bfs_queue.push({neigh_idx, current.second + 1});
                    visited[neigh_idx] = true;
                }
            }
        }
		return max_dist;
    }


	/* 
	 * Computes the diameter of the current network 'g' leveraging the breath_first_search method
	 */
	int compute_diameter() {

		std::cout << "\t\t... computing diameter ..." << std::endl;

		int diam = 0;
        for (int start_node=0; start_node<static_cast<int>(g.size()); start_node++) {
            std::vector<bool> visited(g.size(), false); // visited vector for bfs
            int max_dist =  breadth_first_search(start_node, visited);
            diam = std::max(diam, max_dist);
        }

        return diam;
	}


	// sets the diameter of the current network
	void set_diameter() { diameter = compute_diameter(); }


	/* 
	 * Runs the mean estimation algorithm as prescribed by 'alg' member variable
	 * c_p: nodes to be contacted in the "colme" agorithm at each iteration
	 */
	void run_estimation_agorithm(int max_iter, std::vector<float> mu_values_p, float std_p, bool all_collab, int seed_p, int c_p = -1) {

		bool irrevocable_decision = true; // if true, once a neighbor is excluded it will not be considered anymore
		                                  // this may lead to losing a neigh of the same class, but the probability is small
		bool avg_collab = false;
		bool debug_print = false;

		std::string log_exp = "EXP: " + alg + " (" + std::to_string(seed) + ") " + "N=" + std::to_string(n_nodes) + 
								" epsilon=" + std::to_string(epsilon) + " alpha="+std::to_string(alpha) + 
								" giant ID=" + std::to_string(id_giant_component()) + " second-largest ID=" + std::to_string(id_second_largest());
		Helpers::write_log("log.txt", log_exp);
		Helpers::write_log("log_oracle.txt", log_exp);

		bool topological_change = false; // for consensus detect when (any) node modifies its optimistic neighborhood
										 // used to reset the value of the weight \alpha

	/************************************************** ALGORITHM 1 ***************************************************/

		if (alg == "belief_propagation_v1") { // NodeB

			// debug
			for (size_t j=0; j < g.size(); j++)
				if (g[j]->num_true_neighbors() == 0)
					std::cout << "Node " << j << " is isolated" << std::endl;

			for (int node=0; node<static_cast<int>(g.size()); node++)
				g[node]->init_internal_struct(mu_values, sigma, seed, diameter); // for every neighbor I will init an empty table line

			std::pair<float,float> ep = estimate_and_evaluate(0); // estimate mean (i=0 time instant) and write first record
			                                                      // 'init_internal_struct' initializes both the alg and the oracle

			if (debug_print) {
				std::cout << std::endl << "initialized internal structure" << std::endl;
				print_performance_info(0, ep);
			}

			int prev_prog = -1;
			std::vector<int> isolated_list {};

			int num_of_links = num_directed_links(); // number of directed links in the graph

			for(int iter=1; iter<max_iter; iter++) { // in init the node receive their first sample

				for(int node=0; node<static_cast<int>(g.size()); node++) { // update all nodes
					
					int n_collab = 4 * num_of_links; // 4*2*E
					if (!all_collab) {
						if (avg_collab)
							n_collab = 4 * int_avg_deg;
						else
							n_collab = 4 * g[node]->num_neighbors();
					}


					/*************************************** Discover optimistic neighborhood ****************************************/

					float updated_mean = g[node]->update_mean(iter, mu_values_p, std_p, seed_p); // level 0 of NodeB
					float updated_beta = g[node]->update_beta(std_p, iter, delta, n_collab);

					std::vector<int> new_opt_neighbors {};
					new_opt_neighbors.reserve(g[node]->num_opt_neighbors()); // the 'new' set is necessarily a subset of the old set
					
					std::vector<int> new_opt_neighbors_idx {}; // idx needed to populate the table (as each table row corresponds to a neighbor)
					                                           // here we save the idx in the table associated with new_opt_neighbors[i]
					new_opt_neighbors_idx.reserve(g[node]->num_opt_neighbors()); 

					bool change_in_opt_neighborhood = false; // if changes in the neighborhood occured, deeper records may need to be removed
					                                         // actually deeper values may need to be removed anyways

					if (irrevocable_decision) {
						for(int i=0; i<static_cast<int>(g[node]->neighbors_list().size()); i++) { // loop on all neighbors
							// irrevocable decision -> once I exclude a node I never consider it again
							if (g[node]->is_neighbor_at_opt(i))
								is_new_opt_neighbors(node, new_opt_neighbors, new_opt_neighbors_idx, updated_mean, updated_beta, change_in_opt_neighborhood, i);
						}
					}
					else {
						for(int i=0; i<static_cast<int>(g[node]->neighbors_list().size()); i++)
							is_new_opt_neighbors(node, new_opt_neighbors, new_opt_neighbors_idx, updated_mean, updated_beta, change_in_opt_neighborhood, i);
					}

					// ERASE the table (new) rows associated with neighbors which are no more in the 'optimistic' set
					// NOTE: this encompass also the case in which the node has become isolated (which cannot taken care
					//       in the following portion as we assume that the opt neighborhood has at least 1 element)
					std::vector<int> idx_removed = g[node]->idx_removed_neighbors(); // nodes whose flag (opt) flipped
					if (idx_removed.size() > 0) { // some neighbors have been discarded
						for (auto discarded_row_idx : idx_removed) {
							// erase the row in the table corresponding to the removed neighbor
							g[node]->eliminate_obsolete(discarded_row_idx); // pass the row of the table == index of the neighbor
						}
					}

					/*   Loop over the optimistic neighbors to update the various levels of the mu_k data structure   */

					if (new_opt_neighbors.size() > 0) { // enters if eff_neigh>0 (node not isolated)			

						// at least one (valid) neighbors exist, update level 1 info (k=0) with the neigh local mean
						// I need to do this because the mean, in this implementation, is not in the 'table' structure
						for (int z=0; z<static_cast<int>(new_opt_neighbors.size()); z++) {
							float neigh_mean = g[new_opt_neighbors[z]]->local_mean();
							int neigh_table_idx = new_opt_neighbors_idx[z];
							g[node]->update_internal_struct(neigh_mean, neigh_table_idx, 0, 1); // k=1-hop info is in the first row in the table (depth=0)
							                                                                    // and each value comes from '1' node (-> the neighbor)
						}
						
						int table_depth = 1; // indicates the 'depth' in the neigh_table, table_ix=0 corresponds to k=1-hop info
											 // idx=0 has just been updated with the local means of the (for sure existing) neighbors
											 // I will update vals at 'table_depth' by looking at vals at 'table_depth-1' in the neighbors

						// first loop over ALL the neighbors (like looping over all table entries in the internal structure)
						// then loop over the internal structures of the neighbors (<-> neighbors of neighbors info)
						for (int z=0; z<static_cast<int>(new_opt_neighbors.size()); z++) { // loop over neighbors (<-> table entries)

							// For each node I would like to check its ENTIRE TABLE
							table_depth = 1;
							int neigh = new_opt_neighbors[z];     // id of the neighbor
							int neigh_idx = new_opt_neighbors_idx[z]; // identifies the record in the table of 'node'
							// check in which 'entry' of the neigh's table (if 'node' is present) the 'node' itself is
							int node_idx = g[neigh]->idx_of_neigh(node); // idx of 'node' in adj list of 'neigh' == table index
							                                             // this is the info to be excluded from the table of 'neigh' (self-info)
							
							std::vector<int> neigh_of_neigh_to_check {}; // all optimistic INDICES excluding that associated to 'node'
							neigh_of_neigh_to_check.reserve(g[neigh]->num_opt_neighbors()); // 'non opt' have empty record
							// init the vector, neighbors of neighbors (<-> table entries of 'neigh') to check
							for (int l=0; l<g[neigh]->num_neighbors(); l++) { // loop over 'neigh' table entries 
								if ((l != node_idx) && (g[neigh]->is_neighbor_at_opt(l)))
									neigh_of_neigh_to_check.push_back(l);
							}

							// loop over the 'neigh' table entries which still have info, util table has no more info
							// every time an entry has no more 'deeper' info it is removed from the elements to check

							while ((neigh_of_neigh_to_check.size() > 0) && (table_depth < (table_up_to-1))) {

								float v = 0.0f;
								int num_record = 0; // number of records used, # neighbors info
								int num_of_n_k = 0;
								auto it = neigh_of_neigh_to_check.begin();    // loop from the begginign again

								while (it != neigh_of_neigh_to_check.end()) { // arrive at the end of the index vector
									
									float table_el = g[neigh]->read_struct_at(*it, table_depth-1, false); // it-> entry in the table
									                                                               // **-> depth of info in the table
									if (std::isnan(table_el)) // no more records for table entry, remove idx and update it
										it = neigh_of_neigh_to_check.erase(it);
									else {
										v += table_el;
										num_record++;
										num_of_n_k += static_cast<int>(g[neigh]->read_struct_at(*it, table_depth-1, true)); 
										                                                                                
										it++; // go to next element
									}
								}
								// renormalize and update the value in the table structure
								if (num_record > 0) {
									// I need to pass the sum, not the average value
									g[node]->update_internal_struct(v, neigh_idx, table_depth, num_of_n_k);
									table_depth++; // k-info in neighbors is k+1 in node, also go deeper in neigh table
								}
							}
							// check if there are obsolete records that need to be erased associated with 'neigh_int'
							g[node]->eliminate_obsolete(neigh_idx, table_depth);
						}
					}
					else {

						auto it = std::find(isolated_list.begin(), isolated_list.end(), node);

						if (it == isolated_list.end()) { // not found then add

							isolated_list.push_back(node);

							std::cout << "Node " << node << " became isolated" << std::endl;
							g[node]->print_internal_struct();
							
							std::cout << std::endl << std::endl;
						}
						else {

						}
					}

					

					/************************************************************* Oracle ****************************************/
					
					std::vector<int> true_neigh     = g[node]->true_same_class_neighbors();			
					std::vector<int> true_neigh_idx = g[node]->true_same_class_index();

					if (true_neigh.size() > 0) { // enters if eff_neigh>0 (node not isolated)			

						// at least one (valid) neighbors exist, update level 1 info (k=0) with the neigh local mean
						for (int z=0; z<static_cast<int>(true_neigh.size()); z++) {
							float neigh_mean = g[true_neigh[z]]->local_mean();
							int neigh_table_idx = true_neigh_idx[z]; 
							g[node]->update_oracle_struct(neigh_mean, neigh_table_idx, 0, 1);
						}
						
						int table_depth = 1; // indicates the 'depth' in the neigh_table, table_ix=0 corresponds to k=1-hop info
											 // idx=0 has just been updated with the local means of the (for sure existing) neighbors

						// first loop over ALL the neighbors (like looping over all table entries in the internal structure)
						// then loop over the internal structures of the neighbors (<-> neighbors of neighbors info)
						for (int z=0; z<static_cast<int>(true_neigh.size()); z++) { // loop over neighbors (<-> table entries)

							table_depth = 1;
							int neigh = true_neigh[z];     
							int neigh_idx = true_neigh_idx[z]; 
							// check in which 'entry' of the neigh's table (if 'node' is present) the 'node' itself is
							int node_idx = g[neigh]->idx_of_neigh(node); 
							

							std::vector<int> neigh_of_neigh_to_check {}; // all optimistic INDICES excluding that associated to 'node'
							neigh_of_neigh_to_check.reserve(g[neigh]->num_opt_neighbors()); 
							for (int l=0; l<g[neigh]->num_neighbors(); l++) { // loop over 'neigh' table entries 
								if (l != node_idx && g[neigh]->read_flag_at(l))
									neigh_of_neigh_to_check.push_back(l);
							}
							
							while ((neigh_of_neigh_to_check.size() > 0) && (table_depth < (table_up_to-1))) { 

								float v = 0.0f;
								int num_record = 0; // number of records used, # neighbors info
								int num_of_n_k = 0;
								auto it = neigh_of_neigh_to_check.begin();    // loop from the begginign again

								while (it != neigh_of_neigh_to_check.end()) { // arrive at the end of the index vector
									
									float table_el = g[neigh]->read_oracle_at(*it, table_depth-1, false); 

									if (std::isnan(table_el)) // no more records for table entry, remove idx and update it
										it = neigh_of_neigh_to_check.erase(it);
									else {
										v += table_el;
										num_record++;
										num_of_n_k += static_cast<int>(g[neigh]->read_oracle_at(*it, table_depth-1, true));
										it++; // go to next element
									}
								}
								// renormalize and update the value in the table structure
								if (num_record > 0) {
									g[node]->update_oracle_struct(v, neigh_idx, table_depth, num_of_n_k);
									table_depth++; // k-info in neighbors is k+1 in node, also go deeper in neigh table
								}
							}
						}
					}
				}

				refresh_all_values(); // refresh all the variables that need to be updated

				std::pair<float,float> ep = estimate_and_evaluate(iter);

				if (debug_print)
					print_performance_info(iter, ep);
				
				prev_prog = Helpers::print_progress(iter, prev_prog, max_iter, "\t\t"); // print simulator progression
			}
			// garbage collector: smart pointers de-allocate heap memory automatically when they go out of scope
		}


	/************************************************** ALGORITHM 2 ***************************************************/

		else if (alg == "consensus") { // NodeC

			bool symmetric_matrix = true;
			bool static_alpha = false; // if false, \frac{t}{t+1} is used as sson as the graph has been lernt by the net

			for (int node=0; node<static_cast<int>(g.size()); node++) // init the internal variables (write the first sample)
				g[node]->init_internal_struct(mu_values, sigma, seed);

			std::pair<float,float> ep = estimate_and_evaluate(0); // estimate mean (i=0 time instant)

			if (debug_print) {
				std::cout << std::endl << "initialized internal structure" << std::endl;
				print_performance_info(0, ep);
			}

			int prev_prog = -1;
			bool enter_learned = false;
			int learnt_iter = 0;

			int num_of_links = num_directed_links(); // number of directed links in the graph

			for(int iter=1; iter<max_iter; iter++) { // in init the node receive their first sample

				topological_change = false; // reset the flag

				for(int node=0; node<static_cast<int>(g.size()); node++) { // update all nodes
					
					int n_collab = 4 * num_of_links; // 4*2*E
					if (!all_collab) {
						if (avg_collab)
							n_collab = 4 * int_avg_deg;
						else
							n_collab = 4 * g[node]->num_neighbors();
					}


					/*                                Discover optimistic neighborhood                                */

					float updated_mean = g[node]->update_mean(iter, mu_values_p, std_p, seed_p);
					float updated_beta = g[node]->update_beta(std_p, iter, delta, n_collab);     // NOTE: for the updated one, n_samples = iter+1

					std::vector<int> new_opt_neighbors {};
					new_opt_neighbors.reserve(g[node]->num_opt_neighbors());

					bool change_in_opt_neighborhood = false; // if changes in the neighborhood occured, deeper records may need to be removed

					if (irrevocable_decision) {
						for(int i=0; i<static_cast<int>(g[node]->neighbors_list().size()); i++) { // loop on all neighbors
							// irrevocable decision -> once I exclude a node I never consider it again
							if (g[node]->is_neighbor_at_opt(i)) // thus check only the previouvs 'optimistic' neighbors of 'node'
								is_new_opt_neighbors(node, new_opt_neighbors, updated_mean, updated_beta, change_in_opt_neighborhood, i);
						}
					}
					else { // equivalent of the round-robin approach in [1], all nodes are always recosidered to check their class
						for(int i=0; i<static_cast<int>(g[node]->neighbors_list().size()); i++) // check the neighbors of 'node'
							is_new_opt_neighbors(node, new_opt_neighbors, updated_mean, updated_beta, change_in_opt_neighborhood, i);
					}

					topological_change = change_in_opt_neighborhood || topological_change; // if any node changes its optimistic neighborhood


					if (new_opt_neighbors.size() == 0 && change_in_opt_neighborhood) { // last node has been deleted

						std::string to_log = "node " + std::to_string(node) + " became isolated at" + std::to_string(iter) +
											 " with x=" + std::to_string(updated_mean) +
											 " and y=" + std::to_string(g[node]->read_struct_at());

						Helpers::write_log("isolated_nodes.txt", to_log);
					}

					float v;
					if (symmetric_matrix) {
						// I need to find the maximum C_a (size of the optimistic neighbor) over the neighbors of 'node'
						std::vector<int> max_den{};
						max_den.reserve(new_opt_neighbors.size());
						int self_ca = g[node]->new_num_opt_neighbors();
						float self_coeff = 1.0f;
						for (int i=0; i<static_cast<int>(new_opt_neighbors.size()); i++) {
							max_den.push_back(std::max(self_ca, g[new_opt_neighbors[i]]->num_opt_neighbors()));
							self_coeff -= (1.0f / (1.0f + max_den[i]));
						}

						// compute the new consensus variable, according to the 'symmetric' weights
						v = self_coeff * g[node]->read_struct_at();
						for (int i=0; i<static_cast<int>(new_opt_neighbors.size()); i++) {
							v += (1.0f / (1.0f + max_den[i])) * g[new_opt_neighbors[i]]->read_struct_at();
						}
					}
					else {
						// I can loop over the 'opt' neighbors (if none the node will just consider the "self-record")
						int num_records = 1;
						v = g[node]->read_struct_at();

						for (int i=0; i<static_cast<int>(new_opt_neighbors.size()); i++) {
							v += g[new_opt_neighbors[i]]->read_struct_at();
							num_records++;
						}

						v = (1.0f / static_cast<float>(num_records)) * v;
					}

					if (static_alpha) {
						float cons_v = (1.0f - alpha) * updated_mean + (alpha * v);
						g[node]->update_internal_struct(cons_v);
					}
					else { // \alpha is chosen as \frac{t}{t+1}, where rememeber t=iter
						
						if (graph_learnt) {
							if (!enter_learned) {
								learnt_iter = iter - 1;
								enter_learned = true;
							}
							
							float alpha_v = static_cast<float>(iter - learnt_iter) / (1.0f + iter - learnt_iter);
							float cons_v = (1.0f - alpha_v) * updated_mean + (alpha_v * v);
							g[node]->update_internal_struct(cons_v);
						}
						else { // graph not yet learnt, the nodes use a static alpha

							if (graph_change) { // when flag is true, the normaliation iter is reset
							                    // when the flag stays false, the last value is used
								learnt_iter = iter - 1;
							}
							// NEVER USE THE STATIC VALUE, but a dynamic one resetting at each topological change
							float alpha_v = static_cast<float>(iter - learnt_iter) / (1.0f + iter - learnt_iter);
							float cons_v = (1.0f - alpha_v) * updated_mean + (alpha_v * v);
							g[node]->update_internal_struct(cons_v);
						}
					}

					/************************************************** Oracle structure ******************************/

					std::vector<int> true_neigh = g[node]->true_same_class_neighbors();

					float or_v; // dummy init
					if (symmetric_matrix) {
						if (true_neigh.size() > 0) {
							std::vector<int> max_den{};
							max_den.reserve(true_neigh.size());
							int self_ca = g[node]->num_true_neighbors();
							float self_coeff = 1.0f;
							for (int i=0; i<static_cast<int>(true_neigh.size()); i++) {
								max_den.push_back(std::max(self_ca, g[true_neigh[i]]->num_true_neighbors()));
								self_coeff -= (1.0f / (1.0f + max_den[i]));
							}

							// compute the new consensus variable, according to the 'symmetric' weights
							or_v = self_coeff * g[node]->read_oracle_at();
							for (int i=0; i<static_cast<int>(true_neigh.size()); i++) {
								or_v += (1.0f / (1.0f + max_den[i])) * g[true_neigh[i]]->read_oracle_at();
							}
						}
						else {
							or_v = updated_mean; // x = y, hence (1-a)*x + a*y = x (isolated nodes should do local estimate)
						}
					}
					else {
						int num_records = 1;
						or_v = g[node]->read_oracle_at();

						for (int i=0; i<static_cast<int>(true_neigh.size()); i++) {
							or_v += g[true_neigh[i]]->read_oracle_at();
							num_records++;
						}

						if (num_records > 1) 
							or_v = (1.0f / static_cast<float>(num_records)) * or_v;
						else
							or_v = updated_mean; // x = y, hence (1-a)*x + a*y = x (isolated nodes should do local estimate)
					}

					if (static_alpha) {
						float cons_v = (1.0f - alpha) * updated_mean + (alpha * or_v);
						g[node]->update_oracle_struct(cons_v);
					}
					else { // \alpha is chosen as \frac{t}{t+1}, where rememeber t=iter
						float alpha_v = static_cast<float>(iter) / (1.0f + iter);
						float cons_v = (1.0f - alpha_v) * updated_mean + (alpha_v * or_v);
						g[node]->update_oracle_struct(cons_v);
					}
				}

				refresh_all_values(); // refresh all the variables that need to be updated
				graph_change = topological_change; // update the graph change flag

				std::pair<float,float> ep = estimate_and_evaluate(iter);

				if (debug_print)
					print_performance_info(iter, ep);

				prev_prog = Helpers::print_progress(iter, prev_prog, max_iter, "\t\t"); // print simulator progression
			}
		}


	/************************************************** ALGORITHM 3 ***************************************************/

		else if ((alg == "colme") || (alg == "colme_recompute")) { // NodeA

			bool to_recompute = (alg == "colme_recompute"); // this is the actual O(N^2) algorithm proposed in [1], for each node and at
															// each step entirely RECOMPUTES the set of nodes (also to what concerns
															// previously queried nodes) those that are 'optimistically' close to the node

			for (int node=0; node<static_cast<int>(g.size()); node++) // init the internal table with the k-hop info from neighbors
				g[node]->init_internal_struct(mu_values, sigma, seed, n_nodes, node);
			
			std::pair<float,float> ep = estimate_and_evaluate(0, to_recompute); // estimate mean (i=0 time instant)

			if (debug_print) {
				std::cout << std::endl << "initialized internal structure" << std::endl;
				print_performance_info(0, ep);
			}

			int prev_prog = -1;
			for(int iter=1; iter<max_iter; iter++) { // in init the node receive their first sample

				for(int node=0; node<static_cast<int>(g.size()); node++) { // update all nodes

					int n_collab = 4 * n_nodes; // specifies what 'neighborhood info' to use to parametrize the confidence interval
					                            // in these approaches will aways be (all the nodes)

					float updated_mean = g[node]->update_mean(iter, mu_values_p, std_p, seed_p); // level 0 of NodeBB
					float updated_beta = g[node]->update_beta(std_p, iter, delta, n_collab);

					if (irrevocable_decision) { // restricted round robin

						std::vector<int> neigh_to_contact {}; // 'c_p' is the number of queried nodes, not necessarily optimistic ones
						int neigh_idx = g[node]->read_additional(false); // start index, neigh_idx coincides with (neighbor) ID in this case

						int nodes_query = 0; // avoid repeating the same nodes (if only few are optimistic)

						while (nodes_query < c_p) { // from aove only the check on 'c_p' (now loops_done) changes
							if ((g[node]->read_struct_at(neigh_idx, 1) != -1) &&
									(g[node]->read_struct_at(neigh_idx, 0) != std::numeric_limits<float>::infinity())) { // no the node itself and node already excluded
							                                                   										  // not to be counted in the query count								
								// NOTE: there is a 1-1- correspondance between the nodes IDs in g and the index in the 'internal structure'
								float optimistic_dist = std::abs(updated_mean - g[neigh_idx]->local_mean()) - updated_beta - g[neigh_idx]->read_beta();
								
								// Here I update the optimistic vector of boolean flag (to evaluate wrong and lost neighbors)
								g[node]->set_opt_flag_at((optimistic_dist <= 0.0f), neigh_idx);

								if (optimistic_dist <= 0.0f)
									neigh_to_contact.push_back(neigh_idx);
								else // exclude the node
									g[node]->update_internal_struct(std::numeric_limits<float>::infinity(), neigh_idx, iter);
								
								neigh_idx = (neigh_idx + 1) % n_nodes;
								nodes_query++;
							}
							else
								neigh_idx = (neigh_idx + 1) % n_nodes; // skip the node itself & the 'bad' neighbors (but don't count it the queried nodes)
						}
						g[node]->write_additional(neigh_idx, false); // updated i the while
						for (int w=0; w<static_cast<int>(neigh_to_contact.size()); w++) {
							float neigh_mean = g[neigh_to_contact[w]]->local_mean();
							g[node]->update_internal_struct(neigh_mean, neigh_to_contact[w], iter);
						}

						// oracle update
						std::vector<int> oracle_neigh {};
						int oracle_idx = g[node]->read_additional(true);
						
						int loops_done = 0;
						while (static_cast<int>(oracle_neigh.size())<c_p && loops_done < n_nodes) {
							if (g[node]->read_flag_at(oracle_idx))
								oracle_neigh.push_back(oracle_idx);

							oracle_idx = (oracle_idx + 1) % n_nodes;
							loops_done++;
						}
						g[node]->write_additional(oracle_idx, true); // updated i the while
						for (int w=0; w<static_cast<int>(oracle_neigh.size()); w++) {
							float neigh_mean = g[oracle_neigh[w]]->local_mean();
							g[node]->update_oracle_struct(neigh_mean, oracle_neigh[w], iter);
						}

					}
					else { // round robin, check all the possible nodes (not just the optimistic)
						int neigh_idx = g[node]->read_additional(false);
						int c_p_add = c_p;
						for (int w=0; w<c_p_add; w++) {
							
							if (neigh_idx == node) // exclude the node itself
								c_p_add++;         // add another record
							else { // the neighbor is good to be checked

								// also the previously excluded (pair.first=inf) are checked again, to make the 'non-irrevocable'
								// version of the alg. Thus redo the check everytime!
								float optimistic_dist = std::abs(updated_mean - g[neigh_idx]->local_mean()) - updated_beta - g[neigh_idx]->read_beta();
								
								g[node]->set_opt_flag_at((optimistic_dist <= 0.0f), neigh_idx);

								if (optimistic_dist <= 0.0f) {
									// I can update the structure on the flight
									float neigh_mean = g[neigh_idx]->local_mean();
									g[node]->update_internal_struct(neigh_mean, neigh_idx, iter);
								}
								else { // the node is being exclude (or has been excluded)
									g[node]->update_internal_struct(std::numeric_limits<float>::infinity(), neigh_idx, iter);
								}
							}

							neigh_idx = (neigh_idx+1) % n_nodes;
						}
						g[node]->write_additional((neigh_idx+c_p_add) % n_nodes, false); // deterministic

						// oracle update
						int oracle_idx = g[node]->read_additional(true);
						c_p_add = c_p;
						for (int w=0; w<c_p_add; w++) {
							
							if (oracle_idx == node) // skip the node itself an allow for reading another record
								c_p_add++;         
							else { // the neighbor is good to be checked
								if (g[node]->read_flag_at(oracle_idx)) {
									float neigh_mean = g[oracle_idx]->local_mean();
									g[node]->update_oracle_struct(neigh_mean, oracle_idx, iter);
								}
								// no else is needed, the neighbors not of the smae class are already set to inf at initialization
							}
							oracle_idx = (oracle_idx+1) % n_nodes;
						}
						g[node]->write_additional((oracle_idx+c_p_add) % n_nodes, true); // deterministic
					}
				}

				refresh_all_values(); // refresh all the variables that need to be updated

				std::pair<float,float> ep = estimate_and_evaluate(iter, to_recompute); // uses the appropriate method if "to_recompute is set"

				if (debug_print)
					print_performance_info(iter, ep);

				prev_prog = Helpers::print_progress(iter, prev_prog, max_iter, "\t\t"); // print simulator progression
			}
		}

		else {
			std::cout << "FATAL ERROR[c++:graph]: unsupported algorithm specified" << std::endl;
			exit(1);
		}
	}


	// computes the empirical probability of the nodes which have their estimated mean epsilon-away from the real mean
	// of the distribution they have been assigned to. Returns the value for the algorithm (first) and the oracle (second)
	std::pair<float, float> empirical_prob(int iter) {

		int wrong_th = 10; // should be different for D-COLME and COLME (in the latter many more)

		int under_th = 0;
		int local_under_th = 0;
		int oracle_under_th = 0;
		int cc_under_th = 0;
		std::string oracle_log = "(oracle) ";

		for (int i=0; i<static_cast<int>(g.size()); i++) {

			float true_node_mu = mu_values[g[i]->node_true_class()];
			if (! std::isnan(g[i]->noise())) {
				true_node_mu += g[i]->noise(); // add noise if it was specified
			}
			
			if (std::abs(g[i]->read_estimate() - true_node_mu) <= epsilon) {
				under_th++;
				if ((cluster_id[i] == id_clust_0) || (cluster_id[i] == id_clust_1)) { // node in of of the two cc
					cc_under_th++;
				}
			}
			if (std::abs(g[i]->local_mean() - true_node_mu) <= epsilon)
				local_under_th++;

			if (std::abs(g[i]->read_oracle() - true_node_mu) <= epsilon)
				oracle_under_th++;
			else { // save all "strugglers" save only together with the last optimistic deleted links

				if ((alg != "colme") && (alg != "colme_recompute")) { // don't care much about COLME oracles (super steep)
					oracle_log += "node " + std::to_string(i) +  "; true n: ";
					for (auto& e : g[i]->true_same_class_neighbors()) // the 'true_same_class' would not work for COLME
						oracle_log += std::to_string(e) + ",";
					oracle_log += "\t";
				}
			}
		}
		float emp        = static_cast<float>(under_th) / static_cast<float>(n_nodes);
		float local_emp  = static_cast<float>(local_under_th) / static_cast<float>(n_nodes);
		float oracle_emp = static_cast<float>(oracle_under_th) / static_cast<float>(n_nodes);

		float cc_emp = 0.0f;
		if ((alg != "colme") && (alg != "colme_recompute")) {
			int sum_size_c0_c1 = cluster_size[id_clust_0] + cluster_size[id_clust_1]; // operation otherwise non sensible for the mesh graph
			cc_emp = static_cast<float>(cc_under_th) / static_cast<float>(sum_size_c0_c1);

			wrong_th = 200; // set a higher threshold for colme
		}

		// also evaluate the number of "true" neighbors which are discarded (->lost) and the number of "false" neighbors
		// (i.e., from the opposite class) which are included in the estimation. Here links are considered as directed
		int all_lost  = 0;
		int all_wrong = 0;
		std::string to_log = "";

		for (int i=0; i<static_cast<int>(g.size()); i++) {

			std::pair<int,int> lost_wrong = g[i]->wrong_links();
			all_lost += lost_wrong.first;
			all_wrong += lost_wrong.second;

			// collect info to put in the log (especially about isolated nodes)
			if (lost_wrong.second != 0) {
				if ((alg != "colme") && (alg != "colme_recompute")) {

					to_log += "node " + std::to_string(i) + " [" + std::to_string(cluster_id[i]) + "]" + "; optimistic: ";
					for (auto& e : g[i]->opt_same_class_neighbors())
						to_log += std::to_string(e) + ",";
					to_log += " actual: ";
					for (auto& e : g[i]->true_same_class_neighbors())
						to_log += std::to_string(e) + ",";
					to_log += "\t";
				}
				else {
					std::ostringstream oss; // fix precision of the float val and convert to string
					oss << std::fixed << std::setprecision(5) << std::abs(g[i]->read_estimate() - mu_values[g[i]->node_true_class()]);
					to_log += "n-" + std::to_string(i) + " (" + oss.str() + ");";
				}
			}
		}

		if (all_wrong == 0) { // the graph has been lernt (no more discordant links)
			graph_learnt = true;
		}
		else if (all_wrong < wrong_th) { // record the last nodes which cut their links
			// std::cout << to_log << std::endl;
			Helpers::write_log("log.txt", to_log); // avoids duplicates
			// Helpers::write_log("log_oracle.txt", oracle_log);
		}
		if ((iter > 200) && (alg != "colme") && (alg != "colme_recompute")) // hardcoded value (200), no oracle info for COLME
			Helpers::write_log("log_oracle.txt", oracle_log);

		// save info directly to file
		std::string header = "alg,seed,n,int_avg_deg,sigma,alpha,up_to,epsilon,delta,iter,pe,pl,po,pc,lost_neigh,wrong_neigh";
		// 'pe': empirical correctness probability as for the 'alg'
		// 'pl': empirical correctness of the local estimation of the local estimator (should be the same for all approaches)
		// 'po': empirical correctness of the oracle estimator (the one which knows the 'correct' neighbors from i=0)
		// 'pc': empirical correctness probability as for the 'alg' restricted to the 2 largest connected components (groups)
		std::string csv_line = alg + "," + std::to_string(seed) + "," + std::to_string(n_nodes) + "," + std::to_string(int_avg_deg) +
								"," + std::to_string(sigma) + "," + std::to_string(alpha) + "," + std::to_string(table_up_to) +
								"," + std::to_string(epsilon) + "," + std::to_string(delta) + "," + std::to_string(iter) + "," + std::to_string(emp) +
								"," + std::to_string(local_emp) + "," + std::to_string(oracle_emp) + "," + std::to_string(cc_emp) +
								"," + std::to_string(all_lost) + "," + std::to_string(all_wrong);
		save_info_to_csv("preliminary.csv", header, csv_line);

		return std::make_pair(emp, oracle_emp); // I do not return the local mean, it is not necessary
	}


private:
	// utility function which finds the number of levels (number of "offsprings generations") in a regular tree
	int find_max_level(int offsprings) {
		int l_max = 0; // tmp value for the number of levels
		if (offsprings == 0 || n_nodes == 0) {
			std::cout << "ERROR[c++graph]: trying to initialize an empty regular tree" << std::endl;
			exit(-1);
		}
		else {
			int population_count = 0; // total number of parents and child in the tree up to level l_max
			while (population_count < n_nodes) {
				population_count += static_cast<int>(std::pow(offsprings, l_max));
				l_max++;
			}
			return l_max - 1; // level 0 contains the father, and level 1 the first batch of children and so on
		}
	}


	// utility function which returns the list of 'optimistic' neighbprs and some 'additional info'
	// for NodeBB the size of the neighborhoods of the neighbors (or Nan if the neighbor does not see the node)
	// 'i_p' is the index of the neighbor in 'node' adjacency list
	void is_new_opt_neighbors(int node_p, std::vector<int>& new_opt_neighbors_p, std::vector<int>& add_info,
								float updated_mean_p, float updated_beta_p, bool& change_in_opt_neighborhood_p, int i_p) {

		const int neighbor = g[node_p]->read_neighbor_at(i_p); // index in 'g' of the neighbor
		float optimistic_dist = std::abs(updated_mean_p - g[neighbor]->local_mean()) - updated_beta_p - g[neighbor]->read_beta();

		// also update the internal structure with the classification (optimistic or not) of the neighbors
		// I want the update flag part to be ALWAYS performed! (beware of c++ shortcitcuiting in logical operations)
		change_in_opt_neighborhood_p = g[node_p]->update_neighbor_flag(i_p, optimistic_dist <= 0.0f) || change_in_opt_neighborhood_p;

		if (optimistic_dist <= 0.0f) {

			new_opt_neighbors_p.push_back(neighbor); // add neighbor to the optimistic set

			if (alg == "belief_propagation_v1") {
				add_info.push_back(i_p); // save the index of the 'optimistic' neighbor (idx of table entry)
			}
			else {
				std::cout << "FATAL ERROR: not supported yet!" << std::endl;
				exit(-1);
			}

		}
	}

	// same function as before but with different number of parameters (no 'add_info'), not needed here (works on Node instances)
	void is_new_opt_neighbors(int node_p, std::vector<int>& new_opt_neighbors_p,
								float updated_mean_p, float updated_beta_p, bool& change_in_opt_neighborhood_p, int i_p) {

		const int neighbor = g[node_p]->read_neighbor_at(i_p); // index in 'g' of the neighbor
		float optimistic_dist = std::abs(updated_mean_p - g[neighbor]->local_mean()) - updated_beta_p - g[neighbor]->read_beta();

		// also update the internal structure with the classification (optimistic or not) of the neighbors
		change_in_opt_neighborhood_p = g[node_p]->update_neighbor_flag(i_p, optimistic_dist <= 0.0f) || change_in_opt_neighborhood_p;

		if (optimistic_dist <= 0.0f)
			new_opt_neighbors_p.push_back(neighbor);
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


	// for the "colme" algorithm where all nodes can see all other nodes
	// I aslo need a structure to keep the optimistic neighbors in order to understand how good and fast the algorithm learns
	void mesh_generator() {

		for (int node=0; node<n_nodes; node++) { // loop over the nodes
			g[node]->flag_reserve(n_nodes);      // reserve N values in the true and opt structures

			int node_class = g[node]->node_true_class();
			if (std::isnan(g[node]->noise())) {
				for (int neigh=0; neigh<n_nodes; neigh++) {  // loop over the possible neighbors
					bool check = (node_class==g[neigh]->node_true_class());
					g[node]->set_flag_at(check, neigh);
					g[node]->set_opt_flag_at(true, neigh); // init the nodes as all neighbors in the beginning
				}
				g[node]->set_flag_at(false, node); // manage the self-loop without the if
				g[node]->set_opt_flag_at(false, node);
			}
			else {
				for (int neigh=0; neigh<n_nodes; neigh++) {  // loop over the possible neighbors
					float mu_node  = mu_values[node_class] + g[node]->noise();
					float mu_neigh = mu_values[g[neigh]->node_true_class()] + g[neigh]->noise();
					bool check = Helpers::are_equal(mu_node, mu_neigh);
					g[node]->set_flag_at(check, neigh);
					g[node]->set_opt_flag_at(true, neigh); // init the nodes as all neighbors in the beginning
				}
				g[node]->set_flag_at(false, node); // manage the self-loop without the if
				g[node]->set_opt_flag_at(false, node);
			}

		}
	}


	// for each node performs the mean and the oracle estimate and computes the empirical proabbility of correct estimation
	// which is also saved to file (done also for i=0: init with first sample)
	std::pair<float,float> estimate_and_evaluate(int iter_p, bool recompute=false) {
		if (recompute){ // recompute the optimistic sets at each estimation
			for (int node=0; node< static_cast<int>(g.size()); node++) {
				g[node]->estimate_mean(iter_p, sigma, delta);
				g[node]->oracle_estimate(iter_p, sigma, delta);
			}
		}
		else {
			for (int node=0; node< static_cast<int>(g.size()); node++) {
				g[node]->estimate_mean(iter_p);
				g[node]->oracle_estimate(iter_p);
			}
		}
		
		return empirical_prob(iter_p); // compute empirical probabilities and save
	}


	void print_performance_info(int iter_p, std::pair<float,float> ep_p) {

		// print (if flag is set) the empirical probability of correct estimation of the 'algorithm' estimate and the 'oracle' estimate
		std::cout << std::endl << "i: " << iter_p << " Pr[alg]=" << std::fixed << std::setprecision(4) << ep_p.first << 
			" Pr[oracle]=" << std::fixed << std::setprecision(4) << ep_p.second << std::endl;

		// debug information on the internal structure and on the estimates of the algorithm and of the oracle. Moreover, the 
		// locla mean (l.m.) is reported between parenthesis
		for (int node=0; node< static_cast<int>(g.size()); node++) {
			std::cout << "\tNode " << node << " (l.m." << std::fixed << std::setprecision(4) << g[node]->local_mean() << ") internal info: ";
			g[node]->print_internal_struct();
			std::cout << "\testimate: " << g[node]->read_estimate() << " oracle: " << g[node]->read_oracle() << std::endl;
		}
		std::cout << std::endl;					
	}


	void refresh_all_values() {

		if ((alg == "colme") || (alg == "colme_recompute")) { // no explicit neighbors vectors are saved (all nodes are in principle reachable)

			for (int node=0; node<static_cast<int>(g.size()); node++) { // refresh all the variables that need to be updated
				g[node]->refresh_beta();
				g[node]->refresh_internal_struct();
			}
		}
		else { // here I have also to update the falgs of the optimistic neighbors

			for (int node=0; node<static_cast<int>(g.size()); node++) { // refresh all the variables that need to be updated
				g[node]->refresh_beta();
				g[node]->refresh_internal_struct();
				g[node]->refresh_neigh_flag();
			}
		}
	}
};

