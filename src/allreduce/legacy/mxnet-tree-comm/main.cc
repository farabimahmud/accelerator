#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "./kernighan_lin.h"

using namespace mxnet::kvstore;

void generate_trees_dotfile(std::string filename,
    std::vector<std::vector<size_t>> &topology,
    std::vector<std::vector<size_t>> &scan) {

  std::map<int, std::vector<std::string>> ranks;
  std::vector<std::string> colors = {"#f7f4f9", "#e7e1ef", "#d4b9da", "#c994c7",
    "#df65b0","#e7298a", "#ce1256", "#980043", "#67001f"};

  std::ofstream trees_dot_out;
  trees_dot_out.open(filename.c_str());

  trees_dot_out << "digraph tree {" << std::endl
    << "  rankdir = BT;" << std::endl
    << "  subgraph {" << std::endl;

  int max_depth = -1;
  ranks[0] = std::vector<std::string>();
  for (int root = 0; root < scan.size(); root++) {
    trees_dot_out << "    /* tree " << root << " */" << std::endl;
    std::string node = "\"" + std::to_string(root) + "-" + std::to_string(root) + "\"";
    ranks[0].push_back(node);
    int depth = scan[root].size() - 1;
    if (depth > max_depth) max_depth = depth;
    for (int row = 1; row < depth; row++) {
      int start = scan[root][row];
      int end = scan[root][row + 1];
      int iteration = depth - row;
      for (; start < end; start++) {
        if (start % 2 == 0) {
          int parent = topology[root][start - 1];
          int child = topology[root][start];
          trees_dot_out << "    \"" << root << "-" << child << "\" -> "
            << "\"" << root << "-" << parent << "\" "
            << "[ label=\"" << iteration << "\" ];" << std::endl;
          if (ranks.find(row) == ranks.end()) {
            ranks[row] = std::vector<std::string>();
          }
          node = "\"" + std::to_string(root) + "-" + std::to_string(child) + "\"";
          ranks[row].push_back(node);
        }
      }
    }
  }

  trees_dot_out << "    // note that rank is used in the subgraph\n";
  for (int rank = 0; rank < max_depth; rank++) {
    if (ranks.find(rank) != ranks.end()) {
      std::string level = "    {rank = same;";
      for (int i = 0; i < ranks[rank].size(); i++) {
        level += " ";
        level += ranks[rank][i];
        level += ";";
      }
      level += "}\n";
      trees_dot_out << level;
    }
  }

  trees_dot_out << "    // node colors\n";
  for (int rank = 0; rank < max_depth; rank++) {
    if (ranks.find(rank) != ranks.end()) {
      for (int i = 0; i < ranks[rank].size(); i++) {
        trees_dot_out << "    " << ranks[rank][i]
          << " [style=\"filled\", fillcolor=\""
          << colors[rank] << "\"];\n";
      }
    }
  }

  trees_dot_out << "  } /* closing subgraph */\n";
  trees_dot_out << "}\n";

  trees_dot_out.flush();
  trees_dot_out.close();
}

void link_conflict_detection(std::vector<std::vector<size_t>> &topology,
    std::vector<std::vector<size_t>> &scan) {
  std::vector<std::map<std::string, int>> links_tree_map;
  std::vector<int> conflicts;

  for (int root = 0; root < scan.size(); root++) {
    int depth = scan[root].size() - 1;
    for (int row = 1; row < depth; row++) {
      if (row > links_tree_map.size()) {
        //std::cout << "push back a new map for row " << row << std::endl;
        links_tree_map.resize(row, std::map<std::string, int>());
        conflicts.resize(row, 0);
        //std::cout << "added a new map for row " << row << std::endl;
      }
      int start = scan[root][row];
      int end = scan[root][row + 1];
      int iteration = depth - row;
      for (; start < end; start++) {
        if (start % 2 == 0) {
          int parent = topology[root][start - 1];
          int child = topology[root][start];
          std::string link = std::to_string(child) + "->" + std::to_string(parent);
          if (links_tree_map[row - 1].find(link) != links_tree_map[row - 1].end()) {
            //std::cerr << "Conflict foud for link " << link
            //  << " between tree " << links_tree_map[row - 1][link]
            //  << " and tree " << root << std::endl;
            conflicts[row - 1]++;
          } else {
            links_tree_map[row - 1][link] = root;
          }
        }
      }
    }
  }

  std::cout << "Conflicts summary:\n";
  for (size_t i = 0; i < conflicts.size(); i++) {
    std::cout << "  row " << i + 1 << ": " << conflicts[i] << std::endl;
  }
}

int main(int argc, char **argv) {

  int dimension;
  bool backtrack;
  if (argc == 1) {
    dimension = 4;
    backtrack = false;
  } else if (argc == 2) {
    dimension = atoi(argv[1]);
    backtrack = false;
  } else if (argc == 3) {
    dimension = atoi(argv[1]);
    backtrack = atoi(argv[2]) > 0;
  } else {
    std::cerr << "Usage: ./kernighan_lin <dimension> <backtrack>" << std::endl;
    exit(0);
  }

  int nodes = dimension * dimension;

  std::cout << "network dimension is " << dimension
    << ", network size: " << nodes;
  if (backtrack)
    std::cout << ", backtracking" << std::endl;
  else
    std::cout << ", no backtrack" << std::endl;

  std::vector<float> link_matrix(nodes*nodes);

  int pci_link = 1;
  int nvlink = 2;
  int link_weight = nvlink;//pci_link;

  // link topology weight matrix for torus
  for (int n = 0; n < nodes; n++) {
    int row = n / dimension;
    int col = n % dimension;

    int north = (row == 0) ? n + dimension * (dimension - 1) : n - dimension;
    link_matrix[n * nodes + north] = link_weight;
    link_matrix[north * nodes + n] = link_weight;

    int south = (row == 3) ? n - dimension * (dimension - 1) : n + dimension;
    link_matrix[n * nodes + south] = pci_link;
    link_matrix[south * nodes + n] = pci_link;

    int west = (col == 0) ? n + dimension - 1 : n - 1;
    link_matrix[n * nodes + west] = link_weight;
    link_matrix[west * nodes + n] = link_weight;

    int east = (col == 3) ? n - dimension + 1 : n + 1;
    link_matrix[n * nodes + east] = link_weight;
    link_matrix[east * nodes + n] = link_weight;
  }
  assert(IsConnected(link_matrix, nodes));

  std::vector<std::vector<size_t>> topology;
  std::vector<std::vector<size_t>> scan;

  // Compute trees using mxnet's multi-tree collective communicaiton
  ComputeTrees(link_matrix, nodes, 0.7, backtrack, &topology, &scan);

  // detect and summarize the link conflicts
  link_conflict_detection(topology, scan);

  // resolve link conflicts
  //resolve_link_conflicts(topology, scan);

  // generate the dot file for kl trees
  generate_trees_dotfile("kl_trees.dot", topology, scan);

  return 0;
}
