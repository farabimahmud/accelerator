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
  for (int root = 0; root < 16; root++) {
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

  for (int root = 0; root < 16; root++) {
    int depth = scan[root].size() - 1;
    for (int row = 1; row < depth; row++) {
      if (row > links_tree_map.size()) {
        std::cout << "push back a new map for row " << row << std::endl;
        links_tree_map.resize(row, std::map<std::string, int>());
        conflicts.resize(row, 0);
        std::cout << "added a new map for row " << row << std::endl;
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
            std::cerr << "Conflict foud for link " << link
              << " between tree " << links_tree_map[row - 1][link]
              << " and tree " << root << std::endl;
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

int main() {
  std::vector<float> link_matrix(16*16);

  int pci_link = 1;
  int nvlink = 2;
  int link_weight = nvlink;//pci_link;

  // link topology weight matrix for torus
  for (int n = 0; n < 16; n++) {
    int row = n / 4;
    int col = n % 4;

    int north = (row == 0) ? n + 12 : n - 4;
    link_matrix[n * 16 + north] = link_weight;
    link_matrix[north * 16 + n] = link_weight;

    int south = (row == 3) ? n - 12 : n + 4;
    link_matrix[n * 16 + south] = pci_link;
    link_matrix[south * 16 + n] = pci_link;

    int west = (col == 0) ? n + 3 : n - 1;
    link_matrix[n * 16 + west] = link_weight;
    link_matrix[west * 16 + n] = link_weight;

    int east = (col == 3) ? n - 3 : n + 1;
    link_matrix[n * 16 + east] = link_weight;
    link_matrix[east * 16 + n] = link_weight;
  }
  assert(IsConnected(link_matrix, 16));

  std::vector<std::vector<size_t>> topology;
  std::vector<std::vector<size_t>> scan;

  // Compute trees using mxnet's multi-tree collective communicaiton
  ComputeTrees(link_matrix, 16, 0.7, false, &topology, &scan);

  // detect and summarize the link conflicts
  link_conflict_detection(topology, scan);

  // resolve link conflicts
  //resolve_link_conflicts(topology, scan);

  // generate the dot file for kl trees
  generate_trees_dotfile("kl_trees.dot", topology, scan);

  return 0;
}
