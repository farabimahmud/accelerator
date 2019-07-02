#ifndef _BOOKSIM_WRAPPER_HPP_
#define _BOOKSIM_WRAPPER_HPP_

#include "booksim.hpp"
#include "message.hpp"
#include "booksim_config.hpp"
#include "network.hpp"
#include "scaletrafficmanager.hpp"

#include <string>

class BookSim {

public:
  BookSim(string const & configfile);
  ~BookSim();

  int IssueMessage(int src, int dest, int id, Message::MessageType type);
  int PeekMessage(int node, int vnet);

  bool RunTest();
  inline int GetSimTime() { return _traffic_manager->getTime(); }

  inline static void RegisterGlobalBookSim(BookSim *booksim) {gBookSim = booksim;}

private:
  ScaleTrafficManager *_traffic_manager;
  BookSimConfig *_config;
  vector<Network *> _net;

  int _cur_mid;
  int _outstanding_messages;
  bool _print_messages;
};

#endif // _BOOKSIM_WRAPPER_HPP_

