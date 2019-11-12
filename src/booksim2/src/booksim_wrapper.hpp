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

  int IssueMessage(int flow, int src, int dest, int id, Message::MessageType type);
  pair<int, int> PeekMessage(int node, int vnet);

  bool RunTest();
  inline bool Idle() { return _traffic_manager->Idle() && _outstanding_messages == 0; }
  inline void WakeUp() { _traffic_manager->WakeUp(); }
  inline void SetSimTime(int time) { _traffic_manager->setTime(time); }
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

