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

  int IssueMessage(int flow, int src, int dest, int id, int msg_size, Message::MessageType type); // msg_size: bytes
  tuple<int, int, Message::MessageType> PeekMessage(int node, int vnet);
  void DequeueMessage(int node, int vnet);

  bool RunTest();
  inline bool Idle() { return _traffic_manager->Idle() && _outstanding_messages == 0; }
  inline void WakeUp() { _traffic_manager->WakeUp(); }
  inline void SetSimTime(int time) { _traffic_manager->setTime(time); }
  inline int GetSimTime() { return _traffic_manager->getTime(); }
  inline int GetMessageBufferSize() { return _config->GetInt("msg_buf_size"); }
  inline int GetInjectBufferSize() { return _config->GetInt("inject_buf_size"); }

  inline void RegisterGlobalTrafficManager() {trafficManager = _traffic_manager;}

private:
  ScaleTrafficManager *_traffic_manager;
  BookSimConfig *_config;
  vector<Network *> _net;

  int _cur_mid;
  int _outstanding_messages;
  bool _print_messages;
};

#endif // _BOOKSIM_WRAPPER_HPP_

