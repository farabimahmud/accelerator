#include "booksim_wrapper.hpp"
#include "routefunc.hpp"
#include "random_utils.hpp"
#include "power_module.hpp"
#include "dsent_power.hpp"

BookSim::BookSim(string const & configfile)
{
  _config = new BookSimConfig();
  _config->ParseFile(configfile);

  // initialize routing and global variables
  InitializeRoutingMap(*_config);

  gPrintActivity = (_config->GetInt("print_activity") > 0);
  gTrace = (_config->GetInt("viewer_trace") > 0);

  string watch_out_file = _config->GetStr("watch_out");
  if (watch_out_file == "") {
    gWatchOut = nullptr;
  } else if (watch_out_file == "-") {
    gWatchOut = &cout;
  } else {
    gWatchOut = new ofstream(watch_out_file.c_str());
  }

  // create network
  _net.push_back(nullptr);
  _net[0] = Network::New(*_config, "scale_network");

  // create traffic manager
  _traffic_manager = new ScaleTrafficManager(*_config, _net);

  _cur_mid = 0;
  _outstanding_messages = 0;
  _print_messages = false;

#ifdef LIB_BOOKSIM
  gBookSim = this;
#endif
}

BookSim::~BookSim() {
  if (_config->GetInt("sim_power") > 0) {
    if (_config->GetInt("dsent_power") > 0) {
      DSENTPower pnet(_net[0], *_config);
      pnet.Run();
    } else {
      Power_Module pnet(_net[0], *_config);
      pnet.run();
    }
  }

  delete _net[0];
  delete _config;
  delete _traffic_manager;
}

int BookSim::IssueMessage(int src, int dest, int id,  Message::MessageType type)
{
  if (id == -1) {
    id = _cur_mid++;
  }

  Message *message = Message::New(type, id, src, dest);
  _traffic_manager->Enqueue(message);
  _outstanding_messages++;

  if (_print_messages) {
    *gWatchOut << GetSimTime() << " | node" << src << " | "
      << "Issuing message:" << endl;
   *gWatchOut << *message;
  }

  return id;
}

int BookSim::PeekMessage(int node, int vnet)
{
  int id = -1;
  Message *message = _traffic_manager->Dequeue(node, vnet);
  if (message != nullptr) {
    id = message->id;

    if (_print_messages) {
      *gWatchOut << GetSimTime() << " | node" << node << " | "
        << "Receiving messages:" << endl;
     *gWatchOut << *message;
    }

    message->Free();
    _outstanding_messages--;
  }

  return id;
}

bool BookSim::RunTest()
{
  _print_messages = true;

  for (int src = 0; src < gNodes; src++) {
    for (int dest = 0; dest < gNodes; dest++) {
      Message::MessageType type = (Message::MessageType) RandomInt(Message::MessageType_NUM - 1);
      IssueMessage(src, dest, -1, type);
    }
  }

  while (!_traffic_manager->Idle() || _outstanding_messages > 0) {
    _traffic_manager->WakeUp();
    for (int node = 0; node < gNodes; node++) {
      for (int vnet = 0; vnet < 2; vnet++) {
        PeekMessage(node, vnet);
      }
    }
  }

  _traffic_manager->UpdateStats();
  _traffic_manager->DisplayStats();

  return (_outstanding_messages == 0);
}

