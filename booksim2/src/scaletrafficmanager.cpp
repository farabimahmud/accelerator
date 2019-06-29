#include <cmath>
#include <sstream>
#include <limits>
#include "scaletrafficmanager.hpp"

ScaleTrafficManager::ScaleTrafficManager(const Configuration &config, const
    vector<Network *> &net, int vnets)
  : TrafficManager(config, net)
{
  _vnets = vnets;

  _sim_state = running;
}

ScaleTrafficManager::~ScaleTrafficManager() {}

void ScaleTrafficManager::_RetireFlit(Flit *f, int dest)
{
  _deadlock_timer = 0;

  // send to the output message buffer
  if (f->tail) {
    _output_buffer[dest][f->vnet].push_back(f->msg);
    if (f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
        << " HMC-" << dest
        << " consumes the packet " << f->pid
        << " with message " << *(f->msg) << "." << endl;
    }
  }

  TrafficManager::_RetireFlit(f, dest);
}

void ScaleTrafficManager::_GeneratePacket(int source, int stype, int vnet, int time)
{
  int cl = 0;

  Message *message = _input_buffer[source][vnet].front();

  Flit::FlitType packet_type = Flit::ANY_TYPE;

  int size = (int) ceil((double) Message::GetMessageSize(message->type)*8 / _flit_size);

  int packet_dest = message->dest;
  bool record = true;

  bool watch = gWatchOut && (_packets_to_watch.count(_cur_pid) > 0);
  watch |= _watch_all_pkts;

  switch (stype) {
    case Message::ReadRequest:
    case Message::WriteRequest:
    case Message::ControlRequest:
      packet_type = Flit::REQUEST;
      break;
    case Message::ReadReply:
    case Message::WriteReply:
    case Message::ControlReply:
      packet_type = Flit::REPLY;
      break;
    default:
      ostringstream err;
      err << "Unknown message type " << stype;
      Error(err.str());
  }

  if (packet_dest < 0 || packet_dest >= _nodes) {
    ostringstream err;
    err << "Incorrect packet destination " << packet_dest
      << " for message type " << Message::MessageTypeToString(Message::MessageType(stype));
    Error(err.str());
  }

  int subnetwork = 0;
  
  if (watch) {
    *gWatchOut << GetSimTime() << " | "
      << "node" << source << " | "
      << "Enqueuing packet " << _cur_pid
      << " at time " << time
      << " through router " << source
      << " (to node " << packet_dest
      << " attached to router " << packet_dest
      << ")." << endl;
  }

  int pid = _cur_pid++;

  for (int i = 0; i < size; i++) {
    Flit * f = Flit::New();
    f->id = _cur_id++;
    assert(_cur_id);
    f->pid = pid;
    f->watch = watch | (gWatchOut && (_flits_to_watch.count(f->id) > 0));
    f->subnetwork = subnetwork;
    f->src = source;
    f->ctime = time;
    f->record = record;
    f->cl = cl;
    f->dest = packet_dest;
    f->type = packet_type;
    f->vnet = vnet;
    f->msg = message;

    _total_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    if (record) {
      _measured_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    }

    if (gTrace) {
      cout << "New Flit " << f->src << endl;
    }

    if (i == 0) { // Headd flit
      f->head = true;
      f->dest = packet_dest;
    } else {
      f->head = false;
      f->dest = -1;
    }
    switch (_pri_type) {
      case class_based:
        f->pri = cl;
        break;
      case age_based:
        f->pri = numeric_limits<int>::max() - time;
        break;
      case sequence_based:
        f->pri = numeric_limits<int>::max() - _packet_seq_no[source];
        break;
      default:
        f->pri = 0;
    }
    assert(f->pri >= 0);

    if (i == (size - 1)) { // Tail flit
      f->tail = true;
    } else {
      f->tail = false;
    }

    f->vc = -1;

    if (f->watch) {
      *gWatchOut << GetSimTime() << " | "
        << "node " << source << " | "
        << "Enqueuing flit " << f->id
        << " (packet " << f->pid
        << ") at time " << time
        << "." << endl;
    }

    _partial_packets[source][cl].push_back(f);
  }
  assert(_cur_pid);
}

void ScaleTrafficManager::_Inject()
{
  for (int input = 0; input < _nodes; input++) {
    assert(_classes == 0);

    int const last_vnet = _last_vnet[input];

    for (int v = 1; v <= _vnets; v++) {

      int vnet = (last_vnet + v) % _vnets;

      if (_input_buffer[input][vnet].empty() == false) {
        Message *message = _input_buffer[input][vnet].front();
        Message::MessageType packet_type = message->type;
        _GeneratePacket(input, packet_type, vnet, _time);
        if (_watch_all_pkts) {
          *gWatchOut << GetSimTime() << " | " << FullName()
            << " | " << "HMC-" << input
            << " generate new packets for message: " << endl;
          *gWatchOut << *message;
        }
        _input_buffer[input][vnet].pop_front();
        _last_vnet[input] = vnet;
      }
    }
  }
}
