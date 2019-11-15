#include "message.hpp"

stack<Message *> Message::_all;
stack<Message *> Message::_free;

ostream& operator<<(ostream &os, const Message &m)
{
  os << "  Message ID: " << m.id << " (" << &m << ")"
    << " Type: " << Message::MessageTypeToString(m.type) << endl;
  os << "  Flow: " << m.flow << " Source: " << m.src << "  Dest: " << m.dest
    << "  vnet: " << m.vnet << endl;

  return os;
}

Message::Message()
{
  Reset();
}

void Message::Set(MessageType type_, int id_, int flow_, int src_, int dest_)
{
  type = type_;
  vnet = GetVirtualNetwork(type);
  id = id_;
  flow = flow_;
  src = src_;
  dest = dest_;
}

void Message::Reset()
{
  type = MessageType_NUM;
  vnet = -1;
  id = -1;
  flow = -1;
  src = -1;
  dest = -1;
}


Message * Message::New() {
  Message * m;
  if (_free.empty()) {
    m = new Message;
    _all.push(m);
  } else {
    m = _free.top();
    m->Reset();
    _free.pop();
  }

  return m;
}

Message * Message::New(MessageType type, int id, int flow, int src, int dest)
{
  Message * m;
  if (_free.empty()) {
    m = new Message;
    _all.push(m);
  } else {
    m = _free.top();
    _free.pop();
  }
  m->Set(type, id, flow, src, dest);

  return m;
}

string Message::MessageTypeToString(const MessageType &type)
{
  switch (type) {
    case ReadRequest:
      return "ReadRequest";
    case ReadReply:
      return "ReadReply";
    case WriteRequest:
      return "WriteRequest";
    case WriteReply:
      return "WriteReply";
    case ControlRequest:
      return "ControlRequest";
    case ControlReply:
      return "ControlReply";
    case ReduceData:
      return "ReduceData";
    case GatherData:
      return "GatherData";
    default:
      cerr << "Error: Unknown Message Type " << type << endl;
      exit(-1);
  }
}

int Message::GetVirtualNetwork(const MessageType &type)
{
  switch (type) {
    case ReadRequest:
    case WriteRequest:
    case ControlRequest:
    case ReduceData:
    case GatherData:
      return 0;
    case ReadReply:
    case WriteReply:
    case ControlReply:
      return 1;
    default:
      cerr << "Error: Unknown Message Type " << type << endl;
      exit(-1);
  }
}

int Message::GetMessageSize(const MessageType &type)
{
  switch (type) {
    case ReadReply:
    case WriteRequest:
    case ReduceData:
    case GatherData:
      return 64 + 8;
    case ReadRequest:
    case WriteReply:
    case ControlRequest:
    case ControlReply:
      return 8;
    default:
      cerr << "Error: Unknown Message Type " << type << endl;
      exit(-1);
  }
}

void Message::Free()
{
  _free.push(this);
}

void Message::FreeAll()
{
  while (!_all.empty()) {
    delete _all.top();
    _all.pop();
  }
}
