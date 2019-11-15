#ifndef _MESSAGE_HPP_
#define _MESSAGE_HPP_

#include <iostream>
#include <stack>

#include "booksim.hpp"

class Message {

public:

  enum MessageType {
    ReadRequest    = 0,
    ReadReply      = 1,
    WriteRequest   = 2,
    WriteReply     = 3,
    ControlRequest = 4,
    ControlReply   = 5,
    ReduceData     = 6,
    GatherData     = 7,
    MessageType_NUM
  };

  MessageType type;

  int vnet;

  int id;

  int flow;
  int src;
  int dest;

  void Reset();
  void Set(MessageType type_, int id_, int flow_, int src_, int dest_);

  static Message *New();
  static Message *New(MessageType type, int id, int flow, int src, int dest);
  static string MessageTypeToString(const MessageType &type);
  static int GetVirtualNetwork(const MessageType &type);
  static int GetMessageSize(const MessageType &type);
  void Free();
  static void FreeAll();

private:

  Message();
  ~Message() {}

  static stack<Message *> _all;
  static stack<Message *> _free;
};

ostream& operator<<(ostream &os, const Message &m);

#endif // _MESSAGE_HPP_
