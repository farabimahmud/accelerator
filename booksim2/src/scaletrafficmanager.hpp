#ifndef _SCALE_TRAFFICMANAGER_HPP_
#define _SCALE_TRAFFICMANAGER_HPP_

#include "message.hpp"
#include "trafficmanager.hpp"

class ScaleTrafficManager : public TrafficManager {  

protected:
  int _flit_size;
  bool _watch_all_pkts;
  int _vnets;
  vector<int> _last_vnet;

private:
  vector<vector<list<Message *> > > _input_buffer;
  vector<vector<list<Message *> > > _output_buffer;

protected:
  virtual void _RetireFlit(Flit *f, int dest);

  virtual void _Inject();
  virtual void _Step();

  virtual void _GeneratePacket(int source, int stype, int vnet, int time);

public:
  ScaleTrafficManager(const Configuration &config, const vector<Network *> &net, int vnet);
  ~ScaleTrafficManager();
};


#endif // _SCALETRAFFICMANAGER_HPP_
