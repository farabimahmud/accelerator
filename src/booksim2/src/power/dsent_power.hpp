#ifndef _DSENT_POWER_HPP_
#define _DSENT_POWER_HPP_

#include "network.hpp"
#include "config_utils.hpp"
#include "flitchannel.hpp"
#include "DSENT.h"

class DSENTPower {

protected:
  // network under simulation
  Network * _net;
  int _classes;
  int _frequency;  // in MHz
  double _total_time;
  string _router_cfg;
  string _link_cfg;
  // write result to a tabbed format to file
  string _output_file_name;

  // DSENT instances
  DSENT::DSENT *_router_instance;
  DSENT::DSENT *_link_instance;

  // router dynamic power
  double _router_dyn_power;
  double _buf_wr_dyn_power;
  double _buf_rd_dyn_power;
  double _sw_arb_local_dyn_power;
  double _sw_arb_global_dyn_power;
  double _xbar_dyn_power;
  double _clk_dyn_power;

  // router static power
  double _router_leak_power;
  double _buf_leak_power;
  double _sw_arb_leak_power;
  double _xbar_leak_power;
  double _clk_leak_power;

  // link power
  double _link_dyn_power;
  double _link_leak_power;

  // network power
  double _net_dyn_power;
  double _net_leak_power;
  double _net_power;

  // methods
  void resetPowerStats();
  void calcRouterPower();
  void calcLinkPower();

public:
  DSENTPower(Network * net, const Configuration &config);
  ~DSENTPower();

  void Run();
  double GetNetworkPower() const { return _net_power; }
  double GetNetDynPower() const { return _net_dyn_power; }
  double GetNetLeakPower() const { return _net_leak_power; }
  double GetNetDynPowerWoClk() const { return _net_dyn_power - _clk_dyn_power; }

};

#endif
