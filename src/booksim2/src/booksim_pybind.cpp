#ifdef LIB_BOOKSIM

#include <pybind11/pybind11.h>

#include "booksim_wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pybooksim, m) {
  m.doc() = "pybind11 booksim plugin";

  py::class_<BookSim>(m, "BookSim")
    .def(py::init<const std::string&>())
    .def("IssueMessage", &BookSim::IssueMessage)
    .def("PeekMessage", &BookSim::PeekMessage)
    .def("SetSimTime", &BookSim::SetSimTime)
    .def("GetSimTime", &BookSim::GetSimTime)
    .def("Idle", &BookSim::Idle)
    .def("WakeUp", &BookSim::WakeUp);

  py::class_<Message, std::unique_ptr<Message, py::nodelete>> message(m, "Message");

  py::enum_<Message::MessageType>(message, "MessageType")
    .value("ReadRequest", Message::MessageType::ReadRequest)
    .value("ReadReply", Message::MessageType::ReadReply)
    .value("WriteRequest", Message::MessageType::WriteRequest)
    .value("WriteReply", Message::MessageType::WriteReply)
    .value("ControlRequest", Message::MessageType::ControlRequest)
    .value("ControlReply", Message::MessageType::ControlReply)
    .value("MessageType_NUM", Message::MessageType::MessageType_NUM)
    .export_values();
};

#endif // #ifdef LIB_BOOKSIM
