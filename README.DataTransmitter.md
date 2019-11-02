# Data Transmitter

The SMuRF processor pipeline has a placeholder to insert a user-defined processing block which will take the processed data packet and send them to a third party system.

A base C++ class called [BaseTransmitter](include/smurf/core/transmitters/BaseTransmitter.h) is provided. The user code should write a derivate class from this base class.

The base class contains a virtual method called `transmit`. This method is called when a new processed packet is available. An object of type `SmurfPacketROPtr` is passed as an argument to this method; this object is an (smart) pointer to a [SmurfPacket class](include/smurf/core/common/SmurfPacket.h) object, which will give RO access to the content of the SMuRF packet (see [here](README.SmurfPacket.md) for details).
