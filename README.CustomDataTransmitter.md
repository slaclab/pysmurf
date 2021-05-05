# Data Custom Transmitter

The SMuRF processor pipeline has a placeholder to insert a user-defined processing block which will take the processed data packet and send them to a third party system.

A base C++ class called [BaseTransmitter](include/smurf/core/transmitters/BaseTransmitter.h) is provided. The user code should write a derivate class from this base class.

The base class contains two virtual methods:
 - `dataTransmit`: This method is called when a new processed data packet is available. An object of type `SmurfPacketROPtr` is passed as an argument to this method; this object is an (smart) pointer to a [SmurfPacket class](include/smurf/core/common/SmurfPacket.h) object, which will give RO access to the content of the SMuRF packet (see [here](README.SmurfPacket.md) for details).
 - `metaTransmit`: This method is called when a new frame with metadata is available. The metadata is passed as a `std::string` object to this method.

An example on how to write a custom data transmitter and use it with the pysmurf server is available in the [pysmurf-custom-transmitter-example](https://github.com/slaclab/pysmurf-custom-transmitter-example) git repository.
