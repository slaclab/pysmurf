/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data GCP Transmitter
 * ----------------------------------------------------------------------------
 * File          : GCPTransmitter.cpp
 * Created       : 2020-09-08
 *-----------------------------------------------------------------------------
 * Description :
 *   SMuRF Data GCP Transmitter Class.
 *-----------------------------------------------------------------------------
 * This file is part of the smurf software platform. It is subject to
 * the license terms in the LICENSE.txt file found in the top-level directory
 * of this distribution and at:
    * https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of the smurf software platform, including this file, may be
 * copied, modified, propagated, or distributed except according to the terms
 * contained in the LICENSE.txt file.
 *-----------------------------------------------------------------------------
**/
#include <sys/socket.h>
#include <netdb.h>
#include <iostream>
#include <utility>
#include <boost/python.hpp>
#include "smurf/core/transmitters/GCPTransmitter.h"
#include "smurf/core/processors/SmurfProcessor.h"

namespace bp = boost::python;
namespace sct = smurf::core::transmitters;

sct::GCPTransmitter::GCPTransmitter(std::string gcpHost, unsigned gcpPort) :
    sct::BaseTransmitter(), debugData(false), debugMeta(false),
    gcpHost(std::move(gcpHost)), gcpPort(gcpPort)
{
    gcpFd = -1;
}

sct::GCPTransmitter::~GCPTransmitter()
{
    disconnect();
}

void sct::GCPTransmitter::disconnect()
{
    if (!gcpFd)
        return;
    shutdown(gcpFd, SHUT_RDWR);
    close(gcpFd);
    gcpFd = -1;
}

bool sct::GCPTransmitter::connect()
{
    int sockfd;
    struct sockaddr_in gcp_addr;
    struct hostent* host;

    if (gcpFd > 0)
        disconnect();

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("Failed to create socket.");
        return false;
    }

    memset(&gcp_addr, 0, sizeof(gcp_addr));
    gcp_addr.sin_family = AF_INET;
    gcp_addr.sin_port = htons(gcpPort);
    host = gethostbyname(gcpHost.c_str());
    if (!host) {
        herror("Invalid hostname");
        return false;
    }
    memcpy((char *) &gcp_addr.sin_addr.s_addr, (char *) host->h_addr,
          host->h_length);

    if (::connect(sockfd, (struct sockaddr *)&gcp_addr, sizeof(gcp_addr)) != 0) {
        perror("Failed to connect to GCP");
        return false;
    }
    gcpFd = sockfd;
    return true;
}

void sct::GCPTransmitter::setDisable(bool d)
{
    bool connected = false;
    if (d)
        disconnect();
    else
        connected = connect();

    // If we did not connect successfully, we set disable to true regardless.
    sct::BaseTransmitter::setDisable(d || !connected);
}

void sct::GCPTransmitter::dataTransmit(SmurfPacketROPtr sp)
{
    if (!gcpFd)
        return;

    // Number of channels can change, and we can't access max number of channels.
    // So, just recompute buffer size from header info, and reallocate buffer as
    // needed.
    std::size_t numCh {sp->getHeader()->getNumberChannels()};
    std::size_t bufferSize = 3; // sync_num, numCh, checksum
    bufferSize += TesBiasArray<int>::TesBiasCount;
    bufferSize += numCh;
    if (frameBuffer.size() != bufferSize) {
        frameBuffer.resize(bufferSize);
    }

    // If the debug flag is enabled, print part of the SMuRF Packet
    if (debugData)
    {

        std::cout << "=====================================" << std::endl;
        std::cout << "Packet received" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << std::endl;

        std::cout << "-----------------------" << std::endl;
        std::cout << " HEADER:" << std::endl;
        std::cout << "-----------------------" << std::endl;
        std::cout << "Version            = " << unsigned(sp->getHeader()->getVersion()) << std::endl;
        std::cout << "Crate ID           = " << unsigned(sp->getHeader()->getCrateID()) << std::endl;
        std::cout << "Slot number        = " << unsigned(sp->getHeader()->getSlotNumber()) << std::endl;
        std::cout << "Number of channels = " << unsigned(numCh) << std::endl;
        std::cout << "Unix time          = " << unsigned(sp->getHeader()->getUnixTime()) << std::endl;
        std::cout << "Frame counter      = " << unsigned(sp->getHeader()->getFrameCounter()) << std::endl;
        std::cout << "TES Bias values:" << std::endl;
        for (std::size_t i{0}; i < 16; ++i)
            std::cout << sp->getHeader()->getTESBias(i) << ", ";
        std::cout << std::endl;

        std::cout << std::endl;

        std::cout << "-----------------------" << std::endl;
        std::cout << " DATA (up to the first 20 points):" << std::endl;
        std::cout << "-----------------------" << std::endl;

    std::size_t n{20};
    if (numCh < n)
        n = numCh;

    for (std::size_t i(0); i < n; ++i)
            std::cout << "Data[" << i << "] = " << sp->getData(i) << std::endl;

        std::cout << "-----------------------" << std::endl;
        std::cout << std::endl;

        std::cout << "=====================================" << std::endl;
    }

    // Compute sync number
    // TODO: move these statics somewhere better
    static uint32_t sampleRate = 200;
    static uint32_t maxRecord = 86400*sampleRate;
    static uint64_t nsecSamplePeriod = 1E9/sampleRate;
    uint32_t syncNum = (uint32_t)
        ((sp->getHeader()->getUnixTime()/nsecSamplePeriod) % maxRecord);

    std::size_t ofs = 0;

    /* Form frame header */
    frameBuffer[ofs++] = syncNum;
    // Template parameter <int> doesn't make sense, but is also meaningless
    std::size_t nBiases = TesBiasArray<int>::TesBiasCount;
    for (std::size_t i(0); i < nBiases; ++i) {
        frameBuffer[ofs++] = sp->getHeader()->getTESBias(i);
    }

    /* Insert data */
    // TODO: Should make this a single copy operation.
    for (std::size_t i(0); i < numCh; ++i)
        frameBuffer[ofs++] = sp->getData(i);

    /* Checksum */
    uint32_t checksum = 0;

    // checksum is all fields XOR'd together, for symmetry with MAS
    for (std::size_t i(0); i < bufferSize-1; ++i)
        checksum ^= frameBuffer[i];
    frameBuffer[bufferSize-1] = checksum;

    /* Send */
    // select, write. If time out, call disconnect()
    FD_ZERO(&fdset);
    FD_SET(gcpFd, &fdset);
    // Wait <~ 1 sampling period
    struct timeval timeout = {0, 1000};
    int nret = select(gcpFd+1, nullptr, &fdset,  nullptr, &timeout);
    if (nret < 0) {
        perror("select() failed on GCP data socket.");
        // For now, we will be agressive and disconnect on timeout
        disconnect();
        return;
    } else if (nret == 0) {
        // Timed out. Also disconnect.
        std::cerr << "GCP send timed out" << std::endl;
        disconnect();
        return;
    }
    send(gcpFd, &frameBuffer[0], bufferSize, 0);
};

void sct::GCPTransmitter::metaTransmit(std::string cfg)
{
    // If the debug flag is enabled, print the metadata
    if (debugMeta)
    {
        std::cout << "=====================================" << std::endl;
        std::cout << "Metadata received" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << cfg << std::endl;
        std::cout << "=====================================" << std::endl;
    }
}

void sct::GCPTransmitter::setup_python()
{
    bp::class_< sct::GCPTransmitter,
                std::shared_ptr<sct::GCPTransmitter>,
                boost::noncopyable >
                ("GCPTransmitter", bp::init<std::string, unsigned>())
        .def("setDisable",     &GCPTransmitter::setDisable)
        .def("getDisable",     &GCPTransmitter::getDisable)
        .def("setDebugData",   &GCPTransmitter::setDebugData)
        .def("getDebugData",   &GCPTransmitter::getDebugData)
        .def("setDebugMeta",   &GCPTransmitter::setDebugMeta)
        .def("getDebugMeta",   &GCPTransmitter::getDebugMeta)
        .def("clearCnt",       &GCPTransmitter::clearCnt)
        .def("getDataDropCnt", &GCPTransmitter::getDataDropCnt)
        .def("getMetaDropCnt", &GCPTransmitter::getMetaDropCnt)
        .def("getDataChannel", &GCPTransmitter::getDataChannel)
        .def("getMetaChannel", &GCPTransmitter::getMetaChannel)
    ;
};
