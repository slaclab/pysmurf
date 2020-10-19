#ifndef _SMURF_CORE_TRANSMITTERS_GCPTRANSMITTER_H_
#define _SMURF_CORE_TRANSMITTERS_GCPTRANSMITTER_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data GCP Transmitter
 * ----------------------------------------------------------------------------
 * File          : GCPTransmitter.h
 * Created       : 2020-09-08
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data GCP Transmitter Class.
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

#include <sys/select.h>
#include <string>
#include "smurf/core/transmitters/BaseTransmitter.h"

namespace smurf::core::transmitters {
    // Custom transmitter class
    class GCPTransmitter : public BaseTransmitter
    {
    public:
        // Custom class constructor and destructor
        GCPTransmitter(std::string gcpHost, unsigned gcpPort);
        ~GCPTransmitter();

        // This is the virtual method defined in 'BaseTransmitter' which is call whenever a
        // new SMuRF packet is ready.
        void dataTransmit(SmurfPacketROPtr sp);

        // This is the virtual method defined in 'BaseTransmitter' which is call whenever
        // new metadata is ready.
        void metaTransmit(std::string cfg);

        // Set/Get the debug flags
        void       setDebugData(bool d) { debugData = d;    };
        void       setDebugMeta(bool d) { debugMeta = d;    };
        const bool getDebugData()       { return debugData; };
        const bool getDebugMeta()       { return debugMeta; };

        // Extend BaseTransmitter::setDisable to toggle GCP connection state
        void setDisable(bool d);

        static void setup_python();

    private:
        // Change GCP connection state on setDisable()
        bool connect();
        void disconnect();

        int gcpFd; // socket FD
        fd_set fdset;
        bool debugData; // Debug flag, for data
        bool debugMeta; // Debug flag, for metadata
        std::string gcpHost; // GCP host (usually localhost?)
        unsigned gcpPort; // GCP server port, as determined by slot number
        size_t bufferSize;
        uint32_t *frameBuffer; // Buffers frame to be sent to GCP
    };
}
#endif
