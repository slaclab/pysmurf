import numpy
import matplotlib.pyplot as plt



def fluxmod(DF, SYNC):
    mkr_ratio = 512  # ratio of rates on marker to rates in data
    num_channels = len(DF[1,:])
  
    # setup various signal cut thresholds. 
    threshold = 0.4 # threshold for detecting a peak in the autocorelation
    minscale = 0.02; #peak to peak flux ramp
    min_spectrum = 0.9 # low pass power / high pass power limit. Used to eliminate oscillating signals

    result = num_channels * [0]  # holds the number of phi0 / flux ramp
    peak_to_peak = num_channels*[0]
    lp_power = num_channels*[0]

    for channel in range(0, num_channels):
        ch = channel 
        #A = R[0]
        B = DF
        C = SYNC

        #a = numpy.shape(A)
        b = numpy.shape(B)
        c = numpy.shape(C)
        Bx = B[:,ch]

        # Applying cuts to not process bad resonators
        peak_to_peak[channel] = max(Bx) - min(Bx)
        if peak_to_peak[channel] > 0:
            lp_power[channel] = numpy.std(B[:,channel]) / numpy.std(numpy.diff(B[:,channel]))
        if ((peak_to_peak[channel]  > minscale) and (lp_power[channel] > min_spectrum)):  # just give up

            # dumb way to average flux ramp cycles
            Cx = C[:,0] # marker for flux ramp. 
            n2 = len(Cx) # lengh of flux ramp
            mkr1 = 0
            mkr2 = 0
            mkrgap = 0
            totmkr = 0
            lastmkr = 0
            for n in range (1,n2):  # loop over flux ramp marker
                mkrgap = mkrgap + 1
                if ((Cx[n] >0) and (mkrgap > 100 )):
                    mkrgap =  0
                    totmkr = totmkr+ 1
                    lastmkr = n # last marker 
                    if mkr1 == 0:
                        mkr1 = n
                    elif mkr2 == 0:
                        mkr2 = n
            dn = round((mkr2 - mkr1) /  mkr_ratio)
            sn = round(mkr1 / mkr_ratio)


            flux = dn * [0] # array of zeros
            for mkr in range(0,totmkr-1):  # loop over markers
                for n in range(0, dn):
                    flux[n] = flux[n] +  Bx[sn + mkr * dn + n]
            flux = flux - numpy.mean(flux)

            # end of dumb flux ramp average code

            sxarray = [0]
            pts = len(flux)
            for rlen in range(1,round(pts/2)):
                refsig = flux[0:rlen]
                sx = 0
                for pt in range(0,pts):
                    pr = pt % rlen
                    sx = sx + refsig[pr] * flux[pt]
                sxarray.append(sx)

            ac  = 0 
            for n in range (0, pts):
                ac = ac + flux[n] * flux[n]

            #print("autocorrelation = ", ac);

            scaled_array = sxarray / ac

            pk = 0
            #print("scale len = ", len(scaled_array), "pts = ", pts)
            for n in range(0, round(pts/2)):
                if scaled_array[n] > threshold:
                  if scaled_array[n] > scaled_array[pk]:
                      pk = n
                  else:
                      break;
        #polyfit
            Xf = [-1, 0, 1]
            Yf = [scaled_array[pk-1], scaled_array[pk], scaled_array[pk+1]]
            V = numpy.polyfit(Xf, Yf, 2)
            offset = -V[1]/(2 * V[0]); 
            peak = offset + pk
            #print("polyfit = ", V, 'offset =',offset, "peak = ", peak )
            result[channel] = dn /  peak
            #plt.plot(scaled_array)
            #plt.show()

            #plt.plot(flux)
            #plt.show()
            print("ch = ", channel, "result = ", result[channel]);


            if 0:   # plotting routine to show sin fit
                rs = 0
                rc = 0
                r = pts * [0]
                s = pts * [0]
                c = pts * [0]
                scl = numpy.max(flux) - numpy.min(flux)
                for n in range(0, pts):
                    s[n] = numpy.sin(n * 2 * numpy.pi / (dn/result[channel]));
                    c[n] = numpy.cos(n * 2 * numpy.pi / (dn/result[channel]));
                    rs = rs + s[n] * flux[n]
                    rc = rc + c[n] * flux[n]

                theta = numpy.arctan2(rc, rs)
                for n in range(0, pts):
                    r[n] = 0.5 * scl *  numpy.sin(theta + n * 2 * 3.14159 / (dn/result[channel]));

                plt.plot(r)
                plt.plot(flux)
                plt.show()

    fmod_array = []            
    for n in range(0, 512):
        if(result[n] > 0):
            fmod_array.append(result[n])
    mod_median = numpy.median(fmod_array)
    return(mod_median)
    
    
R= numpy.load("fluxRampCheck.npy")


DF = R[1]
SYNC = R[2]

output = fluxmod(DF, SYNC)
print("median mod frequency = ", output);
