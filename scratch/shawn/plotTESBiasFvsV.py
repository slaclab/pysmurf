plt.clf()

bias = raw_data['bias']
for idx in [0,3,5]:
    fres = raw_data[2][(None,None)]['fvsfr'][idx].copy()
    fmax = np.max(fres)
    fmin = np.min(fres)
    fspan = (fmax-fmin)*1000. # to kHZ
    fmean = np.mean(fres)
    plt.plot(
        bias,
        1000.*(fres-fmean), # to kHZ
        label=f'fmean = {fmean:0.1f} MHz, fspan = {fspan:0.1f} kHz')

plt.title('C03 TES bias voltage resonator response - power-on relay setting')
plt.xlabel('TES bias group 1 voltage (V)')
plt.ylabel('Resonator frequency - mean resonator frequency (kHz)')
plt.legend()
