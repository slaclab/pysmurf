# Take and plot take_debug_data
band=0
ch=0
f, df, sync = S.take_debug_data(band, IQstream = False, 
                                single_channel_readout=0, nsamp=2**19)

fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(f[:, ch]*1e3)
ax[0].set_ylabel('Tracked Freq [kHz]')

bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)
ax[0].text(.95, .9, 'Band {} Ch {:03}'.format(band, ch), fontsize=10,
    transform=ax[0].transAxes, horizontalalignment='right', bbox=bbox)

ax[1].plot(df[:, ch]*1e3)
ax[1].set_ylabel('Freq Error [kHz]')
ax[1].set_xlabel('Samp Num')

sync_idx = S.make_sync_flag(sync)
for s in sync_idx:
    ax[0].axvline(s, color='k', linestyle=':', alpha=.5)
    ax[1].axvline(s, color='k', linestyle=':', alpha=.5)

plt.tight_layout()
