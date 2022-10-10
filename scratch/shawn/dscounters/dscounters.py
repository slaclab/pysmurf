# Written by Matthew Hasselfield to compute DS counter bitmasks for
# Simons Observatory
class DownsampleCounters:
    """Model counters to be used for downsampling.  Each counter has a
    period corresponding to a power (k) of a prime number (p).  The
    configuration is set by specifying list of primes and the range of
    values for k.  For example::

       DownsampleCounters([
           (2, (1, 4)),
           (3, (1, 2)),
       ])

    would configure counters with periods [2**1, 2**2, 2**3, 2**4,
    3**1, 3**2]; i.e. [2, 4, 8, 16, 3, 9].  The ordering corresponds
    to sorting by p and then by k; this should correspond in binary
    bitmasks to ordering from least- to most-significant bit.

    """
    def __init__(self, config):
        self.config = list(config)  # take a copy

    def __len__(self):
        return sum([(hi - lo + 1) for _, (lo, hi) in self.config])

    def get_periods(self):
        """Return the periods of the counters.  These are returned from LSB to
        MSB."""
        periods = []
        for p, (lo, hi) in self.config:
            for k in range(lo, hi+1):
                periods.append(int(p**k))
        return periods

    def get_mask(self, n, format=int):
        """Return the mask for downsampling factor n.  By default this is
        returned as an integer, with the bits to watch marked as 1.
        If format=str, then the binary rendering of this number is
        returned as a string.  If format=list then a list of 0s and 1s
        is returned *from LSB to MSB*.

        If the argument cannot be expressed by our factors, None is
        returned.

        """
        mask = []
        #periods = [p**pwr for 
        for p, (lo, hi) in self.config:
            powers = list(range(hi, lo - 1, -1))
            while len(powers):
                f = p ** powers.pop(0)
                if n % f == 0:
                    mask.extend([1] + [0] * len(powers))
                    n //= f
                    break
                mask.append(0)
            # reshuffle these powers so bit mask order matches config
            # order
            mask[-len(range(lo,hi+1)):]=mask[-len(range(lo,hi+1)):][::-1]

        if n != 1:
            return None
        if format is list:
            return mask
        # Convert to string, MSB..LSB
        mask = ''.join(str(i) for i in mask[::-1])
        if format is str:
            return 'b' + mask
        if format is int:
            return int(mask, 2)
        raise ValueError('format?')

    def get_nearby(self, n, desperation=2):
        """Find a number near n that can be expressed by this period set."""
        if self.get_mask(n):
            return n
        n = round(n)
        if self.get_mask(n):
            return n
        diff = 1
        while diff < n * desperation:
            if self.get_mask(n + diff, format=list):
                return n + diff
            elif n - diff > 0 and self.get_mask(n - diff, format=list):
                return n - diff
            diff += 1
        return None

    def period_from_mask(self, mask):
        """Compute the period associated with the mask.  The mask can be str,
        int, or list; with the same conventions as get_mask."""
        if isinstance(mask, str):
            if mask[0] == 'b':
                mask = mask[1:]
            mask = int(mask, 2)
        if isinstance(mask, int):
            mask = [(mask >> i) & 1 for i in range(len(self))]
        else:
            mask = list(mask)  # copy
        n = 1
        for p, (lo, hi) in self.config:
            powers = list(range(hi, lo - 1, -1))
            while len(powers) and len(mask):
                f = p ** powers.pop(0)
                if mask.pop(0):
                    n *= f
        return n

# Some trial configs -- v1 is better.
configs = {
    'v0': [
        (2, (1, 10)),
        (3, (1, 6)),
        (5, (1, 5)),
        (7, (1, 3)),
        (11, (1, 2)),
        (13, (1, 2)),
        (17, (1, 2)),
        (19, (1, 2)),
        (23, (1, 2)),
        (29, (1, 2)),
        (31, (1, 2)),
        (37, (1, 1)),
        (41, (1, 1)),
    ],
    'v1': [
        (2, (1, 8)),
        (3, (1, 5)),
        (5, (1, 4)),
        (7, (1, 2)),
        (11, (1, 1)),
        (13, (1, 1)),
        (17, (1, 1)),
        (19, (1, 1)),
        (23, (1, 1)),
        (29, (1, 1)),
        (31, (1, 1)),
        (37, (1, 1)),
        (41, (1, 1)),
        (43, (1, 1)),
        (47, (1, 1)),
        (53, (1, 1)),
        (59, (1, 1)),
        (61, (1, 1)),
        (67, (1, 1)),
        (71, (1, 1)),
        (73, (1, 1)),
        (79, (1, 1)),
        (83, (1, 1)),
        (89, (1, 1)),
        (97, (1, 1)),
    ],
    'v2': [
        (2, (1, 8)),
        (3, (1, 5)),
        (5, (1, 4)),
        (7, (1, 1)),
    ],
    'v3': [
        (2, (1, 8)),
        (3, (1, 5)),
        (5, (1, 3)),
    ],
}

def plot_all_configs():
    # Test out that class a little and test performance of the two
    # configs.
    import pylab as pl
    import numpy as np
    for name, config in configs.items():
        print(f'Testing {name}')
        print(f'------------------------')
        ds = DownsampleCounters(config)
        periods = ds.get_periods()
        print(f'There are {len(periods)} counter periods; LSB to MSB:\n  {periods}')
        print()
        print('Test cases:')
        for n in [1024, 1000, 200, 20, 65536, 1023]:
            mask = ds.get_mask(n, str)
            print('%6i : %s' % (n, mask))
            if mask is not None:
                assert(ds.period_from_mask(mask) == n)
        print()

        # Measure performance for readout freqs from 1 Hz to f_ramp/2
        f_ramp = 4000

        n_test = np.arange(2, int(f_ramp) + 1)
        n_achieved = np.array([ds.get_nearby(n) for n in n_test])

        f_test = f_ramp / n_test
        f_achieved = f_ramp / n_achieved

        error = (f_achieved / f_test - 1)

        print('plotting')
        pl.semilogx(f_test, error * 100,
                    ls='none', marker='o', markersize=1, alpha=.4)
        pl.title('Downsampling of %.1f kHz ramps' % (f_ramp/1e3))
        pl.xlabel('Target readout frequency (Hz)')
        pl.ylabel('Readout freq error (%)')
        pl.savefig(f'{name}.png')
        pl.clf()
        
        print('\n')

