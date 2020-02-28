pArtHz=[55.10, 68.47, 56.40, 54.39, 51.65, 44.61, 49.28, 49.71, 46.61, 46.02, 47.58, 56.03, 53.00, 50.57, 71.94, 71.46, 63.10, 60.39, 53.78, 47.20, 61.95, 65.84, 62.07, 67.29, 65.89, 70.53, 56.89, 80.64, 65.08, 86.36, 57.51, 74.79, 53.48, 58.07, 47.51, 57.40, 49.83, 46.22, 54.39, 52.93, 47.00, 43.60, 56.25, 59.71, 46.87, 62.10, 61.05, 88.47, 104.47, 51.56, 44.62, 44.66, 61.34, 58.59, 64.46, 44.48, 73.18, 63.23, 45.06, 57.73, 76.98, 63.84, 82.90, 128.32, 44.42, 59.51, 69.17, 70.15, 64.35, 85.20, 62.72, 102.93, 63.96, 52.18]

n, bins, patches = plt.hist(pArtHz, 50, facecolor='g', alpha=0.5)

plt.xlabel('Readout eq. NEI (pW/$\sqrt{Hz}$)',fontsize=16)
plt.ylabel('N',fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)

plt.show()


