import os
import h5py
import numpy as np

class SinglePeakResult:

	pk_idx   = 0 		## For a given file, each peak found has a unique id
	pk_added = False	## Was the peak location added manually
	f_ctr    = 0.0 		## Identified central frequency of this peak
	mfz_ctr  = 0.0 		## Magnitude of filtered signal at central frequency of this peak

	## Parameter estimates
	f0_est = -1.0
	Qr_est = -1.0
	id_f0  = -1.0
	id_BW  = -1.0

	## Rough fit results
	rough_result = {"f0"    : -1.0, 
	                "Q"     : -1.0,
	                "phi"   : -1.0,
	                "zOff"  : -1.0,
	                "Qc"    : -1.0,
	                "tau1"  : -1.0,
	                "Imtau1": -1.0}

	## Fine fit parameter guesses
	fine_pguess = np.zeros(8)

	## Rough fit results
	fine_result = { "f0"    : -1.0, 
	                "Q"     : -1.0,
	                "phi"   : -1.0,
	                "zOff"  : -1.0,
	                "Qc"    : -1.0,
	                "tau1"  : -1.0,
	                "Imtau1": -1.0}
	fine_errors = { "f0"    : -1.0, 
	                "Q"     : -1.0,
	                "phi"   : -1.0,
	                "zOff"  : -1.0,
	                "Qc"    : -1.0,
	                "tau1"  : -1.0,
	                "Imtau1": -1.0}

	def __init__(self, peak_idx):
		self.pk_idx = peak_idx
		return None

	def show_par_ests(self):
		print("Parameter estimates for peak", self.pk_idx, "(Added? "+str(self.pk_added)+")")
		print("f0_est", self.f0_est)
		print("Qr_est", self.Qr_est)
		print("id_f0" , self.id_f0)
		print("id_BW" , self.id_BW)

	def show_rough_result(self):
		print("Rough Fit result for peak", self.pk_idx)
		for key in self.rough_result.keys():
			print(key , ":" , self.rough_result[key] )

	def show_fine_result(self):
		print("Fine Fit result for peak",self.pk_idx)
		for key in self.fine_result.keys():
			print(key , ":" , self.fine_result[key] , "+/-" , self.fine_errors[key] )

class SingleFileResult:

	## Class attributes
	in_fname = "Psweep_P-00.0_20220101_000000.h5"

	power   = 0.0					## [FLOAT] (dBM) RF stimulus power
	n_pks   = 1						## [INT] How many peaks were found in this file

	start_T = np.array([])			## [array of FLOAT] (mK) Temperature at start of the f sweep
	final_T = np.array([])			## [array of FLOAT] (mK) Temperature at end of the f sweep

	peak_fits = np.zeros(1, dtype=object)

	def __init__(self, file_name):
		self.in_fname   = file_name.split('/')[-1]
		return None

	def resize_peak_fits(self, n_peaks):
		self.n_pks = n_peaks
		self.peak_fits = np.zeros(self.n_pks, dtype=object)

	def show_metadata(self):
		print("In file:     ", self.in_fname)
		print("Power [dBm]: ", self.power)
		print("# of peaks:  ", self.n_pks)
		print("Start T [mK]:", self.start_T)
		print("Final T [mK]:", self.final_T)

	def show_fit_results(self):
		for i in range(self.n_pks):
			self.peak_fits[i].show_fine_result()

class SeriesFitResult:

	## Class attributes
	date    = "20220101"			## [STRING] YYYYMMDD
	series  = "20220101_000000"		## [STRING] YYYYMMDD_HHMMSS

	n_files = 1						## [INT] How many files are in this series

	powers  = np.array([])			## [array of FLOAT] (dBm) RF power for this sweep
	fit_fr  = np.array([])			## [array of FLOAT] (Hz) Central frequency 
	fit_Qr  = np.array([])			## [array of FLOAT] Quality factor
	fit_Qi  = np.array([])			## [array of FLOAT] Quality factor
	fit_Qc  = np.array([])			## [array of FLOAT] Quality factor

	file_fits = np.zeros(1, dtype=object)

	def __init__(self, date, series):
		self.date   = date
		self.series = series
		return None

	def resize_file_fits(self, n_files):
		self.n_files   = n_files
		self.file_fits = np.zeros(self.n_files, dtype=object)
		self.powers    = np.zeros(self.n_files)
		self.fit_fr    = np.zeros(self.n_files)
		self.fit_Qr    = np.zeros(self.n_files)
		self.fit_Qi    = np.zeros(self.n_files)
		self.fit_Qc    = np.zeros(self.n_files)

	def show_series_result(self):
		for i in range(self.n_files):
			self.file_fits[i].show_metadata()
			self.file_fits[i].show_fit_results()

	def save_to_file(self, file_path, file_name=None):

		## Locate, name, and open file
		fname = file_name if file_name is not None else ("ResonanceFits_" + self.series)
		f = h5py.File(os.path.join(file_path,fname+'.h5'),'w')

		## Save the SeriesFit metadata to top level
		f.create_dataset("date"   , data=np.array([self.date]   , dtype='S'))
		f.create_dataset("series" , data=np.array([self.series] , dtype='S'))
		f.create_dataset("nfiles" , data=np.array([self.n_files], dtype='int'))
		f.create_dataset("powers" , data=self.powers)
		f.create_dataset("fit_fr" , data=self.fit_fr)
		f.create_dataset("fit_Qr" , data=self.fit_Qr)
		f.create_dataset("fit_Qi" , data=self.fit_Qi)
		f.create_dataset("fit_Qc" , data=self.fit_Qc)

		## Loop over each single file fit
		grp_array = np.zeros(self.n_files, dtype=object)
		for i in np.arange(self.n_files):
			## Create a group for each single file result
			grp_array[i] = f.create_group(self.file_fits[i].in_fname)#.split(".")[0].split("_")[-1])
			grp_array[i].create_dataset("in_fname", 
				data=np.array([self.file_fits[i].in_fname], dtype='S'))
			grp_array[i].create_dataset("power", 
				data=np.array([self.file_fits[i].power]))
			grp_array[i].create_dataset("n_pks", 
				data=np.array([self.file_fits[i].n_pks]))
			grp_array[i].create_dataset("start_T", 
				data=self.file_fits[i].start_T)
			grp_array[i].create_dataset("final_T", 
				data=self.file_fits[i].final_T)

			## Loop over each single resonance peak fit
			sub_array = np.zeros(self.file_fits[i].n_pks, dtype=object)
			for j in np.arange(self.file_fits[i].n_pks):
				sub_array[j] = grp_array[i].create_group("peak"+str(j))
				sub_array[j].create_dataset("pk_idx", 
					data=np.array([self.file_fits[i].peak_fits[j].pk_idx], dtype='int'))
				sub_array[j].create_dataset("pk_added", 
					data=np.array([self.file_fits[i].peak_fits[j].pk_added], dtype='bool'))
				sub_array[j].create_dataset("f_ctr", 
					data=np.array([self.file_fits[i].peak_fits[j].f_ctr]))
				sub_array[j].create_dataset("mfz_ctr", 
					data=np.array([self.file_fits[i].peak_fits[j].mfz_ctr]))

				sub_array[j].create_dataset("f0_est", 
					data=np.array([self.file_fits[i].peak_fits[j].f0_est]))
				sub_array[j].create_dataset("Qr_est", 
					data=np.array([self.file_fits[i].peak_fits[j].Qr_est]))
				sub_array[j].create_dataset("id_f0", 
					data=np.array([self.file_fits[i].peak_fits[j].id_f0]))
				sub_array[j].create_dataset("id_BW", 
					data=np.array([self.file_fits[i].peak_fits[j].id_BW]))

				sub_array[j].create_dataset("fine_pguess", 
					data=self.file_fits[i].peak_fits[j].fine_pguess)

				sub_array[j].create_dataset("fitval_keys",
					data=np.array([k for k in self.file_fits[i].peak_fits[j].rough_result.keys()],
					dtype='S')
					)
				sub_array[j].create_dataset("rough_result",
					data=[v for (k,v) in self.file_fits[i].peak_fits[j].rough_result.items()])
				sub_array[j].create_dataset("fine_result",
					data=[v for (k,v) in self.file_fits[i].peak_fits[j].fine_result.items()])
				sub_array[j].create_dataset("fine_errors",
					data=[v for (k,v) in self.file_fits[i].peak_fits[j].fine_errors.items()])

		f.close()

		return os.path.join(file_path,fname+'.h5')

def decode_resfit_hdf5(filename):

	with h5py.File(filename, "r") as f:
		_date  = f["date"][0].decode('UTF-8')
		_sers  = f["series"][0].decode('UTF-8')
		fitres = SeriesFitResult(_date,_sers)

		fitres.n_files   = f["nfiles"][0]
		fitres.powers    = np.array(list(f["powers"]))
		fitres.fit_fr    = np.array(list(f["fit_fr"]))
		fitres.fit_Qr    = np.array(list(f["fit_Qr"]))
		fitres.fit_Qi    = np.array(list(f["fit_Qi"]))
		fitres.fit_Qc    = np.array(list(f["fit_Qc"]))

		## Get a list of groups
		grp_keys = list(f.keys())
		for x in set(grp_keys).intersection(["date","series","nfiles","fit_fr","fit_Qr","fit_Qi","fit_Qc"]):
			grp_keys.remove(x)

		# print(grp_keys)

		fitres.file_fits = np.zeros(fitres.n_files, dtype=object)
		for i in np.arange(fitres.n_files):
			fitres.file_fits[i] = SingleFileResult(grp_keys[i])

			fitres.file_fits[i].power   = f[grp_keys[i]]['power'][0]
			fitres.file_fits[i].n_pks   = f[grp_keys[i]]['n_pks'][0]
			fitres.file_fits[i].start_T = np.array([f[grp_keys[i]]['start_T'][0]])
			fitres.file_fits[i].final_T = np.array([f[grp_keys[i]]['final_T'][0]])

			## Get a list of subgroups
			subgrp_keys = list(f[grp_keys[i]].keys())
			for x in set(subgrp_keys).intersection(["in_fname","power","n_pks","start_T","final_T"]):
				subgrp_keys.remove(x)

			fitres.file_fits[i].peak_fits = np.zeros(fitres.file_fits[i].n_pks, dtype=object)
			for j in np.arange(fitres.file_fits[i].n_pks):
				fitres.file_fits[i].peak_fits[j] = SinglePeakResult(f[grp_keys[i]][subgrp_keys[j]]["pk_idx"][0])

				# fitres.file_fits[i].peak_fits[j].pk_idx   = f[grp_keys[i]][subgrp_keys[j]]["pk_idx"][0]
				fitres.file_fits[i].peak_fits[j].pk_added = f[grp_keys[i]][subgrp_keys[j]]["pk_added"][0]
				fitres.file_fits[i].peak_fits[j].f_ctr    = f[grp_keys[i]][subgrp_keys[j]]["f_ctr"][0]
				fitres.file_fits[i].peak_fits[j].mfz_ctr  = f[grp_keys[i]][subgrp_keys[j]]["mfz_ctr"][0]

				fitres.file_fits[i].peak_fits[j].f0_est   = f[grp_keys[i]][subgrp_keys[j]]["f0_est"][0]
				fitres.file_fits[i].peak_fits[j].Qr_est   = f[grp_keys[i]][subgrp_keys[j]]["Qr_est"][0]
				fitres.file_fits[i].peak_fits[j].id_f0    = f[grp_keys[i]][subgrp_keys[j]]["id_f0"][0]
				fitres.file_fits[i].peak_fits[j].id_BW    = f[grp_keys[i]][subgrp_keys[j]]["id_BW"][0]

				fitres.file_fits[i].peak_fits[j].fine_pguess = np.array(f[grp_keys[i]][subgrp_keys[j]]["fine_pguess"])

				dict_keys = list(f[grp_keys[i]][subgrp_keys[j]]["fitval_keys"])
				ruff_vals = np.array(f[grp_keys[i]][subgrp_keys[j]]["rough_result"])
				fine_vals = np.array(f[grp_keys[i]][subgrp_keys[j]]["fine_result"])
				fine_errs = np.array(f[grp_keys[i]][subgrp_keys[j]]["fine_errors"])

				fitres.file_fits[i].peak_fits[j].rough_result = {dict_keys[i]:ruff_vals[i] for i in range(len(dict_keys))}
				fitres.file_fits[i].peak_fits[j].fine_result  = {dict_keys[i]:fine_vals[i] for i in range(len(dict_keys))}
				fitres.file_fits[i].peak_fits[j].fine_errors  = {dict_keys[i]:fine_errs[i] for i in range(len(dict_keys))}

		return fitres