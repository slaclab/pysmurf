#Runlike this exec(open("scratch/shawn/tune_fp30_teses.py").read())


# resonators with significant cross talk
#tes_res=[5132.81, 5190.0, 5076.23, 5257.31, 5124.64, 5287.62, 5144.13, 5069.59, 5186.70, 5255.02, 5199.44]

tes_res=[5248.28,5286.32,5183.00,5084.63,5240.82,5250.13,5134.63,5191.79,5184.64,5085.97,5242.33,5251.41,5136.33,5193.10,5077.97,5087.33,5253.08,5180.06,5137.81,5195.12,5079.63,5071.13,5128.05,5245.19,5196.57,5081.36,5073.07,5224.99,5129.74,5246.62,5256.35,5139.83,5198.01,5074.63,5131.15,5141.47]
S.freq_resp=S.fake_resonance_dict(tes_res)

S.setup_notches(2,new_master_assignment=True)
S.plot_tune_summary(2,eta_scan=True)
S.run_serial_gradient_descent(2)
S.run_serial_eta_scan(2)

