def pytest_addoption(parser):
    parser.addoption("--epics", action="store", required=True,
                     help="The target pysmurf server's epics prefix")

    parser.addoption("--config", action="store",
                     default="/usr/local/src/pysmurf/cfg_files/stanford/"
                             "experiment_fp30_cc02-03_lbOnlyBay0.cfg",
                     help="Path to the pysmurf configuration file")
