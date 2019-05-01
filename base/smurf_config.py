import json
import io
'''
Stolen from Cyndia/Shawns get_config.py
read or dump a config file
not yet tested
do we want to hardcode key names?
'''

class SmurfConfig:
    """Initialize, read, or dump a SMuRF config file.
       Will be updated to not require a json, which is unfortunate

    """

    def __init__(self, filename=None):
        self.filename = filename
        # self.config = [] # do I need to initialize this? I don't think so
        if self.filename is not None:
            self.read(update=True)

    def read(self, update=False):
        """Reads config file and updates the configuration.

           Args:
              update (bool): Whether or not to update the configuration.
        """
        if update:
            comment_char='#'
            with open(self.filename) as config_file:
                no_comments=[]
                for idx, line in enumerate(config_file):
                    if line.lstrip().startswith(comment_char):
                        # line starts with comment character - remove it
                        continue
                    elif comment_char in line:
                        # there's a comment character on this line.
                        # ignore it and everything that follows
                        line=line.split('#')[0]
                        no_comments.append(line)
                    else:
                        # will pass on to json parser
                        no_comments.append(line)
                        
                loaded_config = json.loads('\n'.join(no_comments))
            
            # put in some logic here to make sure parameters in experiment file match 
            # the parameters we're looking for
                self.config = loaded_config

    def update(self, key, val):
        """Updates a single key in the config

           Args:
              key (any): key to update in the config dictionary
              val (any): value to assign to the given key
        """
        self.config[key] = val

    def write(self, outputfile):
        """Dumps the current config to a file

           Args:
              outputfile (str): The name of the file to save the configuration to.
        """
        ## dump current config to outputfile ##
        with io.open(outputfile, 'w', encoding='utf8') as out_file:
            str_ = json.dumps(self.config, indent = 4, separators = (',', ': '))
            out_file.write(str_)

    def has(self, key):
        """Reports if configuration has requested key.

           Args:
              key (any): key to check for in configuration dictionary.
        """
        
        if key in self.config:
            return True
        else:
            return False

    def get(self, key):
        """Returns configuration entry for requested key.  Returns
           None if key not present in configuration.

           Args:
              key (any): key whose configuration entry to retrieve.
        """
        
        if self.has(key):
            return self.config[key]
        else:
            return None

    def get_subkey(self, key, subkey):
        """
        Get the subkey value. A dumb thing that just formats strings for you.
        Will return None if it can't find stuff

        Args:
          key (any): key in config
          subkey (any): config subkey
        """

        if self.has(key):
            sub_dict = self.config[key]
            try:
                return sub_dict[subkey]
            except KeyError:
                print("Key found, but subkey not found")
                return None
        else:
            print("Key not found")
            return None

    def update_subkey(self, key, subkey, val):
        """
        More dumb wrappers for nested dictionaries.

        Args:
          key (any): key in config
          subkey (any): config subkey
          val (any): value to write
        """

        try: 
            self.config[key][subkey] = val
        except TypeError:
            self.config[key] = {} # initialize an empty dictionary first
            self.config[key][subkey] = val
