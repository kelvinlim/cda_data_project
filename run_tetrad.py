#! /usr/bin/env python3

import argparse
import textwrap
import jpype.imports
import os
import sys
import platform
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import tools.TetradSearch as ts
import tools.translate as tr


__version_info__ = ('0', '1', '0')
__version__ = '.'.join(__version_info__)

version_history = \
"""
0.1.0 - initial version  
"""

class TetradWrap:
    
    """
    Class to wrap the Tetrad search functions
    includes SEM
    """
    
    
    def __init__(self, **kwargs):
        
        # load self.config
        self.config = {}
        for key, value in kwargs.items():
            self.config[key] = value
            
        pass


    def dice_coefficient(self, set_a, set_b):
        # Calculate the intersection of the two sets
        intersection = len(set_a.intersection(set_b))
        
        # Calculate the Dice coefficient
        dice = (2 * intersection) / (len(set_a) + len(set_b))
        
        return dice

    def extract_edges(self, text):
        """
        Extract out the edges between Graph Edges and Graph Attributes
        """
        edges = set()
        nodes = set()
        pairs = set()  # alphabetical order of nodes of an edge
        # get the lines
        lines = text.split('\n')
        startFlag=False  # True when we are in the edges, False when not
        for line in lines:
            if startFlag == False:
                if "Graph Edges:" in line:
                    startFlag = True
                    continue  # continue to next line
            if startFlag == True:
                # check if there is edge information a '--'
                if '--' in line:
                    # this is an edge so add to the set
                    # strip out the number in front  1. drinks --> happy
                    # convert to a string
                    linestr = str(line)
                    clean_edge = linestr.split('. ')[1]
                    edges.add(clean_edge)
                    
                    # add nodes
                    nodeA = clean_edge.split(' ')[0]
                    nodes.add(nodeA)
                    nodeB = clean_edge.split(' ')[2]
                    nodes.add(nodeB)
                    combined_string = ''.join(sorted([nodeA, nodeB]))
                    pairs.add(combined_string)
                    pass
        return edges, nodes, pairs    

    def jvm_initialize(self):
        """
        initialize the jvm
        """
        
        
        # get platform information
        uname_info = platform.uname()


        if uname_info.node == 'c30-wsacsk4hm3':
            # avd is azure virtual desktop version 

            # set JAVA_HOME
            os.environ['JAVA_HOME'] = "R:/DVBIC/jdk21.0.4_7"
            # check for env JAVA_HOME, this was None
            java_home = os.environ.get('JAVA_HOME')
    
        try:
            # works for win11
            jpype.startJVM(classpath=[f"resources/tetrad-current.jar"])
            #jpype.startJVM(classpath=[f"py-tetrad/pytetrad/resources/tetrad-current.jar"])
            pass
        except OSError as e:
            if e.args[0] == "JVM is already started":
                pass
            else:    
                print(f"Error occurred: {e}")
                sys.exit(1)
            pass


        # add method to the class
        setattr(ts.TetradSearch,'load_df',self.load_df)
        
        pass

    # create a new method to load df
    def load_df(self,df):
        self.data = tr.pandas_data_to_tetrad(df)

    def test_search(self):
        """
        test running a search
        """
        # read in the data file into a pandas df
        test_file = "data/sub_1019.csv"
        df = pd.read_csv(test_file, sep=",")
        df = df.astype({col: "float64" for col in df.columns})
        
        search = ts.TetradSearch(df)
        
        ## Use a SEM BIC score
        res =search.use_sem_bic(penalty_discount=1)

        ## Set knowledge
        search.add_to_tier(0, "lagdrinks")
        search.add_to_tier(0, "lagsad")
        search.add_to_tier(0, "lagirr")
        search.add_to_tier(0, "lagrelax")
        search.add_to_tier(0, "laghappy")
        search.add_to_tier(0, "lagenerg")
        search.add_to_tier(0, "lagstress")
        search.add_to_tier(1, "drinks")
        search.add_to_tier(1, "sad")
        search.add_to_tier(1, "irr")
        search.add_to_tier(1, "relax")
        search.add_to_tier(1, "happy")
        search.add_to_tier(1, "energ")
        search.add_to_tier(1, "stress")

        # load the data
        x = search.load_df(df)
            
        ## Run the search
        x = search.run_fges()
        soutput = search.get_string()
        setEdges, setNodes, setPairs = self.extract_edges(soutput)
        pass

if __name__ == "__main__":
    
    # provide a description of the program with format control
    description = textwrap.dedent('''\
    A description of the program goes here.
    
    Account information is read from a .env file which contains the 
    APITOKEN, DATACENTER and DIRECTORYID.
    
    Here are some examples of using the command. Text following the $ is
    the command that is entered at the command line in a terminal window.
    
    $ LNPIQualtrics
    Without any arguments, the mailingLists are listed with their index. 
    ''')
    
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--env", type = str,
                     help="name of env file in the current directory, default .env",
                      default=".env") 

    parser.add_argument("--config", type = str,
                     help="name of yaml config file in the current directory, default config.yaml",
                      default="config.yaml") 
        
    parser.add_argument("--cmd", type = str,
                    help="cmd - [list, summarize], default list",
                    default = 'list')

    parser.add_argument("--format", type = str,
                    help="format to use, default json",
                    default = 'json')
    
    parser.add_argument("-H", "--history", action="store_true", help="Show program history")
     
    # parser.add_argument("--quiet", help="Don't output results to console, default false",
    #                     default=False, action = "store_true")  
    
    parser.add_argument("--verbose", type=int, help="verbose level default 2",
                         default=2) 
        
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    if args.history:
        print(f"{os.path.basename(__file__) } Version: {__version__}")
        print(version_history)
        exit(0)

    obj = TetradWrap(   cmd=args.cmd, 
                        env=args.env, 
                        verbose=args.verbose, 
                        config=args.config,
                        format=args.format,
                    )   
    
    obj.jvm_initialize()
    obj.test_search()





