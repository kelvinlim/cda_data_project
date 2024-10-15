#! /usr/bin/env python3

import jpype.imports
import os
import platform
import argparse
import textwrap
import sys

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
# import tools.TetradSearch as ts
# import tools.translate as tr

from run_tetrad import TetradWrap

NUM_ITERATIONS = 100

__version_info__ = ('0', '1', '0')
__version__ = '.'.join(__version_info__)

version_history = \
"""
0.1.0 - initial version  
"""

class MerrillFGES:
    
    
    def __init__(self, **kwargs):
        
        # load self.config
        self.config = {}
        for key, value in kwargs.items():
            self.config[key] = value
            
        
        self.tetrad_wrap = TetradWrap()
        self.tetrad_wrap.jvm_initialize()
        # result = self.tetrad_wrap.test_search()
    
        self.search = self.tetrad_wrap.search_init()
    
        pass
    
    def dice_coefficient(self, set_a, set_b):
        # Calculate the intersection of the two sets
        intersection = len(set_a.intersection(set_b))
        
        # Calculate the Dice coefficient
        dice = (2 * intersection) / (len(set_a) + len(set_b))
        
        return dice


    def run_proportions(self,df, subj, verbose=False):

            
        ## Use a SEM BIC score
        res =self.search.use_sem_bic(penalty_discount=1)

        ## Set knowledge
        self.search.add_to_tier(0, "lagdrinks")
        self.search.add_to_tier(0, "lagsad")
        self.search.add_to_tier(0, "lagirr")
        self.search.add_to_tier(0, "lagrelax")
        self.search.add_to_tier(0, "laghappy")
        self.search.add_to_tier(0, "lagenerg")
        self.search.add_to_tier(0, "lagstress")
        self.search.add_to_tier(1, "drinks")
        self.search.add_to_tier(1, "sad")
        self.search.add_to_tier(1, "irr")
        self.search.add_to_tier(1, "relax")
        self.search.add_to_tier(1, "happy")
        self.search.add_to_tier(1, "energ")
        self.search.add_to_tier(1, "stress")


        # create lists to hold information for creating a df
        caseList = list()
        proportionList = list()
        iterationList = list()
        diceCoeffList = list()
        diceCoeffNodeList = list()
        diceCoeffPairList = list()

        # loop for the proportion of data to sample

        for proportion in [1.0,0.9,0.8,0.7,0.6,0.5,0.4]:
            for iteration in range(NUM_ITERATIONS):
                # resample the dataframe based on proportion
                resampled_df = df.sample(frac=proportion)
                
                # load the data
                x = self.search.load_df(resampled_df)

                # setEdges, setNodes, setPairs = self.extract_edges(soutput)
                if proportion==1.0:
                    fullEdges = setEdges.copy()  # full edges from 100%
                    fullNodes = setNodes.copy()
                    fullPairs = setPairs.copy()
                    
                # calculate the dice coefficient
                diceCoeff = self.dice_coefficient(fullEdges,setEdges)
                diceCoeffNode = self.dice_coefficient(fullNodes,setNodes)
                diceCoeffPair = self.dice_coefficient(fullPairs,setPairs)
                if verbose: print(f"Proportion: {proportion} Iteration: {iteration+1} diceCoeff: {diceCoeff}")
                
                # load into lists
                caseList.append(subj)
                proportionList.append(proportion)
                iterationList.append(iteration)
                diceCoeffList.append(diceCoeff)
                diceCoeffNodeList.append(diceCoeffNode)
                diceCoeffPairList.append(diceCoeffPair)
                pass
            pass

        # create dataframe
        summaryDf = pd.DataFrame(  {"case": caseList,
                                    "proportion": proportionList,
                                    "iteration": iterationList,
                                    "diceCoeff": diceCoeffList,
                                    "diceCoeffNodes": diceCoeffNodeList,
                                    "diceCoeffPairs": diceCoeffPairList}
                                )

        return summaryDf
    
    def create_plots(self, summaryDf):

        # create the plots
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='proportion', y='diceCoeff', data=summaryDf)
        plt.title(f'Violin Plots at Different Proportions for edges')
        plt.savefig('edges.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='proportion', y='diceCoeffNodes', data=summaryDf)
        plt.title(f'Violin Plots at Different Proportions for nodes')
        plt.savefig('nodes.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='proportion', y='diceCoeffPairs', data=summaryDf)
        plt.title(f'Violin Plots at Different Proportions for pairs')
        plt.savefig('pairs.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()

    def work_1(self):
        summaryDf = None

        for file in ["data/sub_1008.csv","data/sub_1019.csv","data/sub_1033.csv"]:
            df = pd.read_csv(file, sep=",")
            df = df.astype({col: "float64" for col in df.columns})
            subj = Path(file).stem.split('_')[1]
            df_result = self.run_proportions(df, subj)

            # concatenate the dataframes
            if summaryDf is None:
                summaryDf = df_result.copy()
            else:
                summaryDf = pd.concat([summaryDf,df_result],ignore_index=True)
                pass

        return summaryDf
    



if __name__ == "__main__":
    
    # provide a description of the program with format control
    description = textwrap.dedent('''\
    A description of the program goes here.
    

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

    # obj = TetradWrap(   cmd=args.cmd, 
    #                     env=args.env, 
    #                     verbose=args.verbose, 
    #                     config=args.config,
    #                     format=args.format,
    #                 )   
    
    # obj.jvm_initialize()
    # obj.test_search()
    
    # tetradSearch = obj.search_init()
    # create the MerrillFGES object
    m = MerrillFGES()
    summaryDf = m.work_1()