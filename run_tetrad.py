#! /usr/bin/env python3

import argparse
import json
import textwrap
import jpype.imports
import os
import sys
import re

import platform
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import tools.TetradSearch as ts
import tools.translate as tr

import semopy


__version_info__ = ('0', '1', '1')
__version__ = '.'.join(__version_info__)

version_history = \
"""
0.1.1 - refactored to generalize operations with run_model_search
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


        # add method to the class for loading data
        setattr(ts.TetradSearch,'load_df',self.load_df)
        
        pass

    # create a new method to load df
    def load_df(self,df):
        self.data = tr.pandas_data_to_tetrad(df)
        return self.data

    def search_init(self,df=None):
        """
        returns the search object with dummy data
        """
        
        # simple test data frame
        test_df = pd.DataFrame({
            'drinks': [1, 2, 3, 4, 5],
            'sad': [5, 4, 3, 2, 1],
            'irr': [1, 2, 3, 4, 5],
            'relax': [5, 4, 3, 2, 1],
            'happy': [1, 2, 3, 4, 5],
            'energ': [5, 4, 3, 2, 1],
            'stress': [1, 2, 3, 4, 5],
            'lagdrinks': [1, 2, 3, 4, 5],
            'lagsad': [5, 4, 3, 2, 1],
            'lagirr': [1, 2, 3, 4, 5],
            'lagrelax': [5, 4, 3, 2, 1],
            'laghappy': [1, 2, 3, 4, 5],
            'lagenerg': [5, 4, 3, 2, 1],
            'lagstress': [1, 2, 3, 4, 5]
        })
        
        if df is None:
            df = test_df
        search = ts.TetradSearch(df)
        return search

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
            # check if line begins with a number and period
            # convert line to python string
            line = str(line)
            if re.match(r"^\d+\.", line):
            # if startFlag == False:
            #     if "Graph Edges:" in line:
            #         startFlag = True
            #         continue  # continue to next line
            # if startFlag == True:
                # # check if there is edge information a '--'
                # if '-' in line:
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

    
    def test_search1(self, test_file = "data/sub_1019.csv"):
        """
        test running a search
        """
        # read in the data file into a pandas df
        df = pd.read_csv(test_file, sep=",")
        df = df.astype({col: "float64" for col in df.columns})
        
        # search = ts.TetradSearch(df)
        search = self.search_init(df)
        
        ## Use a SEM BIC score
        res =search.use_sem_bic(penalty_discount=1)

        # Set knowledge
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
        
        lavaan_model = self.edges_to_lavaan(setEdges)
        
        # run semopy
        sem_results = self.run_semopy(lavaan_model, df)
        
        # get the estmates
        estimates_sem = sem_results['estimates']
        # change column names lval to dest and rval to src
        #estimates_sem.rename(columns={'lval': 'dest', 'rval': 'src'}, inplace=True) 
        
        # summary of the estimates
        estimates = self.summarize_estimates(estimates_sem)
        
        result = {'setEdges': setEdges, 
                  'setNodes': setNodes, 
                  'setPairs': setPairs, 
                  'ESMean': estimates['mean_abs_estimates'],
                  'ESStd': estimates['std_abs_estimates'],
                  'estimatesSEM': estimates_sem
                  } 
        
        return result

    def test_search2(self, test_file = "data/sub_1019.csv"):
        """
        test running a search
        """

       # read in the data file into a pandas df
        df = self.load_dataframe(test_file)
        
        # search = ts.TetradSearch(df)
        search = self.search_init(df)
        
        # load the data into the search object
        x = search.load_df(df)
                
        ## Use a SEM BIC score
        res =search.use_sem_bic(penalty_discount=1)

        knowledge = {'addtemporal': {
            2: ['drinks', 'sad', 'irr', 'relax', 'happy', 'energ', 'stress'],
            1: ['lagdrinks', 'lagsad', 'lagirr', 'lagrelax', 'laghappy', 'lagenerg', 'lagstress']}
        }


        self.load_knowledge(search, knowledge)


        ## Run the search
        x = search.run_fges()
        soutput = search.get_string()
        setEdges, setNodes, setPairs = self.extract_edges(soutput)
        
        lavaan_model = self.edges_to_lavaan(setEdges)
        
        # run semopy
        sem_results = self.run_semopy(lavaan_model, df)
        
        # get the estmates
        estimates_sem = sem_results['estimates']
        
        # summary of the estimates
        estimates = self.summarize_estimates(estimates_sem)
        
        result = {'setEdges': setEdges, 
                  'setNodes': setNodes, 
                  'setPairs': setPairs, 
                  'ESMean': estimates['mean_abs_estimates'],
                  'ESStd': estimates['std_abs_estimates'],
                  'estimatesSEM': estimates_sem
                  } 
        
        return result

    def load_dataframe(self, file) -> pd.DataFrame:
        """
        Load a dataframe from a file
        
        Args:
        file - string with the file name
        
        Returns:
        df - pandas dataframe
        """
        df = pd.read_csv(file, sep=",")
        df = df.astype({col: "float64" for col in df.columns})
        return df

    def test_search3(self, test_file = "data/sub_1019.csv"):
        """
        test running a search
        """

       # read in the data file into a pandas df
        df = self.load_dataframe(test_file)
        
        # search = ts.TetradSearch(df)
        search = self.search_init(df)
        
        # load the data into the search object
        x = search.load_df(df)
                
        ## Use a SEM BIC score
        res =search.use_sem_bic(penalty_discount=1)

        knowledge = {'addtemporal': {
            2: ['drinks', 'sad', 'irr', 'relax', 'happy', 'energ', 'stress'],
            1: ['lagdrinks', 'lagsad', 'lagirr', 'lagrelax', 'laghappy', 'lagenerg', 'lagstress']}
        }


        self.load_knowledge(search, knowledge)

        ## Run the search
        searchResult = self.run_model_search(df, model='gfci', 
                                             knowledge=knowledge, 
                                             score={'sem_bic': {'penalty_discount': 1}},
                                             test={'fisher_z': {'alpha': .01}})
        
        #soutput = search.get_string()
        #setEdges, setNodes, setPairs = self.extract_edges(soutput)
        
        lavaan_model = self.edges_to_lavaan(searchResult['setEdges'])
        
        # run semopy
        sem_results = self.run_semopy(lavaan_model, df)
        
        # get the estmates
        estimates_sem = sem_results['estimates']
        
        # summary of the estimates
        estimates = self.summarize_estimates(estimates_sem)
        
        result = {'setEdges': list(searchResult['setEdges']), 
                  'setNodes': list(searchResult['setNodes']), 
                  'setPairs': list(searchResult['setPairs']), 
                  'ESMean': estimates['mean_abs_estimates'],
                  'ESStd': estimates['std_abs_estimates'],
                  'estimatesSEM': sem_results['estimatesDict']
                  } 
        
        return result

    def load_dataframe(self, file) -> pd.DataFrame:
        """
        Load a dataframe from a file
        
        Args:
        file - string with the file name
        
        Returns:
        df - pandas dataframe
        """
        df = pd.read_csv(file, sep=",")
        df = df.astype({col: "float64" for col in df.columns})
        return df

    
    def run_model_search(self, df, model='gfci', 
                         knowledge=None, 
                         score=None,
                         test=None):
        """
        Run a search
        
        Args:
        df - pandas dataframe
        model - string with the model to use, default gfci
        knowledge - dictionary with the knowledge
        score - dictionary with the arguments for the score
            e.g. {"sem_bic": {"penalty_discount": 1}}
            
        test - dictionary with the arguments for the test alpha 
        
        Returns:
        result - dictionary with the results
        """
        
        # search = ts.TetradSearch(df)
        search = self.search_init(df)
        
        
        # check if score is not None
        if score is not None:  
            ## Use a SEM BIC score
            if 'sem_bic' in score:
                penalty_discount = score['sem_bic']['penalty_discount']
                res =search.use_sem_bic(penalty_discount=penalty_discount)
                
        if test is not None:
            if 'fisher_z' in test:
                alpha = test['fisher_z'].get('alpha',.01)
                search.use_fisher_z(alpha=alpha)
            

        if knowledge is not None:
            self.load_knowledge(search, knowledge)
        
        ## Run the selected search
        if model == 'fges':
            x = search.run_fges()
        elif model == 'gfci':   
            x = search.run_gfci()
            

        soutput = search.get_string()
        setEdges, setNodes, setPairs = self.extract_edges(soutput)
        
        result = {'setEdges': setEdges, 
                  'setNodes': setNodes, 
                  'setPairs': setPairs
                  } 
        
        return result
    
    def load_knowledge(self, search, knowledge:dict):
        """
        Load the knowledge
        
        The standard prior.txt file looks like this:
        
        /knowledge

        addtemporal
        1 Q2_exer_intensity_ Q3_exer_min_ Q2_sleep_hours_ PANAS_PA_ PANAS_NA_ stressed_ Span3meanSec_ Span3meanAccuracy_ Span4meanSec_ Span4meanAccuracy_ Span5meanSec_ Span5meanAccuracy_ TrailsATotalSec_ TrailsAErrors_ TrailsBTotalSec_ TrailsBErrors_ COV_neuro_ COV_pain_ COV_cardio_ COV_psych_
        2 Q2_exer_intensity Q3_exer_min Q2_sleep_hours PANAS_PA PANAS_NA stressed Span3meanSec Span3meanAccuracy Span4meanSec Span4meanAccuracy Span5meanSec Span5meanAccuracy TrailsATotalSec TrailsAErrors TrailsBTotalSec TrailsBErrors COV_neuro COV_pain COV_cardio COV_psych

        forbiddirect

        requiredirect
        
        The input dict will have the keys of addtemporal, forbiddirect, requiredirect
        
        For the addtemporal key, the value will be another dict with the keys of 1, 2, 3, etc.
        representing the tiers. The values will be a list of the nodes in that tier.
        
        Args:
        search - search object
        knowledge - dictionary with the knowledge
        
        """
        
        # check if addtemporal is in the knowledge dict
        if 'addtemporal' in knowledge:
            tiers = knowledge['addtemporal']
            for tier, nodes in tiers.items():
                # tier is a number, tetrad uses 0 based indexing so subtract 1
                tier_entry = int(tier) - 1
                for node in nodes:
                    search.add_to_tier(tier_entry, node)
                    pass
        pass
        
        
    def summarize_estimates(self, df):
        """
        Summarize the estimates
        """
        # get the Estimate column from the df 
        estimates = df['Estimate']       
        # get the absolute value of the estimates
        abs_estimates = estimates.abs()
        # get the mean of the absolute values
        mean_abs_estimates = abs_estimates.mean()
        # get the standard deviation of the absolute values
        std_abs_estimates = abs_estimates.std()
        return {'mean_abs_estimates': mean_abs_estimates, 'std_abs_estimates': std_abs_estimates}
        
    def edges_to_lavaan(self, edges, exclude_edges = ['---','<->','o-o']):
        """
        Convert edges to a lavaan string
        """
        lavaan_model = ""
        for edge in edges:
            nodeA = edge.split(' ')[0]
            nodeB = edge.split(' ')[2]
            edge_type = edge.split(' ')[1]
            if edge_type in exclude_edges:
                continue
            # remember that for lavaan, target ~ source
            lavaan_model += f"{nodeB} ~ {nodeA}\n"
        return lavaan_model
    
    def run_semopy(self, lavaan_model, data):  
        
        """
        run sem using semopy package
        
        lavaan_model - string with lavaan model
        data - pandas df with data
        """
        
        # create a sem model   
        model = semopy.Model(lavaan_model)

        ## TODO - check if there is a usable model,
        ## for proj_dyscross2/config_v2.yaml - no direct edges!
        ## TODO - also delete output files before writing to them so that
        ## we don't have hold overs from prior runs.
        opt_res = model.fit(data)
        estimates = model.inspect()
        stats = semopy.calc_stats(model)
        
        # change column names lval to dest and rval to src
        estimatesRenamed = estimates.rename(columns={'lval': 'dest', 'rval': 'src'})
        # convert the estimates to a dict using records
        estimatesDict = estimatesRenamed.to_dict(orient='records')        

        return ({'opt_res': opt_res,
                 'estimates': estimates, 
                 'estimatesDict': estimatesDict,
                 'stats': stats})
        
            
if __name__ == "__main__":
    
    # provide a description of the program with format control
    description = textwrap.dedent('''\
    Program to run a Tetrad search using the py-tetrad package.
    

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
                    help="cmd - [test1,test2], default test1",
                    default = 'test1')

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
    
    if args.cmd == 'test1':
        obj.test_search1()
    elif args.cmd == 'test2':
        result1 = obj.test_search1()
        result2 = obj.test_search2()  
        # compare results
        if result1['setEdges'] == result2['setEdges']:
            print("setEdges are the same")
        pass  
    elif args.cmd == 'test3':
        results3 = obj.test_search3()
        # save results into a json file
        with open('results3.json', 'w') as f:
            json.dump(results3, f, indent=4)
        pass

