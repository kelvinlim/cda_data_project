import jpype.imports
import os
import platform
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

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
    
except OSError:
    pass

import pandas as pd
import tools.TetradSearch as ts
import tools.translate as tr

NUM_ITERATIONS = 100

def dice_coefficient(set_a, set_b):
    # Calculate the intersection of the two sets
    intersection = len(set_a.intersection(set_b))
    
    # Calculate the Dice coefficient
    dice = (2 * intersection) / (len(set_a) + len(set_b))
    
    return dice

def extract_edges(text):
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
    
# read in the data file into a pandas df
test_file = "data/sub_1019.csv"
df = pd.read_csv(test_file, sep=",")
df = df.astype({col: "float64" for col in df.columns})



# create a new method to load df
def load_df(self,df):
    self.data = tr.pandas_data_to_tetrad(df)
# add method to the class
setattr(ts.TetradSearch,'load_df',load_df)

def run_proportions(df, subj, verbose=False):
    # create the TetradSearch object
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
            x = search.load_df(resampled_df)
            
            ## Run the search
            x = search.run_fges()
            soutput = search.get_string()
            setEdges, setNodes, setPairs = extract_edges(soutput)
            if proportion==1.0:
                fullEdges = setEdges.copy()  # full edges from 100%
                fullNodes = setNodes.copy()
                fullPairs = setPairs.copy()
                
            # calculate the dice coefficient
            diceCoeff = dice_coefficient(fullEdges,setEdges)
            diceCoeffNode = dice_coefficient(fullNodes,setNodes)
            diceCoeffPair = dice_coefficient(fullPairs,setPairs)
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
    

summaryDf = None

for file in ["data/sub_1008.csv","data/sub_1019.csv","data/sub_1033.csv"]:
    df = pd.read_csv(test_file, sep=",")
    df = df.astype({col: "float64" for col in df.columns})
    subj = Path(file).stem.split('_')[1]
    df_result = run_proportions(df, subj)
    
    # concatenate the dataframes
    if summaryDf is None:
        summaryDf = df_result.copy()
    else:
        summaryDf = pd.concat([summaryDf,df_result],ignore_index=True)
        pass


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