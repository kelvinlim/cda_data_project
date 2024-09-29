import jpype.imports

try:
    #jpype.startJVM(classpath=[f"resources/tetrad-current.jar"])
    jpype.startJVM(classpath=[f"py-tetrad/pytetrad/resources/tetrad-current.jar"])
    
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
                pass
    return edges
    
# read in the data file into a pandas df
test_file = "data/sub_1019.csv"
df = pd.read_csv(test_file, sep=",")
df = df.astype({col: "float64" for col in df.columns})



# create a new method to load df
def load_df(self,df):
    self.data = tr.pandas_data_to_tetrad(df)
# add method to the class
setattr(ts.TetradSearch,'load_df',load_df)

# create the TetradSearch object
search = ts.TetradSearch(df)
    
## Use a SEM BIC score
search.use_sem_bic(penalty_discount=1)

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
proportionList = list()
iterationList = list()
diceCoeffList = list()

# loop for the proportion of data to sample

for proportion in [1.0,0.9,0.8,0.7,0.6,0.5,0.4]:
    for iteration in range(NUM_ITERATIONS):
        # resample the dataframe based on proportion
        resampled_df = df.sample(frac=proportion)
        # load the data
        search.load_df(resampled_df)
        
        ## Run the search
        search.run_fges()
        soutput = search.get_string()
        setEdges = extract_edges(soutput)
        if proportion==1.0:
            fullEdges = setEdges.copy()  # full edges from 100%
            
        # calculate the dice coefficient
        diceCoeff = dice_coefficient(fullEdges,setEdges)
        print(f"Proportion: {proportion} Iteration: {iteration+1} diceCoeff: {diceCoeff}")
        
        # load into lists
        proportionList.append(proportion)
        iterationList.append(iteration)
        diceCoeffList.append(diceCoeff)
        pass
    pass

# create dataframe
summaryDf = pd.DataFrame(  {"proportion": proportionList,
                            "iteration": iterationList,
                            "diceCoeff": diceCoeffList
                            }
                        )

# write out to csv
with open("proportion_summary.csv","w") as fp:
    summaryDf.to_csv(fp, index=False)
