import jpype.imports

# try:
#     #jpype.startJVM(classpath=[f"resources/tetrad-current.jar"])
#     jpype.startJVM(classpath=[f"py-tetrad/pytetrad/resources/tetrad-current.jar"])
    
# except OSError:
#     pass

import pandas as pd
import tools.TetradSearch as ts

test_file = "py-tetrad/pytetrad/resources/airfoil-self-noise.continuous.txt"
df = pd.read_csv(test_file, sep="\t")
df = df.astype({col: "float64" for col in df.columns})

search = ts.TetradSearch(df)

## Use a SEM BIC score
search.use_sem_bic(penalty_discount=1)

## Set knowledge
search.add_to_tier(0, "Frequency")
search.add_to_tier(0, "Attack")
search.add_to_tier(0, "Chord")
search.add_to_tier(1, "Velocity")
search.add_to_tier(1, "Displacement")
search.add_to_tier(1, "Pressure")

## Run the search
search.run_fges()
print("=====\nfges\n=====")
print(search.get_string())

print("=====\ngfci\n=====")
# test
search.use_fisher_z()
search.run_gfci()
print(search.get_string())

print("=====\ndagma\n=====")
search.run_dagma()
print(search.get_string())

print("=====\ndirect_lingam\n=====")
search.run_direct_lingam()
print(search.get_string())

## Print all subsets independence facts.
print(search.all_subsets_independence_facts(search.get_java()))