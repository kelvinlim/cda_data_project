# csci5465_project

Code for my csci5465 project.  This project will use the jpype interface 
to tetrad to speed up processing compared with using causal-cmd.

Possible scenarios are to do analysis of the effect of 
number of samples in ema analysis by comparing the graphs with the
full dataset and different percentages of sampling such as
90%, 80% 70%, 60%, 50%.

It would be possible to do this 100 times at each sampling and 
compare the metrics.  The proposed metric would be to use the
Dice coefficient. 

Some cases to use with fges

Stored in data directory

1. sub_1019 has 118 rows and 13(direct)/15 edges
2. sub_1008 has 100 rows and 7(direct)/13 edges
3. sub_1033 has 93 rows and  10(direct)/10 edges
4. sub_1112 has 89 rows and 9(direct)/15 edges 