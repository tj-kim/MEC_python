import sys
import numpy as np
import matplotlib.pyplot as plt

if(len(sys.argv)<2):
    print ("Usage error")
    exit(0)
filename=sys.argv[1]
fd=open(filename,"r")

Lines = fd.readlines()
de=[]
# Strips the newline character
for line in Lines:
	line=line.split("\n")[0]
	id1=line.split(" ")
	if(float(id1[-1])>500.00):
		de.append(250.0)
	else:
		de.append(float(id1[-1]))
plt.plot(de)
plt.ylim(top=250.0,bottom=0.0)
plt.show()
