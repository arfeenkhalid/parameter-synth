import pytest
from TeLEX.telex.stl import parse
from TeLEX.telex.parametrizer import getParams, setParams
from TeLEX.telex.synth import explore
from TeLEX.telex.scorer import qualitativescore, quantitativescore, smartscore
import TeLEX.telex.inputreader as inputreader
import pandas
import math
import sys

"""templogicdata = [
    'G[0,20]({ppERK / totERK} < 0.95)',
    'F[0,24](({ppERK / totERK} < 0.1) & F[0,22]({ppERK / totERK} > 0.5))',
    'F[0,300](({ppERK / totERK} > 0.5) & F[0,300]({ppERK / totERK} < 0.1))'
]"""

"""templogicdata = [
    'G[0,300]({ppERK / totERK} < 0.95)'
]"""

#@pytest.mark.parametrize("tlStr", templogicdata)
def test_stl(tlStr, trace_file_name, program_time_stamp):
    stl = parse(tlStr)
    param = getParams(stl)
    valmap = explore(param)
    stl1 = setParams(stl, valmap)
    #print(stl)
    #print(stl1)
    #x = pandas.DataFrame([[1,2, True], [1,4, True], [4,2, False], [1,2, True], [1,4, True], [4,2, False], [1,2, True], [1,4, True], [4,2, False], [4,2, False]], index=[0,1,2,3,4,5,6,7,8,9], columns=["x1", "x2", "x3"])
    x = inputreader.readtracefile("output/" + program_time_stamp + "/" +trace_file_name+ ".csv")
    try:
        boolscore = qualitativescore(stl1, x, 0)
        #print("Qualitative Score: ", boolscore)
    except ValueError:
        print("Value error")
    try:
        quantscore = quantitativescore(stl1, x, 0)        
        #TODO: this check is a quick fix. Need to figure out why scorer.py results in division by 0 and causes the quantscore to return 'nan'.
	#NOTE: Supressing division by zero warnings for video demo. 
        if math.isnan(quantscore):
             quantscore = 0.0
        #print("Quantitative Score: ", quantscore)
        #print("\n")
    except ValueError:
        print("Value error in quant")
    """try:
        sscore = smartscore(stl1, x, 0)
        print("Smart Score: ", sscore)
    except ValueError:
        print("Value error in smart")"""
    return quantscore        
  
def main(trace_file_name, program_time_stamp, templogicdata):
    if not sys.warnoptions:
       import warnings
       warnings.simplefilter("ignore")

    """for templ in templogicdata:
        robustscore = test_stl(templ, trace_file_name)"""
    robustscore = test_stl(templogicdata, trace_file_name, program_time_stamp)
    return robustscore

if __name__ == "__main__":
    main(trace_file_name, program_time_stamp, templogicdata)
