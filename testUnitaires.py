import unittest
import app 
import artefacts
import pandas as pd
import numpy as np
 
class Test_Api_Python(unittest.TestCase) :
    
    data = [['45:23']]
    df = pd.DataFrame(data, columns = ['time'])
    dfToTest = app.secondConverter(df['time'])    
       
    valueForTest = 2723    
    
    smooValue = [[112]]
    dfSmoo = pd.DataFrame(smooValue, columns = ['HR'])
    dfSmoothing = artefacts.smoothing(dfSmoo, 0.63, 'soft', 'db8', 'per')
    
    signal = dfSmoo.HR.values
    dfLowpassfilter = artefacts.lowpassfilter(dfSmoo, signal, 0.63, 'soft', 'db8', 'per', 'per')
    
    def test_secondConverter(self):                
        assert type(self.dfToTest) is pd.Series
        assert self.dfToTest[0] == self.valueForTest
    
    def test_smoothing(self):                
        assert type(self.dfSmoothing) is pd.DataFrame
    
    def test_lowpassfilter(self):                
        assert type(self.dfLowpassfilter) is pd.DataFrame
        
        
        

if __name__ == "__main__":
    unittest.main()
    