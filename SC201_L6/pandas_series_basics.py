"""
File: pandas_series_basics.py
Name:
-----------------------------------
This file shows the basic pandas syntax, especially
on Series. Series is a single column
of data, similar to what a 1D array looks like.
We will be practicing creating a pandas Series and
call its attributes and methods
"""

import pandas as pd


def main():
    s = pd.Series([20,20,10])
    new_s = s.append(pd.Series([30,20,20]), ignore_index= True)
    print(nes_s)


if __name__ == '__main__':
    main()
