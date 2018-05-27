"""
pymad8 - python tools for working with MAD8 output and input.

| Dependencies:
| package     - minimum version required
| numpy       - 1.7.1
| matplotlib  - 1.3.0

| Modules:
| Input         -
| Output        -
| Plot          -
| Sim           -
| Track         -
| Visualisation -

Copyright Royal Holloway, University of London 2018.
"""

__version__ = "1.3"

# import Mad8
import Input
import Output
import Plot
import Sim
#import Track  #not imported by default - can be explicitly imported
import Visualisation

#import Saveline
#import Converter
#import Track

__all__ = ['Input',
           'Output',
           'Plot',
           'Sim',
           'Visualisation'
]
