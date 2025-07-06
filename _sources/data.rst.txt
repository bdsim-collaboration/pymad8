================================
MAD8 File Loading & Manipulation
================================

MAD8 outputs Twiss information in their own file format.
pymad8 includes a class called Output for the purpose of loading and manipulating this data.

The MAD8 format is described in the manual available from `the mad8 website <http://mad8.web.cern.ch>`_.
The format is roughly described as a text file.
The MAD8 files contain first a few lines that indicate when the file was generated and other general informations.
After this, each segment of 5 lines typically represents the values of the lattice for a particular element with each new segment containing the values at a subsequent element in the lattice.

Output Class Features
---------------------

 * Loading different types of MAD8 files.
 * Get a particular column.
 * Get a particular row.
 * Get elements of a particular type.
 * Get a numerical index from the name of the element.
 * Find the curvilinear S coordinate of an element by name.
 * Find the name of the nearest element at a given S coordinate.
 * Plot an optics diagram.
 * Calculate a beam size given the Twiss parameters, dispersion and emittance.
 * Make a slice of the initial lattice

Loading
-------

MAD8 files can be of different types.
Twiss files are the main ones but we can also load Rmat files, Chrom files, Envelope files or Survey files

A file may be loading by constructing an Output instance from a file name :

>>> import pymad8
>>> t = pymad8.Output("myTwissFile")
>>> r = pymad8.Output("myRmatFile", "rmat")
>>> c = pymad8.Output("myChromFile", "chrom")
>>> e = pymad8.Output("myEnvelopeFile", "envel")
>>> s = pymad8.Output("mySurveyFile", "survey")

.. note:: The import will be assumed from now on in examples.

Querying
--------

The Output class can be used to query the data in various ways.

Basic Information
*****************

 * All data is stored in the **data** object inside the class
 * The number of elements is stored in **nrec**.
 * The file name is stored in **filename**.
 * The file type is stored in **filetype**.
 * A dict of each element type and corresponding properties is stored in **keys_dict**
 * The names of columns common to all file types is stored in **colnames_common**.

The **data** object is a pandas dataframe that can be displayed as follow :

>>> t = pymad8.Output("myTwissFile")
>>> t.data

Indexing and Slicing
********************

The information stored in the dataframe is accessible using regular pandas syntax : ::
  
  t.data.iloc[3]                      # 4th element in sequence (Pandas.Series returned)
  t.data.iloc[3: 10]                  # 4th to 11th elements (Pandas.Dataframe returned)
  t.data.iloc[[3, 5, 10]]             # 4th, 6th and 11th elements (Dataframe)
  t.data['S']                         # column named exactly S (Series)
  t.data[['S', 'L']]                  # columns named exactly S and L (Dataframe)
  t.data['S'][3]                      # value of the 4th element in the column S
  t.data[t.data['NAME'] == 'INITIAL'] # Row of the element with this exact name (Dataframe)

But you can also find information about elements usind built-in functions.

To get index for example ::

  t.getIndexByNames('INITIAL')              # can have a list of names as input
  t.getIndexByTypes('QUAD')                 # can have a list of types as input
  t.getIndexByValues(key='S', minValue=200) # indices of elements with S value above 200
  t.getIndexByNearestS(150)                 # index of element with S value closest to 150

The results are returned in the form of one value or a list of values, depending on the input given.
In the case of the getIndex function, we get either an integer or a list of integers 

.. note:: Similar functions are avaliable to find names and types of lattice elements

Rows and Columns
****************

A row of data is an entry for a particular element. The Output class is conceptually a list of
elements. Each element is represented by a row in the pandas dataframe that has a key for each column.
The list of acceptable keys (i.e. names of columns) can be found in the member named 'colums' : ::

  t.data.columns #prints out list of column names

A specific row or set of rows can be accessed using similar functions as those previously shown : ::

  t.getRowsByIndex(3)                      # can have a list of indices as input
  t.getRowsByNames('INITIAL')              # can have a list of names as input
  t.getRowsByTypes('QUAD')                 # can have a list of types as input
  t.getRowsByValues(key='S', minValue=200) # rows of elements with S value above 200
  t.getRowByNearestS(150)                  # row of element with S value closest to 150

The results are return either in the form of a dataframe or serie (which is equivalent to a dataframe with only one row), depending on the input given. 

A specific column or set of columns can be accessed using its keys (i.e. its names) : ::

  t.getColumnsByKeys(['S','L'])

Beam Sizes
----------

For convenience the beam size is calculated from the Beta amplitude functions, the emittance, dispersion  and enegy spread using `calcBeamSize()`.
The emittance is defined by 'EmitX' and 'EmitY' and the energy spread by 'Esprd'.
Those three parameters aren't provided by MAD8 and must be manualy given to the function : ::

  EmitX = 3e-11
  EmitY = 3e-11
  Esprd = 1e-6
  t.calcBeamSize(EmitX, EmitY, Esprd)

In this function, the beam sizes are calculated according to :

.. math::

   \sigma_x &= \sqrt{ \beta_x \epsilon_x + D(S)^2 \frac{\sigma_{E}^{2}}{E_{0}^{2}}} \\
   \sigma_y &= \sqrt{ \beta_y \epsilon_y + D(S)^2 \frac{\sigma_{E}^{2}}{E_{0}^{2}}}

