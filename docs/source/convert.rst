=================
Converting Models
=================

pymad8 provdies converters to allow BDSIM models to prepared from optical
descriptions in MAD8.

Mad8 output required 
--------------------

To make the TWISS and RMAT output::

   use, LINE
   twiss, beta0=LINE.B0, save, tape=twiss_LINE rtape=rmat_LINE

Envelope::

   use, LINE
   envel, sigma0=LINE.SIGMA0, save, tape=envel_LINE

Survey::

   use, LINE
   survey, tape=survey_LINE


Mad8Twiss2Gmad
--------------

TBC

Mad8Saveline2Gmad
-----------------

TBC
