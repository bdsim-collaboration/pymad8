=================
Converting Models
=================

pymad8 provdies converters to allow BDSIM models to prepared from optical
descriptions in MAD8.

For conversion of MAD8 to BDSIM GMAD format, please see the pybdsim documentation
`<http://www.pp.rhul.ac.uk/bdsim/pybdsim/convert.html#mad8-twiss-2-gmad>`_.

Mad8 output required
--------------------

Listed here are the MAD8 lines required to generate various output files.

To make the Twiss and Rmat outputs : ::

   use, LINE
   twiss, beta0=LINE.B0, save, couple, tape=TWISS_LINE rtape=RMAT_LINE

To make the Chrom output : ::

   use, LINE                                                                                           
   twiss, beta0=LINE.B0, save, chrom, tape=CHROM_LINE

To make the Envelope output : ::

   use, LINE
   envel, sigma0=LINE.SIGMA0, save, tape=ENVEL_LINE

To make the Survey output : ::

   use, LINE
   survey, tape=SURVEY_LINE

