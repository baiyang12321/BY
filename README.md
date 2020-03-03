# This code includes the main program and multiple functions. Among them, "CM_FCM_Cross" is the main program, and the remaining m files are functions；
# In the main program, you first need to enter training data and prediction data in the format of an Excel table；
# Secondly, the data is pre-processed and the gate network is calculated；
# Then, three experts processed and trained the data allocated by the gate network to obtain multiple sub-models；
# Finally, use the combiner to combine the outputs of the three experts to get the final output；
# This code can adaptively analyze the logging data, but does not include the MIV method to screen sensitive data；
# Because the format of the input data is noted in the code, and the original data is confidential, we do not provide examples.
