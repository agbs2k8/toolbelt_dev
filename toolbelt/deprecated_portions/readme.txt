bsa.py
	fix_bsa_email
		Timing:
			a. With the domains_file load in each iteration of the function:
				1. Looping through 100,000 itertations of most complex fix:
					i. Elapsed Time: 1109 seconds
					ii. Time per Iteration: 0.01109 seconds
				2. Apply to 100,000 iterations of the same emails to fix in pandas data frame:
					i. Elapsed Time: 1100 seconds
					ii. Time per Iteration: 0.01100 seconds 
			b. With the domains_file load with init without function call
				1. Looping through 100,000 itertations of most complex fix:
					i. Elapsed Time: 980.5 seconds
					ii. Time per Iteration: 0.00980 seconds
				2. Apply to 100,000 iterations of the same emails to fix in pandas data frame:
					i. Elapsed Time: 981.2 seconds
					ii. Time per Iteration:  0.00981 seconds
			c. Multiprocessing (Re-run with pandas apply to validate any difference) - 2 Processor Computer
				1.Passing a list of 100,000 of the most complex fix:
					i. Elapsed Time: 467.65
					ii. Time per Iteration: 0.0046765
					iii. Validation Runs
						1. Elapsed time: 1047
						2. Time Per Iteration: 0.01048 
			d. Conclusion: With large lists, multithreading is the way to go!