install:

	@echo "\nCreating conda environment for parameter-synth....."	
	conda env create -f parameter-synth.yml

run:

	@echo "\nActivating parameter-synth conda environment....."
	@./run_estimation_process.sh $(spec) $(n)

#ifeq ($(spec), t-cell_1)
#	./run_estimation_process.sh $(spec)
#else ($(spec), t-cell_2)
#	./run_estimation_process.sh $(spec)
#else
#	@echo "\nNo specification file found by this name. Please try a different name."
#endif
	
