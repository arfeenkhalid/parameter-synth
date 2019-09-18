from __future__ import division
import numpy as np
import subprocess
import csv
import os
import time
import random
import sys
from datetime import datetime
from copy import deepcopy
from TeLEX.tests import test_robustcalc as rc
from param import Param
from param import Point
from file_handler import FileHandler
from collections import OrderedDict
import signal

trace_file_name = "trace"
model_file_name = ""
spec_file_name = "t-cell_1"
process_number = 1
spec_list = []
confidence = []
opt_iteration = 1
program_time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
PARAM_SPACE_LOWER_RANGE = -3
PARAM_SPACE_UPPER_RANGE = 3
BIONETGEN_SIMULATION_TIMEOUT = 90 #1.5 mins 
file_handler = FileHandler(program_time_stamp, model_file_name, confidence)
iteration_start_time = 0.0
iteration_end_time = 0.0

#simulated annealing
MIN_TEMPERATURE = 10
current_temperature = 100000
COOLING_RATE = 0.99
PARAM_PERTURBATION_FACTOR = 0.1

#hyp test
#confidence = [0.85, 0.80]
delta = 0.1
alpha = 0.005
beta = 0.2
MAX_TRIALS_FOR_HYPOTHESIS_TESTING = 1000

def main(argv):
    global spec_list
    global model_file_name
    global iteration_start_time
    global iteration_end_time
    global opt_iteration
    global current_temperature
    global confidence

    signal.signal(signal.SIGTERM, terminateProcess)

    if not sys.warnoptions:
       import warnings
       warnings.simplefilter("ignore")

    #print("Required Probabilities: ", confidence)
    #print("\nDelta: ", delta)
    #print("\nAlpha: ", alpha)
    #print("Beta: ", beta)
    #print("Minimum Temperature: ", MIN_TEMPERATURE)
    #print("Current Temperature: ", current_temperature)
    #print("Cooling Rate: ", COOLING_RATE)
    #print("Program Timestamp: ", program_time_stamp)
    #print("-----------------------------------------------------------------")     

    iteration_start_time = time.time()    
    
    file_handler.spec_file_name = argv[0] if len(argv) > 0 else spec_file_name
    spec_file = file_handler.read_spec_file()

    if len(argv) > 1:
        process_number = int(argv[1])
    
    np.random.seed(int(argv[2])) if len(argv) > 2 else np.random.seed()

    param_list = []
    no_of_params = int(spec_file["no_of_params"])
    for param_no in range(no_of_params):
        param = spec_file["param" + str(param_no)].split(";")
        param_list.append(Param(param[0], float(param[1]), float(param[2]), float(param[3])))

    spec_list = []
    no_of_specs = int(spec_file["no_of_specs"])
    model_file_name = spec_file["model_file_name"]
    for spec_no in range(no_of_specs):
        spec = spec_file["spec" + str(spec_no)]
        spec_list.append(spec)
        conf = float(spec_file["req_prob" + str(spec_no)])
        confidence.append(conf)

    #updating no_of_parameters for file_handler object
    file_handler.no_of_params = no_of_params
    file_handler.no_of_specs = no_of_specs
    file_handler.model_file_name = model_file_name
    file_handler.create_dir_for_prog_run()
    file_handler.confidence = confidence

    print("Running Parameter Estimation Process "+str(process_number)+"...")
    is_parameters_synthesized = False
    is_process_timed_out = True

    #keep looking for initial points until we get a set of points where bionetgen simulations take a reasonable amount of time to execute
    while is_process_timed_out == True:
        point_vector = []
        for i in range(no_of_params):
            #rand = np.random.random_integers(PARAM_SPACE_LOWER_RANGE, PARAM_SPACE_UPPER_RANGE)
            rand = np.random.uniform(PARAM_SPACE_LOWER_RANGE, PARAM_SPACE_UPPER_RANGE)
            point_vector.append(param_list[i].init_val * np.power(10, float(rand)))
        initial_point = Point(point_vector, 0, 0)
        #print(initial_point.vector)
        #return

        file_handler.sa_generate_model_file_with_new_point(initial_point, param_list, opt_iteration)

        hyp_test_result, mean, is_process_timed_out = perform_hyp_test()

        opt_iteration += 1

        #iteration_end_time = time.time() - iteration_start_time
        #print("\n\n-----------------------------------------------------------------")
        #print("Total time taken till now(in sec): ", iteration_end_time, "s")
        #print("Total time taken till now(in hrs): ", iteration_end_time/60.0/60.0, "h")

    if hyp_test_result == 1:
        is_parameters_synthesized = True
        #current_point = deepcopy(initial_point_1)
        #print("Hypothesis test result: ", hyp_test_result, ". H1 accepted")
    elif hyp_test_result == 0:
        is_parameters_synthesized = False
        #current_point = deepcopy(initial_point_2)
        #print("Hypothesis test result: ", hyp_test_result, ". H1 rejected")
    else:
        is_parameters_synthesized = False
        #print("Hypothesis test result: ", hyp_test_result, ". need more samples...")

    initial_point.mean = mean
    current_point = deepcopy(initial_point)

    while current_temperature > MIN_TEMPERATURE and is_parameters_synthesized == False:
        is_process_timed_out = True
        while is_process_timed_out == True:
            new_point = select_a_neighbouring_point(current_point, param_list, no_of_params)
            previous_point = deepcopy(current_point)
            current_point = deepcopy(new_point)

            file_handler.sa_generate_model_file_with_new_point(current_point, param_list, opt_iteration)

            hyp_test_result, mean, is_process_timed_out = perform_hyp_test()
            opt_iteration += 1

            if is_process_timed_out == True:
                current_point = deepcopy(previous_point)

            #iteration_end_time = time.time() - iteration_start_time
            #print("Total time taken till now(in sec): ", iteration_end_time, "s")
            #print("Total time taken till now(in hrs): ", iteration_end_time/60.0/60.0, "h")

        if hyp_test_result == 1:
            is_parameters_synthesized = True
            #print("Hypothesis test result: ", hyp_test_result, ". H1 accepted")
        elif hyp_test_result == 0:
            is_parameters_synthesized = False
            #print("Hypothesis test result: ", hyp_test_result, ". H1 rejected")
        else:
            is_parameters_synthesized = False
            #print("Hypothesis test result: ", hyp_test_result, ". need more samples...")

        current_point.mean = mean


        if is_parameters_synthesized == False:
            if (current_point.mean >= previous_point.mean):
                #print("keeping current point")
                pass      
            else:
                escape_probability = np.exp((current_point.mean - previous_point.mean)/current_temperature)
                rand_num = random.uniform(0,1)

                # move to the new point with some probability
                if rand_num < escape_probability:
                    #print("using current point even though previous point had the larger mean")
                    pass
                else:
                    #print("using previous point because it had the larger mean")
                    current_point = deepcopy(previous_point)

            current_temperature *= COOLING_RATE
            #print("-----------------------------------------------------------------")
            #print("Current Temperature: ", current_temperature)
            #iteration_end_time = time.time() - iteration_start_time
            #print("Total time taken till now(in sec): ", iteration_end_time, "s")
            #print("Total time taken till now(in hrs): ", iteration_end_time/60.0/60.0, "h")

    output_file = file_handler.create_output_file() 
    iteration_end_time = time.time() - iteration_start_time
    print("\n\n---------------------------Process number: "+str(process_number)+"--------------------------------------")
    print("Total time taken (in sec): ", iteration_end_time, "s")
    print("Total time taken (in hrs): ", iteration_end_time/60.0/60.0, "h")

    if is_parameters_synthesized == True:
        #print("-----------------------------------------------------------------")
        #print("Estimated Parameter Values:")
        print("Successful in estimating parameter values!!!")
        print("Please find estimated set of parameter values inside the \"output/"+ program_time_stamp +"/\" folder")
        print("\n\n")

        file_handler.write_output_file(output_file, "Specification: " + file_handler.spec_file_name + "\n\n")
        file_handler.write_output_file(output_file, "Estimated Parameter Values:\n")

        for i in range(no_of_params):
            #print("Param Name: ", param_list[i].name)
            #print("Param Value: ", current_point.vector[i])

            file_handler.write_output_file(output_file, "Param Name: " + param_list[i].name + "\n")
            file_handler.write_output_file(output_file, "Param Value: " + str(current_point.vector[i]) + "\n")
    else:
        #print("-----------------------------------------------------------------")
        #print("No parameter found.")
        print("No parameter set found!!!")
        print("\n\n")

        file_handler.write_output_file(output_file, "No parameter found.")

    file_handler.close_output_file(output_file)
    
def perform_hyp_test():
    global MAX_TRIALS_FOR_HYPOTHESIS_TESTING

    MAX_TRIALS_FOR_HYPOTHESIS_TESTING = 7   
    
    is_process_timed_out = False
    is_invalid_score = False

    no_of_specs = len(spec_list)

    theta0 = [None] * no_of_specs
    theta1 = [None] * no_of_specs

    alpha_hyp = [None] * no_of_specs
    beta_hyp = [None] * no_of_specs

    A = [None] * no_of_specs
    B = [None] * no_of_specs

    d1 = [None] * no_of_specs
    d2 = [None] * no_of_specs
    sum = [None] * no_of_specs
    mean = [None] * no_of_specs

    spec_probability = [None] * no_of_specs
    no_of_successful_samples = [None] * no_of_specs
    is_spec_prob_greater_than_conf = [None] * no_of_specs

    hyp_test_result = [None] * no_of_specs
    is_stopping_boundary_crossed = [None] * no_of_specs
    samples = [[] for i in range(no_of_specs)]

    idx = 0
    step_down_idx = 0
    is_useful_iteration = False
    epsilon = 0.01

    for i in range(no_of_specs):
        d1[i] = 0
        d2[i] = 0
        sum[i] = 0

        no_of_successful_samples[i] = 0

        s = i + 1

        alpha_hyp[i] = ((no_of_specs - s + 1 - alpha) * beta) / ((no_of_specs - s + 1) * (no_of_specs - alpha))
        beta_hyp[i] = ((no_of_specs - s + 1 - beta) * alpha) / ((no_of_specs - s + 1) * (no_of_specs - beta))

        A[i] = np.log(((1 - beta_hyp[i]) * (no_of_specs - s + 1)) / alpha)
        B[i] = np.log(beta / ((1 - alpha_hyp[i]) * (no_of_specs - s + 1)))     

        is_stopping_boundary_crossed[i] = False

    while idx <= MAX_TRIALS_FOR_HYPOTHESIS_TESTING and is_process_timed_out == False and is_invalid_score == False:
        idx = idx + 1

        s, is_process_timed_out = run_bionetgen_simulation_and_verify_trace_with_telex(idx)
        for i in range(no_of_specs):

            if s[i] > 0.0:
                no_of_successful_samples[i] += 1                         

            samples[i].append(s[i])
            sum[i] += s[i]
            mean[i] = sum[i]/idx

            theta0[i] = 0 - delta
            theta1[i] = 0 + delta

            #print("theta0: ", theta0[i])
            #print("theta1: ", theta1[i])

            if is_process_timed_out:
                is_process_timed_out = True
                return hyp_test_result, mean, is_process_timed_out

            if np.isnan(s[i]) or np.isinf(s[i]):
                is_invalid_score = True
                return hyp_test_result, mean, is_invalid_score

            if idx > 3 and is_stopping_boundary_crossed[i] == False:
                d = 0
                for j in range(idx):
                    d = d + np.power((samples[i][j] - mean[i]), 2)
                    
                var = d/(idx-1)

                d1[i] = d1[i] + np.power((s[i] - theta1[i]), 2)
                d2[i] = d2[i] + np.power((s[i] - theta0[i]), 2)        

                f1 = d1[i] * (-1 * 1/(2*(var+epsilon)))
                f2 = d2[i] * (-1 * 1/(2*(var+epsilon)))
                f = np.exp(f1 - f2)
                
                z = np.log(f)
                #print("z: ", z)

                if step_down_idx == i:                
                    if z >= A[i]: #test accepts null hypothesis H0, rejects H1
                        hyp_test_result[i] = 1
                        is_stopping_boundary_crossed[i] = True
                        step_down_idx = step_down_idx + 1
                    elif z <= B[i]: #test rejects null hypothesis H0, accepts H1
                        hyp_test_result[i] = 0
                        is_stopping_boundary_crossed[i] = True
                        step_down_idx = step_down_idx + 1
                    else: #need more samples to arrive at some decision(accept/reject)
                        hyp_test_result[i] = -1
                        is_stopping_boundary_crossed[i] = False

            if np.min(s) >= 0 and is_useful_iteration == False:
                is_useful_iteration = True
                MAX_TRIALS_FOR_HYPOTHESIS_TESTING = 1000

                
        if all(is_stopping_boundary_crossed): #all True
            for i in range(no_of_specs):
                spec_probability[i] = no_of_successful_samples[i]/idx
                #print("Spec ", i, "Probability: ", spec_probability[i])
                if spec_probability[i] > confidence[i]:
                    is_spec_prob_greater_than_conf[i] = True
            break

    final_hyp_result = -1
    if hyp_test_result.count(1) == no_of_specs and all(is_spec_prob_greater_than_conf):
        final_hyp_result = 1
    elif hyp_test_result.count(-1) == no_of_specs:
        final_hyp_result = -1
    else:
        final_hyp_result = 0

    final_mean = np.mean([i for i in mean if i < 0]) #mean of only -ive(False) values

    # Printing results
    #print("\n\n------------------")
    #print("Results")
    #print("------------------")
    #print("quant score: ", quant_score_sum)
    #print("number of trials: ", idx)
    #print("number of successful trials: ", m)
    #print("mean: ", mean)
    #print("final mean: ", final_mean)

    return final_hyp_result, final_mean, is_process_timed_out


def run_bionetgen_simulation_and_verify_trace_with_telex(idx, point_num = 0):  
    time_out = BIONETGEN_SIMULATION_TIMEOUT
    is_process_timed_out = False
    no_of_specs = len(spec_list)
    #print("-----------------------------------------------------------------")
    #print("Simulation no. ", idx, "starts...")
    start_time = time.time()
    if point_num == 0:
        model_file_name_final = model_file_name + "_" + str(opt_iteration)
        #print(model_file_name_final)
        process = subprocess.Popen(["./simulate.sh", model_file_name_final, program_time_stamp], stdout=open(os.devnull, 'wb'))
    else:
        model_file_name_final = model_file_name + "_" + str(opt_iteration) + "_" + str(point_num)
        #print(model_file_name_final)
        process = subprocess.Popen(["./simulate.sh", model_file_name_final, program_time_stamp], stdout=open(os.devnull, 'wb'))

    #keep polling the process to check how much time does it take
    while process.poll() is None and time_out > 0:
        time.sleep(1)
        time_out -= 1
            
    #if simulations takes more than the time, kill that simulation and treat it as a failed/unsuccessful simulation
    if not time_out > 0:
        is_process_timed_out = True
        os.system('pkill -TERM -P ' + str(process.pid))
        #print("process timeout: killed")
        return [-1], is_process_timed_out #returning a negative value so that hyp test counts this as a unsuccessful simulation

    #print("Simulation no. ", idx, "completes...")
    #print("Total Simulation Time: ", time.time() - start_time, "seconds\n")
    
    telex_result= []
    for i in range(no_of_specs):
        telex_result.append(rc.main(trace_file_name, program_time_stamp, spec_list[i]))
    return telex_result, is_process_timed_out

def select_a_neighbouring_point(previous_point, param_list, no_of_params):
    while True:
        try:
            new_point = deepcopy(previous_point)

            index_to_perturb =  random.randint(0, no_of_params-1)

            if new_point.vector[index_to_perturb] == param_list[index_to_perturb].min_val:
                high_or_low = 1
            elif new_point.vector[index_to_perturb] == param_list[index_to_perturb].max_val:
                high_or_low = 0
            else:
                high_or_low = random.randint(0, 1)

            if high_or_low == 1:
                new_point.vector[index_to_perturb] += PARAM_PERTURBATION_FACTOR * (param_list[index_to_perturb].max_val - previous_point.vector[index_to_perturb])
            else:
                new_point.vector[index_to_perturb] -= PARAM_PERTURBATION_FACTOR * (previous_point.vector[index_to_perturb] - param_list[index_to_perturb].min_val)

            assert ((new_point.vector[index_to_perturb] <= param_list[index_to_perturb].max_val) 
                and (new_point.vector[index_to_perturb] >= param_list[index_to_perturb].min_val))
        except AssertionError:
            #print("neighbouring point out of parameter defined min-max range. Trying again.")
            pass
        else:
            break
    
    return new_point  

def terminateProcess(signalNumber, frame):
    print ('Process terminated.')
    file_handler.remove_dir_for_prog_run()
    sys.exit()       

if __name__ == "__main__":
    main(sys.argv[1:])

