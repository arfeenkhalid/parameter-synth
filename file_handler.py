import numpy as np
import os

class FileHandler:
    def __init__(self, program_time_stamp, model_file_name, confidence, spec_file_name="", no_of_params=0):
        self.program_time_stamp = program_time_stamp
        self.model_file_name = model_file_name
        self.confidence = confidence
        self.spec_file_name = spec_file_name
        self.no_of_params = no_of_params

    def read_spec_file(self):
        spec_file = {}
        for line in open('specs/' + self.spec_file_name + '.txt'):
            if line[0] != "%":
                name, val = line.partition("=")[::2]
                spec_file[name.strip()] = val.strip()
        return spec_file

    def sa_generate_model_file_with_new_point(self, point, param_list, iteration):
        new_path = 'models/' + self.program_time_stamp + '/'
        model_file_contents = ""
        for line in open('models/' + self.model_file_name + '.bngl'):
            model_file_contents += line
            if line.startswith("generate_network"):
                for i in range(self.no_of_params):
                    model_file_contents += "setParameter(\"" + param_list[i].name + "\", " + str(point.vector[i]) + ");\n"
 
        new_model_file = open(new_path + self.model_file_name + '_' + str(iteration) + '.bngl', 'w')
        new_model_file.write(model_file_contents)
        new_model_file.close()

    def create_dir_for_prog_run(self):
        model_dir = os.path.join("models/", self.program_time_stamp)
        os.makedirs(model_dir)

        output_dir = os.path.join("output/", self.program_time_stamp)
        os.makedirs(output_dir)
        
    def create_output_file(self):
        output_file = open("output/" + self.program_time_stamp + "/estimated_parameters.txt", 'a')
        return output_file
    
    def write_output_file(self, output_file, message_text):
        output_file.write(message_text)

    def close_output_file(self, output_file):
        output_file.close()
