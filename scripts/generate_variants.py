import numpy as np
import os
import json

            # "avg_sum": avg_sum,
            # "tcp_height": tcp_height,
            # "tcp_vel": tcp_vel,
            # "tcp_to_obj": tcp_to_obj



for height in range(3):
    height_weight_low = height * 2 / 3 - 1
    height_weight_high = height_weight_low + 2 / 3
    height_weight = np.random.uniform(low=height_weight_low, high=height_weight_high)

    for vel in range(3):
        vel_weight_low = vel * 2 / 3 - 1
        vel_weight_high = vel_weight_low + 2 / 3
        vel_weight = np.random.uniform(low=vel_weight_low, high=vel_weight_high)

        for distance_to_obj in range(3):
            distance_to_obj_weight_low = distance_to_obj * 2 / 3 - 1
            distance_to_obj_weight_high = distance_to_obj_weight_low + 2 / 3
            distance_to_obj_weight = np.random.uniform(low=distance_to_obj_weight_low,
                                                            high=distance_to_obj_weight_high)
            
            for avg_sum in range(3):
                avgsum_weight_low = avg_sum * 2 / 3 - 1
                avgsum_weight_high = avgsum_weight_low + 2 / 3
                avgsum_weight = np.random.uniform(low=avgsum_weight_low, high=avgsum_weight_high)

                # get directory of this file
                cur_dir = os.path.dirname(os.path.abspath(__file__))

                template_path = os.path.abspath(os.path.join(cur_dir, "..", "training_configs"))
                filename = "template.json"

                # Opening JSON file
                with open(os.path.join(template_path, filename), 'r') as openfile:
                    template = json.load(openfile)

                template["eval_environment_kwargs"]["weights"] = [ height_weight, 
                                                                    vel_weight, 
                                                                    distance_to_obj_weight, 
                                                                    avgsum_weight ]
                template["expl_environment_kwargs"]["weights"] = [ height_weight, 
                                                                    vel_weight, 
                                                                    distance_to_obj_weight, 
                                                                    avgsum_weight ]

                output_path = os.path.abspath(os.path.join(cur_dir, "..", "training_configs/sawyer_button_press_v2/variants", "{}-{}-{}-{}-{}/".format(height, vel, height, distance_to_obj, avg_sum))) 

                filename = "variant.json"
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                with open(os.path.join(output_path, filename), 'w') as temp_file:
                    json.dump(template, temp_file, indent=4)
