from __future__ import annotations

import mteb

print("mteb imported.")

model_name = "laion/clap-htsat-fused"
model = mteb.get_model(model_name=model_name)
print("model loaded..")
tasks = mteb.get_tasks(tasks=["ESC50_Zeroshot"])
print("task loaded..")

print(tasks)

evaluation = mteb.MTEB(tasks=tasks)

results = evaluation.run(model)
print("eval complete.")

print(results[0].scores)

#
#

# from datasets import load_dataset

# d = load_dataset("ashraq/esc50", split="train")

# # # print(d[0]['target'], d[0]['category'])

# # l = []
# # l_id = []
# dic = {}

# for i in d:
#     dic[i['target']] = i['category']
#     # l.append(i['category'])
#     # l_id.append(i['target'])

# sorted_values = [dic[key] for key in sorted(dic.keys())]

# # Print the values in list format
# print(sorted_values)


# # ac_list = list(set(l))
# # ac_list_id = list(set(l_id))
# # print(len(ac_list), len(ac_list_id))

# # # print(d[1])


# l = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw']


# l = ["Sound of " + j for j in l]

# print(l)
