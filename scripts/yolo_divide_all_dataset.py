# Import the `random` module to shuffle the lines
import random
import sys
import os

all_data_path_txt = sys.argv[1]
num_class = int(input("number of classes:"))
proportion_train = float(input("training data proportion:"))
proportion_val = float(input("val data proportion:"))
if(proportion_train + proportion_val > 1):
    print("proportion_train + proportion_val is bigger than 1!")
    exit()
proportion_test = 1 - proportion_train - proportion_val
print("testing data proportion is" + str(proportion_test))

# Open the input file in read mode
with open(all_data_path_txt, 'r') as input_file:
  # Read all the lines from the input file
  lines = input_file.readlines()

label_not_exists_num = 0
image_not_exists_num = 0
label_wrong_num = 0
valid_paths = []
for line in lines:
    if not os.path.exists(line.strip()):
        label_not_exists_num += 1
        print(line.strip() + ' not exists')
    elif not os.path.exists(line.strip()[:-3] + 'jpg'):
        image_not_exists_num += 1
        print(line.strip()[:-3] + 'jpg not exists')
    else:
        with open(line.strip(), 'r') as label_file:
            boxes = label_file.readlines()
            wrong_box_found = False
            for box in boxes:
                if int(box[0]) > (num_class - 1):
                    label_wrong_num += 1
                    wrong_box_found = True
                    print(line + 'label wrong')
            if not wrong_box_found:
                valid_paths.append(line)

print("all files number:" + str(len(lines)))
print("label file not exists number:" + str(label_not_exists_num))
print("image file not exists number:" + str(image_not_exists_num))
print("label class wrong number:" + str(label_wrong_num))
print("valid files number:" + str(len(valid_paths)))

# Shuffle the lines randomly
random.shuffle(valid_paths)

# Calculate the number of lines to include in the first output file
num_line_train_file = int(len(valid_paths) * proportion_train)
num_line_test_file = int(len(valid_paths) * proportion_test)
num_line_val_file = int(len(valid_paths) * proportion_val)

# Open the first output file in write mode
with open('all_train_data.txt', 'w') as output_file_1:
  # Write the first `num_lines_first_file` lines to the first output file
  # Replace the last three characters of each line from "txt" to "jpg"
  for i in range(num_line_train_file):
    output_file_1.write(valid_paths[i].strip()[:-3] + 'jpg\n')

# Open the second output file in write mode
with open('all_test_data.txt', 'w') as output_file_2:
  # Write the remaining lines to the second output file
  # Replace the last three characters of each line from "txt" to "jpg"
  for i in range(num_line_train_file, num_line_train_file + num_line_test_file):
    output_file_2.write(valid_paths[i].strip()[:-3] + 'jpg\n')

with open('all_val_data.txt', 'w') as output_file_3:
    for i in range(num_line_train_file + num_line_test_file, len(valid_paths)):
        output_file_3.write(valid_paths[i].strip()[:-3] + 'jpg\n')
