import os
import shutil

N_MALE = 2777
N_FEMALE = 4976

# for creating the male training set
source_male_dir = 'G:/11k_hands/hands/male_hands'       ## Source directory
destination_male_dir = 'G:/11k_hands/dataset/train/male'        ## Destination directory   

for filename in os.listdir(source_male_dir):
    if N_MALE>=0:
        src_file = os.path.join(source_male_dir, filename)
        dst_file = os.path.join(destination_male_dir, filename)
        N_MALE -=1
        
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)
    
    if N_MALE == 0:
        break
        
# for creating the female training set
source_female_dir = 'G:/11k_hands/hands/female_hands'       ## Source directory
destination_female_dir = 'G:/11k_hands/dataset/train/female' ## Destination directory

for filename in os.listdir(source_female_dir):
    if N_FEMALE>=0:
        src_file = os.path.join(source_female_dir, filename)
        dst_file = os.path.join(destination_female_dir, filename)
        N_FEMALE -=1
        
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)
    
    if N_FEMALE == 0:
        break
    
TEST_MALE = 1190
TEST_FEMALE = 2133

# for creating the male testing set
source_male_dir = 'G:/11k_hands/hands/male_hands'
destination_male_dir = 'G:/11k_hands/dataset/test/male'

for filename in os.listdir(source_male_dir):
    if TEST_MALE>=0:
        src_file = os.path.join(source_male_dir, filename)
        dst_file = os.path.join(destination_male_dir, filename)
        TEST_MALE -=1
        
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)
    
    if TEST_MALE == 0:
        break

# for creating the female testing set
source_female_dir = 'G:/11k_hands/hands/female_hands'
destination_female_dir = 'G:/11k_hands/dataset/test/female'

for filename in os.listdir(source_female_dir):
    if TEST_FEMALE>=0:
        src_file = os.path.join(source_female_dir, filename)
        dst_file = os.path.join(destination_female_dir, filename)
        TEST_FEMALE -=1
        
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)
    
    if TEST_FEMALE == 0:
        break