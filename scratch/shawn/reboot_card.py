import os
import time

print('-> Turning off board')
os.system('ssh root@10.0.1.4 "clia deactivate board 5"')
    
print('-> Pausing 15 sec after turning off board')
time.sleep(15);

print('-> Turning on board')
os.system('ssh root@10.0.1.4 "clia activate board 5"')

print('-> Pausing 30 sec after turning on board')
time.sleep(30)

