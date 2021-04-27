# CM50265-Group-Project

Download the dataset from:
    https://drive.google.com/file/d/1yXJfHPddenqwtZS1G5DT08JMj40IA8vF/view?usp=sharing
and decompress in the root folder as it is (CM50265-Group-Project/dataset). 

Alternatively, you can download the haze removed dataset from: 
    https://drive.google.com/file/d/10yIBQSgl1Ygm62IHiWutjKpFctWUe_Sg/view?usp=sharing
and set DATASET_PATH to "dataset-haze-removed/" in the notebook.


**NOTE:** Google Drive does not allow you (theoretically it does, but unnecessarily difficult!) to directly download a file, which means we cannot download a dataset directly to a server. In this case, download the dataset on your local machine and use scp to transfer. For example, ```scp dataset.tar.gz username@garlick.cs.bath.ac.uk:~/CM50265-Group-Project/``` and decompress it there using command such as ```tar -xzvf dataset.tar.gz```
