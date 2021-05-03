# CM50265 Group Project: Amazon Forest Multi-label Image Classification

Download the dataset from:

    https://drive.google.com/file/d/1KbOIwXffxGr6doi76OANAuRpfmezDPWl/view?usp=sharing

and decompress in the root folder as it is (CM50265-Group-Project/dataset). 

Alternatively, you can download the haze removed dataset from: 

    https://drive.google.com/file/d/18Sza7MrRj4YKJSCW-Ydv9VepFG-l33hB/view?usp=sharing

and set DATASET_PATH to "dataset-haze-removed/" in the notebook.


**NOTE:** Google Drive does not allow you (theoretically it does, but unnecessarily difficult!) to directly download a file, which means we cannot download a dataset directly to a server. In this case, download the dataset on your local machine and use scp to transfer. For example, ```scp dataset.tar.gz username@garlick.cs.bath.ac.uk:~/CM50265-Group-Project/``` will transfer a copy of ```dataset.tar.gz``` to ```~/CM50265-Group-Project/``` folder in ```garlick```. Decompress the tar.gz file there using command such as ```tar -xzvf dataset.tar.gz```.


## References

Understanding-the-Amazon-from-Space: https://github.com/aishwarya-pawar/Deep-Learning-Multi-Label-Image-Classification-Amazon-Forest
