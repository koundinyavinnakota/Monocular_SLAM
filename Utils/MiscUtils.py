import numpy as np
import tempfile
import os


# ftmp = tempfile.NamedTemporaryFile(delete=False)
# fname = ftmp.name + ".npy"
# inp = np.random.rand(100)
# np.save(fname, inp, allow_pickle=False)


# out = np.load(fname, allow_pickle=False)

# print(out)

# os.remove(fname)

def save_np_array(arr, save_path):
    with open(save_path, 'wb') as f:
        np.save(f, arr)

# save_np_array(np.array([1, 2, 3, "you go free"]))
# with open('test.npy', 'rb') as f:
#     arr = np.load(f)
# print(arr)
    
    
# def save_np_array(arrayX):
#     ftmp = tempfile.NamedTemporaryFile(delete=False)
#     fname = ftmp.name + ".npy"
#     np.save(fname, arrayX, allow_pickle=False)
#     return fname
    