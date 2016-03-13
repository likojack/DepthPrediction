import h5py
def write_h5(filename, dataname, X):
    with h5py.File(filename,mode='a') as h:
        h.create_dataset(dataname,data=X,compression='gzip',compression_opts=1)
        
        