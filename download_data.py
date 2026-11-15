"""Fetch the public Keras MNIST archive, enforcing the upstream SHA256."""
import hashlib
import urllib.request
from pathlib import Path

URL='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
EXPECTED='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
DEST=Path(__file__).resolve().parent/'data/mnist.npz'

def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists():
        if hashlib.sha256(DEST.read_bytes()).hexdigest()==EXPECTED:
            print('MNIST archive already present; upstream SHA256 verified.');return
        raise RuntimeError('Existing mnist.npz has an unexpected hash; preserve/investigate it before replacing.')
    temp=DEST.with_suffix('.download')
    urllib.request.urlretrieve(URL,temp)
    actual=hashlib.sha256(temp.read_bytes()).hexdigest()
    if actual!=EXPECTED:raise RuntimeError(f'Archive SHA256 mismatch: {actual}; incomplete file retained at {temp}')
    temp.replace(DEST)
    print('MNIST archive downloaded; upstream SHA256 verified.')

if __name__=='__main__':main()
