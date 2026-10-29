import numpy as np
from analyse_final50k import holm, boot_mean

def test_holm_nine_and_all_ties_and_bootstrap_shape():
    assert holm([.01]*9) == [.09]*9
    assert holm([0.0]*9) == [0.0]*9
    rng=np.random.default_rng(4)
    assert boot_mean([0,0,0],rng,200) == [0.0,0.0]

if __name__ == "__main__":
    test_holm_nine_and_all_ties_and_bootstrap_shape()
    print("Holm and all-ties bootstrap regression passed")
