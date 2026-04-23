# Overly deep hierarchical mentalizing produces paranoia: a new formal theory

## Intorduction
This accompanying code for ["Overly deep hierarchical mentalizing produces paranoia: a new formal theory"](https://osf.io/preprints/psyarxiv/szj5n)
paper by [Nitay Alon](https://nitayalon.github.io/), [Lion Schulz](https://sites.google.com/view/lionschulz/home),
[Vaughan Bell](https://vaughanbell.net/), [Michael Moutoussis](https://profiles.ucl.ac.uk/36080-michael-moutoussis),
[Peter Dayan](https://www.mpg.de/12309370/biological-cybernetics-dayan) and [Joseph Barnby](https://joebarnby.com/)

The paper proposes a computational model, binding Theory-of-Mind, mentalization and
Paranoid behaviour.

## Usage

To run the code clone the repository and run the following:

```
python3 main.py 
```

This function takes multiple arguments as input:
```
--environment (str): set to aleph_idomdp
 
--seed: (int) for reproducibility

--softmax_temp: (float) set the SoftMax temperature (default 0.01)

--sender_tom (str) Either DoM-1 or DoM1

--receiver_tom (str) Either DoM0 or DoM2
```

To edit the Sender's and Receiver's threshold edit the following entries on `congif.yml`:
```
sender_thresholds: [0.0,0.1,0.5]
receiver_thresholds: [0.0]
```
To expedite search set `use_memoization: True`

## Citation

If you refer to our work, please cite it as:
```
@article{alon2024overly,
  title={Overly deep hierarchical mentalizing produces paranoia: a new formal theory},
  author={Alon, Nitay and Schulz, Lion and Bell, Vaughan and Moutoussis, Michael and Dayan, Peter and Barnby, Joseph M},
  year={2024},
  publisher={OSF}
}
```

