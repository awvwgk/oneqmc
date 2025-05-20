# Model Card for Orbformer

## Model Details

### Model Description

Orbformer is a model developed by the OneQMC team. It is a chemically transferable wave function model that is pretrained on a region of chemical space using Variational Monte Carlo. It can provide extremely high accuracy energy and electron density data directly from the equations of physics and does not require labelled data for any of its training. Users of this model are expected to have reasonable experience in the field of quantum chemistry.

- **Developed by:** OneQMC Team (Adam Foster, Zeno Schätzle, P Bernát Szabó, Lixue Cheng, Jonas Köhler, Gino Cassella, Nicholas Gao, Jiawei Li, Jan Hermann, Frank Noé), Microsoft Research AI for Science  
- **Model type:** Neural Network Wave Function
- **Language(s)**: Python, JAX
- **License:** MIT

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/microsoft/oneqmc
- **Papers:** Orbformer wave function foundation model, [Electron density extraction](https://arxiv.org/pdf/2409.01306)

## Uses

### Direct intended uses

1.	Fine-tuning the published checkpoints on specific geometries, reactions, etc
2.	Evaluating the energy from OneQMC checkpoints
3.	Obtaining electron densities from OneQMC checkpoints
4.	Pretraining own models from scratch
5.	OneQMC is being shared with the research community to facilitate reproduction of our results and foster further research in this area.  



### Out-of-Scope Use

1.	Zero-shot evaluation of published checkpoints is not expected to yield sufficiently accurate results, we recommend at least minimal fine-tuning in all cases.
2.	We do not provide pretrained wave function checkpoints for individual molecules or geometries.
3.	We do not recommend using OneQMC in commercial or real-world applications without further testing and development. It is being released for research purposes.



## Risks and Limitations

1.	Memory usage will be high for large molecules (above 100 electrons), and multiple GPUs should be used for fine-tuning on large structures.
2.	Pretrained checkpoints support the following atom types: H, Li, B, C, N, O, F. Molecules involving other atoms types are likely to suffer from worse performance (He, Be) or require training from scratch (Ne and heavier).
3.	Interpretation of results requires expertise in quantum chemistry.


### Recommendations

1.	When facing memory scaling issues, we recommend reducing the electron batch size, using more GPUs in parallel and exploring MCMC sampling options to achieve good fine-tuning performance
2.	For other atom types, we recommend training from scratch


## How to Get Started with the Model

See the [README](./README.md).

## Training Details

### Training Data

The code to generate the training structures that were used to create the checkpoints can be found [here](./notebooks/paper/pretraining/generate_light_atom_curriculum.ipynb). Some training data was derived from NIST experimental geometries.

### Training Procedure

#### Preprocessing

Preprocessing was done at the same time as training data generation.

#### Training Hyperparameters

- All hyperparameters that differ from the default values are captured in the following command line instructions
- **Phase 1a**: `python scripts/transferable.py -d lightatomcurriculum/level1 --data-file-whitelist '.*(bend|stretch)\.(json|yaml)' --data-augmentation rotation fuzz -a orbformer-se --electron-batch-size 1024 --mol-batch-size 8 -n 500000 --max-restarts 20 --multi-system-sampler double-langevin --repeated-sampling-len 20 --max-eq-steps 300 --chkpts-fast-interval 101 --metric-logger-period 25`
- **Phase 1b**:  `python scripts/transferable.py -d lightatomcurriculum/level1 --data-file-whitelist '.*(bend|stretch|break)\.(json|yaml)' --data-augmentation rotation fuzz -a orbformer-se -c chkpt-1a.pt --discard-sampler-state --electron-batch-size 1024 --mol-batch-size 4 -n 500000  --max-restarts 100  --multi-system-sampler double-langevin --repeated-sampling-len 20 --max-eq-steps 300 --chkpts-fast-interval 101 --metric-logger-period 25`
- **Phase 2**: `python scripts/transferable.py -d lightatomcurriculum/level2 --data-augmentation rotation fuzz -a orbformer-se -c chkpt-1b.pt --discard-sampler-state --electron-batch-size 1024 --mol-batch-size 16 -n 1000000 --max-restarts 200 --multi-system-sampler double-langevin --repeated-sampling-len 40 --max-eq-steps 300 --metric-logger-period 25`

#### Speeds, sizes and timings

Pretraining times

| Phase | Data                        | Steps | Mol batch size | Electron batch size | A100 hours (est.) |
|-------|-----------------------------|-------|----------------|---------------------|-------------------|
| 1a    | LAC Level 1 bend and stretch| 200k  | 8              | 1024                | 800               |
| 1b    | LAC Level 1                 | 200k  | 8              | 1024                | 800               |
| 2     | LAC Level 2                 | 400k  | 16             | 1024                | 9600              |

Size: 3032938 parameters

Fine-tune speed will depend very much on the fine-tune dataset.


## Evaluation
### Testing data, factors, and metrics
#### Testing data
We test on 5 main experiments:
1.	Diels-Alder reaction: a collection of MRAQCC-optimized geometries for products, reactants and transition states of a canonical Diels-Alder reaction
2.	Bond-breaking MEPs consisting of organic molecules up to 48 electrons along bond dissociation pathways of 20 points
3.	TinyMol dataset consisting of small organic molecules near equilibrium, plus LiCN bond stretching
4.	Alkanes: alkane chains from 6 to 13 carbon atoms
5.	N2 and ethene: N2 dissociation curve combined with a geometries of ethene

#### Factors
The breakdown into 5 experiments is designed to isolate performance on particular tasks. For example, 1 and 2 focus on multireference performance, 3 focuses on benchmarking against previous work, 4 focuses on scalability and understanding the model, and 5 highlights the low “distractibility” of the model.

#### Metrics
Our primary metric is mean absolute relative energy error (MARE) across a reaction curve. If we have a curve with G geometries with reference R and proposed energies E, we compute the metric as  
```latex
MARE = \frac{1}{G} \sum_{g=1}^G |E_g - \bar{E} - R_g + \bar{R}|
```
### Evaluation results

#### Summary

We find that our innovations reduce the training cost to reach chemical accuracy by over one order of magnitude compared to single-point calculations with the best published neural network ansatz.
We compare against earlier methods for chemically transferable VMC and find that Orbformer is more accurate and benefits more from pretraining.

Switching focus to multi-reference problems, we compare Orbformer to classical quantum chemistry methods including DFT, NEVPT2, MRCI and MRCC on five bond-breaking curves and on transition states of a Diels-Alder reaction.

We show that Orbformer demonstrates an extremely good trade-off between cost and accuracy across these tests. 

Furthermore, the accuracy of Orbformer systematically improves with more fine-tuning time, whereas converging classical methods is not always straightforward.

Finally, we show that Orbformer can be fine-tuned on systems with over 100 electrons and learns features that can be transferred from small to large molecules.

The Orbformer, though by no means the end of the story, marks significant progress towards a model that generalizes the electronic structure of molecules.
Given its favourable cost--accuracy trade-off for multi-reference problems, we argue that Orbformer is a useful tool for quantum chemistry applications today.



## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions were estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA A100 PCIe
- **Hours used:** 4000 pretraining 
- **Cloud Provider:** Azure
- **Compute Region:** various, use francecentral for this calculation
- **Carbon Emitted:** 112kg CO2 equivalent, of which 100% were offset by the cloud provider

## Technical Specifications


### Compute Infrastructure

 - Pretraining Phase 1: 8 A100
 - Pretraining Phase 2: 16 A100
 - Fine-tuning: between 1 and 32 A100 per calculation

## License
MIT


## Citation 

If you use this repository, please consider citing our work.
The Orbformer model, checkpoints, training scheme:
```bibtex
@article{ main paper }
```
Density extraction
```bibtex
@article{cheng2025highly,
  title={Highly accurate real-space electron densities with neural networks},
  author={Cheng, Lixue and Szab{\'o}, P Bern{\'a}t and Sch{\"a}tzle, Zeno and Kooi, Derk P and K{\"o}hler, Jonas and Giesbertz, Klaas JH and No{\'e}, Frank and Hermann, Jan and Gori-Giorgi, Paola and Foster, Adam},
  journal={The Journal of Chemical Physics},
  volume={162},
  number={3},
  year={2025},
  publisher={AIP Publishing}
}
```

## Model Card Contact

Please contact Adam Foster for more information
