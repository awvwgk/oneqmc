This folder contains the structures used in this study.
 - [LAC dataset](lightatomcurriculum): The Light Atom Curriculum (LAC) dataset proposed in this study aims to be able to deal with both single- and multi-reference organic and inorganic chemistry for the second row elements. The detailed geometry generation protocol and species in each subset are included in the main text and supporting information of the paper. The LAC dataset contains the following subfolders：
   - `level1`
   - `level2` 
 Each subset contains different species in it, and for different species, their corresponding structures are saved in one json file generated using different operations as described in the main text.

 - [BBMEP](bbmep): Inspired by the [BSE49 dataset](https://github.com/aoterodelaroza/bse49) introduced in [Prasad, V. K., Khalilian, M. H., Otero-de-la-Roza, A., & DiLabio, G. A. (2021). BSE49, a diverse, high-quality benchmark dataset of separation energies of chemical bonds. Scientific data, 8(1), 300.](https://doi.org/10.1038/s41597-021-01088-2), we created five bond dissociation minimum energy path (MEP) for `AB --> A + B` as described in the main paper using AutoNEB. Each subfolder contains 20 images on the MEP with its image ID labeled.  

 - TinyMol: [Scherbela, M., Gerard, L., & Grohs, P. (2024). Towards a transferable fermionic neural wavefunction for molecules. Nature Communications, 15(1), 120.](https://doi.org/10.1038/s41467-023-44216-9) [TinyMol dataset](https://github.com/mdsunivie/deeperwin/tree/master/datasets/db). The pretraining and test datasets can be downloaded and processed into our format by running the script `scripts/download_tinymol_dataset.py`. This will create datasets with the names `TinyMol_CNO_rot_dist_test_in_distribution_30geoms`, `TinyMol_CNO_rot_dist_test_out_of_distribution_40geoms`,`TinyMol_CNO_rot_dist_train_18compounds_360geoms`.
    
 - [LiCN](licn): We generated 5 LiCN structures with bond stretching to test the performances of the pretained models.
    
 - [Alkanes](alkane_scalability): To study the scalability of Orbformer, we further test the alkane chains $C_{n}H_{(2n+2)}$ (n=3,4,...,14). `chains.json` contains all the alkane structures with the same repeating -$CH_{2}$- unit using QCElemental format

- Ablation studies used the [CH4](CH4) and [benzene](benzene) molecules in equilibrium geometry, which are both included as datasets.
