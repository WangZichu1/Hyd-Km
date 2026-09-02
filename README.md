# Hyd-Km
The Hyd-Km model, and code
The code is stored here(including the packed full code). And the model weights store at zenodo(https://doi.org/10.5281/zenodo.22233616), the model wetights for mutant testset were provided here in the "mutant_prediction" folder.

################################################################################################Data

raw data and the prepared dataset are provided at (https://doi.org/10.5281/zenodo.19853620).

For convenience， we collected the generated embeddings, entry ids, logKM values, and saved them as ".pth" dataset files. You can directly load them for model test. 

The dimension-reduced trainnig dataset(fulldata_train, fulldata_test) for Hyd-KmDR is provided here, and can be loaded for a easy model test. 

Datasets with originnal dimensional embeddings were stored at zenodo(https://doi.org/10.5281/zenodo.22233616: fulldata(10f).pth refers to the whole 8385 entris, traindataV2.pth and testdataV2,pth contain originnal dimensional embeddings named train(test)_fs(p)feats). If you want to test Ori, WO and CM models, please load V2 or V3 version data.

The mutant testset can be found here and directly used.

If you want to start from the pre-trained models, you need to:

1. Deploy these models(ESM2,MolT5,Superwater) and weights locally (pre-trained model weights are also in https://doi.org/10.5281/zenodo.22233616). You can use tools provided here or download
(ESM2:https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt("facebook/esm2_t33_650M_UR50D")
MolT5:https://github.com/blender-nlp/MolT5
Super water：https://github.com/kuangxh9/SuperWater)

2.Protein sequences, SMILES, and structure files need to be prepared to generate embeddings.

3.Once the embeddings are obtained, you can use the tools provided here(data_processing&utils) to construct the training and test sets.

################################################################################################Fusion&control&baselines

The dataset loading module, the model module, and the training and inference modules are decoupled into independent, callable components.

The weights and scalers for models in our research are stored at zenodo(https://doi.org/10.5281/zenodo.22233616) including final models and 10-fold validations.

Load the prepared datasets, weights and scalers to run the fusion models.
