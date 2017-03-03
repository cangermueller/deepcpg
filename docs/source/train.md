
# Model training

This tutorials describes how to train DeepCpG models and tries to answer some of the frequently asked questions related to model training.

## Initialization

We first initialize some variables that we will use throughout the tutorial. `test_mode=1` should be used for testing purposes, which speeds up computations by only using a subset of the data. For real applications, `test_mode=0` should be used.


```bash
function run {
  local cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  eval $cmd
}

test_mode=1
data_dir="../../data"
```

    

## Creating DeepCpG data files

First, we create a toy dataset that we will use for model training. For more details about creating DeepCpG input data please refer to [this tutorial](../basics/index.ipynb).


```bash
dcpg_data="./data"
cmd="dcpg_data.py
    --cpg_profiles $data_dir/cpg/*.tsv
    --dna_files $data_dir/dna/mm10
    --out_dir $dcpg_data
    --cpg_wlen 50
    --dna_wlen 1001
"
if [[ $test_mode -eq 1 ]]; then
    cmd="$cmd
        --chromo 19
        --nb_sample 50000
        "
fi
run $cmd
```

    
    #################################
    dcpg_data.py --cpg_profiles ../../data/cpg/BS27_1_SER.tsv ../../data/cpg/BS27_3_SER.tsv ../../data/cpg/BS27_5_SER.tsv ../../data/cpg/BS27_6_SER.tsv ../../data/cpg/BS27_8_SER.tsv --dna_files ../../data/dna/mm10 --out_dir ./data --cpg_wlen 50 --dna_wlen 1001 --chromo 19 --nb_sample 50000
    #################################
    INFO (2017-02-27 16:38:47,644): Reading single-cell profiles ...
    INFO (2017-02-27 16:39:27,505): 50000 samples
    INFO (2017-02-27 16:39:27,507): --------------------------------------------------------------------------------
    INFO (2017-02-27 16:39:27,507): Chromosome 19 ...
    INFO (2017-02-27 16:39:27,666): 50000 / 50000 (100.0%) sites matched minimum coverage filter
    INFO (2017-02-27 16:39:28,939): Chunk 	1 / 2
    INFO (2017-02-27 16:39:29,010): Extracting DNA sequence windows ...
    INFO (2017-02-27 16:39:34,807): Extracting CpG neighbors ...
    INFO (2017-02-27 16:39:38,282): Chunk 	2 / 2
    INFO (2017-02-27 16:39:38,315): Extracting DNA sequence windows ...
    INFO (2017-02-27 16:39:41,422): Extracting CpG neighbors ...
    INFO (2017-02-27 16:39:43,263): Done!



```bash
ls $dcpg_data
```

    c19_000000-032768.h5 c19_032768-050000.h5


## Splitting data into training, validation, and test set

For comparing different models, it is necessary to train, select hyper-parameters, and test models on distinct data. In holdout validation, the dataset is split into a training set (~60% of the data), validation set (~20% of the data), and test set (~20% of the data). Models are trained on the training set, hyper-parameters selected on the validation set, and the selected models compared on the test set. For example, you could use chromosome 1-5, 7, 9, 11, 13 as training set, chromsome 14-19 as validation set, and chromosome 6, 8, 10, 12, 14 as test set.

DeepCpG allows to easily split the data by glob patterns. For example, `dcpg_train.py ./data/c{1,2,3,4,5,7,9,11,13}_*.h5 --val_files ./data/c{14,15,16,17,18,19}_*.h5` will split the data as described above, use the training set to train the model, and the validation set to measure the generalization performance of the model and determine when to stop the training by early stopping (**see below**). 

If you are not concerend about comparing DeepCpG with other models, you do not need a test set. In this case, you could, for example, leave out chromsome 14-19 for optimizing hyper-parameters, and use the remaining chromosomes for training.

If you are dealing with genome-wide profiling data then the number of CpG sites on few chromosomes is already usually sufficient for training and you might not need all data. For example, chromsome 1, 3, and 5 from *Smallwood et al (2014)* cover already more than 3 million CpG sites. I found about 3 million CpG sites as sufficient for training models without overfitting. To check how many CpG sites are stored in a set of DeepCpG data files, you can use the `dcpg_data_stats.py` command:


```bash
cmd="dcpg_data_stats.py $dcpg_data/*.h5"
run $cmd
```

    
    #################################
    dcpg_data_stats.py ./data/c19_000000-032768.h5 ./data/c19_032768-050000.h5
    #################################
               output  nb_tot  nb_obs  frac_obs      mean       var
    0  cpg/BS27_1_SER   50000   20621   0.41242  0.665972  0.222453
    1  cpg/BS27_3_SER   50000   13488   0.26976  0.573102  0.244656
    2  cpg/BS27_5_SER   50000   25748   0.51496  0.529633  0.249122
    3  cpg/BS27_6_SER   50000   17618   0.35236  0.508117  0.249934
    4  cpg/BS27_8_SER   50000   16998   0.33996  0.661019  0.224073


`nb_tot` is the total number of CpG files in all data files, `nb_obs` the number of CpG sites with observerations per output cell, `frac_obs` the ratio between `nb_obs` and `nb_tot`, `mean` the mean methylation rate, and `var` the variance of the methylation rate.

## Training DeepCpG modules jointly

As described in [Angermueller et al (2017)](http://biorxiv.org/content/early/2017/02/01/055715), DeepCpG consists of a DNA, CpG, and joint module. The DNA module recognizes features in the DNA sequence window that is centered on a target site, the CpG module recognizes features in observed neighboring methylation states of multiple cells, and the joint module integrates features from the DNA and CpG module and predicts the methylation state of all cells.

The easiest way is to train all modules jointly:


```bash
models_dir="./models"
run "mkdir -p $models_dir"

cmd="dcpg_train.py
    $dcpg_data/c19_000000-032768.h5
    --val_files $dcpg_data/c19_032768-050000.h5
    --dna_model CnnL2h128
    --cpg_model RnnL1
    --out_dir $models_dir/joint
    --nb_epoch 1
    --nb_train_sample 1000
    --nb_val_sample 1000
    "
run $cmd
```

    
    #################################
    mkdir -p ./models
    #################################
    
    #################################
    dcpg_train.py ./data/c19_000000-032768.h5 --val_files ./data/c19_032768-050000.h5 --dna_model CnnL2h128 --cpg_model RnnL1 --out_dir ./models/joint --nb_epoch 1 --nb_train_sample 1000 --nb_val_sample 1000
    #################################
    Using TensorFlow backend.
    INFO (2017-02-27 16:41:28,311): Building model ...
    INFO (2017-02-27 16:41:28,313): Building DNA model ...
    Replicate names:
    BS27_1_SER, BS27_3_SER, BS27_5_SER, BS27_6_SER, BS27_8_SER
    
    INFO (2017-02-27 16:41:28,360): Building CpG model ...
    INFO (2017-02-27 16:41:28,837): Joining models ...
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    dna (InputLayer)                 (None, 1001, 4)       0                                            
    ____________________________________________________________________________________________________
    dna/convolution1d_1 (Convolution (None, 991, 128)      5760        dna[0][0]                        
    ____________________________________________________________________________________________________
    dna/activation_1 (Activation)    (None, 991, 128)      0           dna/convolution1d_1[0][0]        
    ____________________________________________________________________________________________________
    dna/maxpooling1d_1 (MaxPooling1D (None, 247, 128)      0           dna/activation_1[0][0]           
    ____________________________________________________________________________________________________
    dna/convolution1d_2 (Convolution (None, 245, 256)      98560       dna/maxpooling1d_1[0][0]         
    ____________________________________________________________________________________________________
    dna/activation_2 (Activation)    (None, 245, 256)      0           dna/convolution1d_2[0][0]        
    ____________________________________________________________________________________________________
    dna/maxpooling1d_2 (MaxPooling1D (None, 122, 256)      0           dna/activation_2[0][0]           
    ____________________________________________________________________________________________________
    cpg/state (InputLayer)           (None, 5, 50)         0                                            
    ____________________________________________________________________________________________________
    cpg/dist (InputLayer)            (None, 5, 50)         0                                            
    ____________________________________________________________________________________________________
    dna/flatten_1 (Flatten)          (None, 31232)         0           dna/maxpooling1d_2[0][0]         
    ____________________________________________________________________________________________________
    cpg/merge_1 (Merge)              (None, 5, 100)        0           cpg/state[0][0]                  
                                                                       cpg/dist[0][0]                   
    ____________________________________________________________________________________________________
    dna/dense_1 (Dense)              (None, 128)           3997824     dna/flatten_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/timedistributed_1 (TimeDistr (None, 5, 256)        25856       cpg/merge_1[0][0]                
    ____________________________________________________________________________________________________
    dna/activation_3 (Activation)    (None, 128)           0           dna/dense_1[0][0]                
    ____________________________________________________________________________________________________
    cpg/bidirectional_1 (Bidirection (None, 512)           787968      cpg/timedistributed_1[0][0]      
    ____________________________________________________________________________________________________
    dna/dropout_1 (Dropout)          (None, 128)           0           dna/activation_3[0][0]           
    ____________________________________________________________________________________________________
    cpg/dropout_2 (Dropout)          (None, 512)           0           cpg/bidirectional_1[0][0]        
    ____________________________________________________________________________________________________
    merge_2 (Merge)                  (None, 640)           0           dna/dropout_1[0][0]              
                                                                       cpg/dropout_2[0][0]              
    ____________________________________________________________________________________________________
    joint/dense_3 (Dense)            (None, 512)           328192      merge_2[0][0]                    
    ____________________________________________________________________________________________________
    joint/activation_5 (Activation)  (None, 512)           0           joint/dense_3[0][0]              
    ____________________________________________________________________________________________________
    joint/dropout_3 (Dropout)        (None, 512)           0           joint/activation_5[0][0]         
    ____________________________________________________________________________________________________
    cpg/BS27_1_SER (Dense)           (None, 1)             513         joint/dropout_3[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_3_SER (Dense)           (None, 1)             513         joint/dropout_3[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_5_SER (Dense)           (None, 1)             513         joint/dropout_3[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_6_SER (Dense)           (None, 1)             513         joint/dropout_3[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_8_SER (Dense)           (None, 1)             513         joint/dropout_3[0][0]            
    ====================================================================================================
    Total params: 5246725
    ____________________________________________________________________________________________________
    INFO (2017-02-27 16:41:28,895): Computing output statistics ...
    Output statistics:
              name | nb_tot | nb_obs | frac_obs | mean |  var
    ---------------------------------------------------------
    cpg/BS27_1_SER |   1000 |    392 |     0.39 | 0.86 | 0.12
    cpg/BS27_3_SER |   1000 |    290 |     0.29 | 0.92 | 0.08
    cpg/BS27_5_SER |   1000 |    434 |     0.43 | 0.70 | 0.21
    cpg/BS27_6_SER |   1000 |    249 |     0.25 | 0.68 | 0.22
    cpg/BS27_8_SER |   1000 |    408 |     0.41 | 0.84 | 0.13
    
    Class weights:
    cpg/BS27_1_SER | cpg/BS27_3_SER | cpg/BS27_5_SER | cpg/BS27_6_SER | cpg/BS27_8_SER
    ----------------------------------------------------------------------------------
            0=0.86 |         0=0.92 |         0=0.70 |         0=0.68 |         0=0.84
            1=0.14 |         1=0.08 |         1=0.30 |         1=0.32 |         1=0.16
    
    INFO (2017-02-27 16:41:29,162): Loading data ...
    INFO (2017-02-27 16:41:29,165): Initializing callbacks ...
    INFO (2017-02-27 16:41:29,165): Training model ...
    
    Training samples: 1000
    Validation samples: 1000
    Epochs: 1
    Learning rate: 0.0001
    ====================================================================================================
    Epoch 1/1
    ====================================================================================================
    done (%) | time |   loss |    acc | cpg/BS27_6_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_1_SER_acc | cpg/BS27_5_SER_acc | cpg/BS27_3_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        12.8 |  0.0 | 7.0370 | 0.3558 |              0.1008 |              0.1785 |              0.0590 |              0.0887 |              0.0764 |             0.3590 |             0.2619 |             0.3750 |             0.4928 |             0.2903
        25.6 |  0.1 | 7.0446 | 0.4031 |              0.1095 |              0.1807 |              0.0621 |              0.1009 |              0.0813 |             0.3772 |             0.2799 |             0.4575 |             0.4964 |             0.4044
        38.4 |  0.1 | 7.0463 | 0.4225 |              0.1179 |              0.1820 |              0.0652 |              0.1042 |              0.0900 |             0.4251 |             0.2956 |             0.5101 |             0.4952 |             0.3863
        51.2 |  0.1 | 7.0181 | 0.4057 |              0.1191 |              0.1811 |              0.0621 |              0.1019 |              0.0902 |             0.4230 |             0.2743 |             0.4691 |             0.4806 |             0.3814
        64.0 |  0.1 | 6.9931 | 0.4095 |              0.1204 |              0.1746 |              0.0657 |              0.1039 |              0.0876 |             0.4514 |             0.2879 |             0.4567 |             0.4692 |             0.3824
        76.8 |  0.2 | 6.9709 | 0.4136 |              0.1184 |              0.1766 |              0.0641 |              0.1072 |              0.0866 |             0.4762 |             0.2997 |             0.4427 |             0.4766 |             0.3726
        89.6 |  0.2 | 6.9441 | 0.4055 |              0.1132 |              0.1764 |              0.0635 |              0.1081 |              0.0877 |             0.4617 |             0.2989 |             0.4287 |             0.4738 |             0.3642
       100.0 |  0.2 | 6.9244 | 0.4086 |              0.1108 |              0.1756 |              0.0646 |              0.1087 |              0.0883 |             0.4657 |             0.3124 |             0.4188 |             0.4710 |             0.3753
    Epoch 00000: val_loss improved from inf to 1.55967, saving model to ./models/joint/model_weights_val.h5
    
     split |   loss |    acc | cpg/BS27_5_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_1_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_5_SER_acc
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     train | 6.9244 | 0.4086 |              0.1756 |              0.0646 |              0.1087 |              0.0883 |              0.1108 |             0.4188 |             0.3753 |             0.4657 |             0.3124 |             0.4710
       val | 1.5597 | 0.1671 |              0.3966 |              0.4077 |              0.3229 |              0.1699 |              0.2625 |             0.1507 |             0.1313 |             0.2468 |             0.0601 |             0.2467
    ====================================================================================================
    
    Training set performance:
      loss |    acc | cpg/BS27_5_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_1_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_5_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    6.9244 | 0.4086 |              0.1756 |              0.0646 |              0.1087 |              0.0883 |              0.1108 |             0.4188 |             0.3753 |             0.4657 |             0.3124 |             0.4710
    
    Validation set performance:
      loss |    acc | cpg/BS27_1_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_5_SER_acc | cpg/BS27_1_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    1.5597 | 0.1671 |              0.3229 |              0.2625 |              0.4077 |              0.3966 |              0.1699 |             0.2467 |             0.1507 |             0.1313 |             0.2468 |             0.0601
    INFO (2017-02-27 16:41:56,971): Done!


`--dna_model`, `--cpg_model`, and `--joint_module` specify the name of DNA, CpG, and joint architecture, respectively, which will be described in **X**.

We used `c19_000000-032768.h5` as training set, `c19_032768-050000.h5` as validation set, and only trained for one epoch on 1000 training and validation samples. In practice, we would train on more data as described above, and also longer, e.g. 30 epochs.

## Training DeepCpG modules separtely

Although it is convenient to train all modules jointly by running only a single command as described above, I suggest to train modules separtely. First, because it allows to train the DNA and CpG module in parallel on separate machines and thereby to reduce training time. Second, it allows to compare how predictive the DNA module is relative to CpG module. If you think the CpG module is already accruate enough alone, you might not need the DNA module. Thirdly, I obtained better results by training the modules separately. However, this might not be true for your paticular dataset.

You can train the CpG module separately by only using the `--cpg_model` argument, but not `--dna_module`:


```bash
cmd="dcpg_train.py
    $dcpg_data/c19_000000-032768.h5
    --val_files $dcpg_data/c19_032768-050000.h5
    --dna_model CnnL2h128
    --out_dir $models_dir/dna
    --nb_epoch 1
    --nb_train_sample 1000
    --nb_val_sample 1000
    "
run $cmd
```

    
    #################################
    dcpg_train.py ./data/c19_000000-032768.h5 --val_files ./data/c19_032768-050000.h5 --dna_model CnnL2h128 --out_dir ./models/dna --nb_epoch 1 --nb_train_sample 1000 --nb_val_sample 1000
    #################################
    Using TensorFlow backend.
    INFO (2017-02-27 16:42:48,594): Building model ...
    INFO (2017-02-27 16:42:48,597): Building DNA model ...
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    dna (InputLayer)                 (None, 1001, 4)       0                                            
    ____________________________________________________________________________________________________
    dna/convolution1d_1 (Convolution (None, 991, 128)      5760        dna[0][0]                        
    ____________________________________________________________________________________________________
    dna/activation_1 (Activation)    (None, 991, 128)      0           dna/convolution1d_1[0][0]        
    ____________________________________________________________________________________________________
    dna/maxpooling1d_1 (MaxPooling1D (None, 247, 128)      0           dna/activation_1[0][0]           
    ____________________________________________________________________________________________________
    dna/convolution1d_2 (Convolution (None, 245, 256)      98560       dna/maxpooling1d_1[0][0]         
    ____________________________________________________________________________________________________
    dna/activation_2 (Activation)    (None, 245, 256)      0           dna/convolution1d_2[0][0]        
    ____________________________________________________________________________________________________
    dna/maxpooling1d_2 (MaxPooling1D (None, 122, 256)      0           dna/activation_2[0][0]           
    ____________________________________________________________________________________________________
    dna/flatten_1 (Flatten)          (None, 31232)         0           dna/maxpooling1d_2[0][0]         
    ____________________________________________________________________________________________________
    dna/dense_1 (Dense)              (None, 128)           3997824     dna/flatten_1[0][0]              
    ____________________________________________________________________________________________________
    dna/activation_3 (Activation)    (None, 128)           0           dna/dense_1[0][0]                
    ____________________________________________________________________________________________________
    dna/dropout_1 (Dropout)          (None, 128)           0           dna/activation_3[0][0]           
    ____________________________________________________________________________________________________
    cpg/BS27_1_SER (Dense)           (None, 1)             129         dna/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_3_SER (Dense)           (None, 1)             129         dna/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_5_SER (Dense)           (None, 1)             129         dna/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_6_SER (Dense)           (None, 1)             129         dna/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_8_SER (Dense)           (None, 1)             129         dna/dropout_1[0][0]              
    ====================================================================================================
    Total params: 4102789
    ____________________________________________________________________________________________________
    INFO (2017-02-27 16:42:48,691): Computing output statistics ...
    Output statistics:
              name | nb_tot | nb_obs | frac_obs | mean |  var
    ---------------------------------------------------------
    cpg/BS27_1_SER |   1000 |    392 |     0.39 | 0.86 | 0.12
    cpg/BS27_3_SER |   1000 |    290 |     0.29 | 0.92 | 0.08
    cpg/BS27_5_SER |   1000 |    434 |     0.43 | 0.70 | 0.21
    cpg/BS27_6_SER |   1000 |    249 |     0.25 | 0.68 | 0.22
    cpg/BS27_8_SER |   1000 |    408 |     0.41 | 0.84 | 0.13
    
    Class weights:
    cpg/BS27_1_SER | cpg/BS27_3_SER | cpg/BS27_5_SER | cpg/BS27_6_SER | cpg/BS27_8_SER
    ----------------------------------------------------------------------------------
            0=0.86 |         0=0.92 |         0=0.70 |         0=0.68 |         0=0.84
            1=0.14 |         1=0.08 |         1=0.30 |         1=0.32 |         1=0.16
    
    INFO (2017-02-27 16:42:48,991): Loading data ...
    INFO (2017-02-27 16:42:48,994): Initializing callbacks ...
    INFO (2017-02-27 16:42:48,994): Training model ...
    
    Training samples: 1000
    Validation samples: 1000
    Epochs: 1
    Learning rate: 0.0001
    ====================================================================================================
    Epoch 1/1
    ====================================================================================================
    done (%) | time |   loss |    acc | cpg/BS27_1_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_5_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_1_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        12.8 |  0.0 | 3.4556 | 0.4980 |              0.1364 |              0.1194 |              0.0894 |              0.1730 |              0.1042 |             0.4203 |             0.5263 |             0.5854 |             0.4286 |             0.5294
        25.6 |  0.1 | 3.4027 | 0.4392 |              0.1117 |              0.1055 |              0.0861 |              0.1758 |              0.1101 |             0.4244 |             0.4774 |             0.5308 |             0.3632 |             0.4001
        38.4 |  0.1 | 3.3939 | 0.4332 |              0.1141 |              0.1017 |              0.0833 |              0.1830 |              0.1180 |             0.4519 |             0.4556 |             0.5171 |             0.3691 |             0.3723
        51.2 |  0.1 | 3.3869 | 0.4385 |              0.1214 |              0.1052 |              0.0826 |              0.1848 |              0.1189 |             0.4730 |             0.4589 |             0.5072 |             0.3749 |             0.3784
        64.0 |  0.1 | 3.3521 | 0.4206 |              0.1206 |              0.0973 |              0.0780 |              0.1844 |              0.1174 |             0.4645 |             0.4242 |             0.5174 |             0.3380 |             0.3590
        76.8 |  0.2 | 3.3189 | 0.4150 |              0.1148 |              0.0930 |              0.0761 |              0.1819 |              0.1180 |             0.4666 |             0.4141 |             0.5274 |             0.3405 |             0.3265
        89.6 |  0.2 | 3.3004 | 0.4196 |              0.1135 |              0.0944 |              0.0771 |              0.1815 |              0.1179 |             0.4745 |             0.4242 |             0.5357 |             0.3342 |             0.3293
       100.0 |  0.2 | 3.2805 | 0.4195 |              0.1132 |              0.0936 |              0.0792 |              0.1785 |              0.1160 |             0.4750 |             0.4247 |             0.5320 |             0.3361 |             0.3297
    Epoch 00000: val_loss improved from inf to 1.26517, saving model to ./models/dna/model_weights_val.h5
    
     split |   loss |    acc | cpg/BS27_5_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_5_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_1_SER_acc
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     train | 3.2805 | 0.4195 |              0.1785 |              0.1132 |              0.0936 |              0.0792 |              0.1160 |             0.4750 |             0.3361 |             0.4247 |             0.5320 |             0.3297
       val | 1.2652 | 0.1642 |              0.3777 |              0.2891 |              0.1717 |              0.2123 |              0.2144 |             0.2467 |             0.0456 |             0.1313 |             0.2468 |             0.1507
    ====================================================================================================
    
    Training set performance:
      loss |    acc | cpg/BS27_5_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_5_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_1_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    3.2805 | 0.4195 |              0.1785 |              0.1132 |              0.0936 |              0.0792 |              0.1160 |             0.4750 |             0.3361 |             0.4247 |             0.5320 |             0.3297
    
    Validation set performance:
      loss |    acc | cpg/BS27_8_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_5_SER_acc | cpg/BS27_1_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    1.2652 | 0.1642 |              0.1717 |              0.2891 |              0.2123 |              0.2144 |              0.3777 |             0.1313 |             0.2468 |             0.0456 |             0.2467 |             0.1507
    INFO (2017-02-27 16:43:08,988): Done!


You can train the DNA module separtely by only using `--dna_model`:


```bash
cmd="dcpg_train.py
    $dcpg_data/c19_000000-032768.h5
    --val_files $dcpg_data/c19_032768-050000.h5
    --cpg_model RnnL1
    --out_dir $models_dir/cpg
    --nb_epoch 1
    --nb_train_sample 1000
    --nb_val_sample 1000
    "
run $cmd
```

    
    #################################
    dcpg_train.py ./data/c19_000000-032768.h5 --val_files ./data/c19_032768-050000.h5 --cpg_model RnnL1 --out_dir ./models/cpg --nb_epoch 1 --nb_train_sample 1000 --nb_val_sample 1000
    #################################
    Using TensorFlow backend.
    INFO (2017-02-27 16:43:34,166): Building model ...
    Replicate names:
    BS27_1_SER, BS27_3_SER, BS27_5_SER, BS27_6_SER, BS27_8_SER
    
    INFO (2017-02-27 16:43:34,171): Building CpG model ...
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    cpg/state (InputLayer)           (None, 5, 50)         0                                            
    ____________________________________________________________________________________________________
    cpg/dist (InputLayer)            (None, 5, 50)         0                                            
    ____________________________________________________________________________________________________
    cpg/merge_1 (Merge)              (None, 5, 100)        0           cpg/state[0][0]                  
                                                                       cpg/dist[0][0]                   
    ____________________________________________________________________________________________________
    cpg/timedistributed_1 (TimeDistr (None, 5, 256)        25856       cpg/merge_1[0][0]                
    ____________________________________________________________________________________________________
    cpg/bidirectional_1 (Bidirection (None, 512)           787968      cpg/timedistributed_1[0][0]      
    ____________________________________________________________________________________________________
    cpg/dropout_1 (Dropout)          (None, 512)           0           cpg/bidirectional_1[0][0]        
    ____________________________________________________________________________________________________
    cpg/BS27_1_SER (Dense)           (None, 1)             513         cpg/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_3_SER (Dense)           (None, 1)             513         cpg/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_5_SER (Dense)           (None, 1)             513         cpg/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_6_SER (Dense)           (None, 1)             513         cpg/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/BS27_8_SER (Dense)           (None, 1)             513         cpg/dropout_1[0][0]              
    ====================================================================================================
    Total params: 816389
    ____________________________________________________________________________________________________
    INFO (2017-02-27 16:43:34,700): Computing output statistics ...
    Output statistics:
              name | nb_tot | nb_obs | frac_obs | mean |  var
    ---------------------------------------------------------
    cpg/BS27_1_SER |   1000 |    392 |     0.39 | 0.86 | 0.12
    cpg/BS27_3_SER |   1000 |    290 |     0.29 | 0.92 | 0.08
    cpg/BS27_5_SER |   1000 |    434 |     0.43 | 0.70 | 0.21
    cpg/BS27_6_SER |   1000 |    249 |     0.25 | 0.68 | 0.22
    cpg/BS27_8_SER |   1000 |    408 |     0.41 | 0.84 | 0.13
    
    Class weights:
    cpg/BS27_1_SER | cpg/BS27_3_SER | cpg/BS27_5_SER | cpg/BS27_6_SER | cpg/BS27_8_SER
    ----------------------------------------------------------------------------------
            0=0.86 |         0=0.92 |         0=0.70 |         0=0.68 |         0=0.84
            1=0.14 |         1=0.08 |         1=0.30 |         1=0.32 |         1=0.16
    
    INFO (2017-02-27 16:43:34,962): Loading data ...
    INFO (2017-02-27 16:43:34,964): Initializing callbacks ...
    INFO (2017-02-27 16:43:34,964): Training model ...
    
    Training samples: 1000
    Validation samples: 1000
    Epochs: 1
    Learning rate: 0.0001
    ====================================================================================================
    Epoch 1/1
    ====================================================================================================
    done (%) | time |   loss |    acc | cpg/BS27_8_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_8_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_1_SER_acc | cpg/BS27_5_SER_acc | cpg/BS27_3_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        12.8 |  0.0 | 2.9620 | 0.5195 |              0.0650 |              0.1097 |              0.1577 |              0.1064 |              0.0626 |             0.2500 |             0.5610 |             0.6102 |             0.6765 |             0.5000
        25.6 |  0.0 | 2.9809 | 0.4636 |              0.0793 |              0.1040 |              0.1628 |              0.1187 |              0.0572 |             0.3087 |             0.4188 |             0.5218 |             0.6337 |             0.4352
        38.4 |  0.0 | 2.9746 | 0.4507 |              0.0807 |              0.1028 |              0.1639 |              0.1126 |              0.0574 |             0.3095 |             0.3968 |             0.5078 |             0.6500 |             0.3892
        51.2 |  0.0 | 2.9976 | 0.4531 |              0.0782 |              0.1075 |              0.1695 |              0.1211 |              0.0657 |             0.3224 |             0.4067 |             0.4686 |             0.6541 |             0.4139
        64.0 |  0.0 | 2.9942 | 0.4477 |              0.0830 |              0.1081 |              0.1648 |              0.1159 |              0.0685 |             0.3234 |             0.4143 |             0.4362 |             0.6251 |             0.4397
        76.8 |  0.0 | 2.9928 | 0.4388 |              0.0830 |              0.1077 |              0.1644 |              0.1150 |              0.0704 |             0.3130 |             0.4244 |             0.4115 |             0.5977 |             0.4476
        89.6 |  0.0 | 2.9759 | 0.4234 |              0.0795 |              0.1070 |              0.1607 |              0.1094 |              0.0686 |             0.3061 |             0.4179 |             0.4021 |             0.5607 |             0.4300
       100.0 |  0.0 | 2.9823 | 0.4290 |              0.0803 |              0.1091 |              0.1644 |              0.1102 |              0.0691 |             0.3230 |             0.4357 |             0.4006 |             0.5470 |             0.4391
    Epoch 00000: val_loss improved from inf to 1.57197, saving model to ./models/cpg/model_weights_val.h5
    
     split |   loss |    acc | cpg/BS27_5_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_5_SER_acc
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     train | 2.9823 | 0.4290 |              0.1644 |              0.0803 |              0.1102 |              0.1091 |              0.0691 |             0.4006 |             0.4391 |             0.4357 |             0.3230 |             0.5470
       val | 1.5720 | 0.1642 |              0.4196 |              0.2141 |              0.2536 |              0.3399 |              0.3447 |             0.1507 |             0.1313 |             0.2468 |             0.0456 |             0.2467
    ====================================================================================================
    
    Training set performance:
      loss |    acc | cpg/BS27_5_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_5_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    2.9823 | 0.4290 |              0.1644 |              0.0803 |              0.1102 |              0.1091 |              0.0691 |             0.4006 |             0.4391 |             0.4357 |             0.3230 |             0.5470
    
    Validation set performance:
      loss |    acc | cpg/BS27_5_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_8_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_1_SER_acc | cpg/BS27_5_SER_acc | cpg/BS27_6_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    1.5720 | 0.1642 |              0.4196 |              0.2536 |              0.2141 |              0.3447 |              0.3399 |             0.0456 |             0.1313 |             0.1507 |             0.2467 |             0.2468
    INFO (2017-02-27 16:43:45,791): Done!


After training the CpG and DNA module, we are joining them by specifying the name of the joint module with `--joint_module`:


```bash
cmd="dcpg_train.py
    $dcpg_data/c19_000000-032768.h5
    --val_files $dcpg_data/c19_032768-050000.h5
    --dna_model $models_dir/dna
    --cpg_model $models_dir/cpg
    --joint_model JointL2h512
    --out_dir $models_dir/dna_cpg
    --nb_epoch 1
    --nb_train_sample 1000
    --nb_val_sample 1000
    "
run $cmd
```

    
    #################################
    dcpg_train.py ./data/c19_000000-032768.h5 --val_files ./data/c19_032768-050000.h5 --dna_model ./models/dna --cpg_model ./models/cpg --joint_model JointL2h512 --out_dir ./models/dna_cpg --nb_epoch 1 --nb_train_sample 1000 --nb_val_sample 1000
    #################################
    Using TensorFlow backend.
    INFO (2017-02-27 16:46:17,628): Building model ...
    INFO (2017-02-27 16:46:17,631): Loading existing DNA model ...
    INFO (2017-02-27 16:46:17,631): Using model files ./models/dna/model.json ./models/dna/model_weights_val.h5
    Replicate names:
    BS27_1_SER, BS27_3_SER, BS27_5_SER, BS27_6_SER, BS27_8_SER
    
    INFO (2017-02-27 16:46:17,936): Loading existing CpG model ...
    INFO (2017-02-27 16:46:17,937): Using model files ./models/cpg/model.json ./models/cpg/model_weights_val.h5
    INFO (2017-02-27 16:46:18,653): Joining models ...
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    dna (InputLayer)                 (None, 1001, 4)       0                                            
    ____________________________________________________________________________________________________
    dna/convolution1d_1 (Convolution (None, 991, 128)      5760        dna[0][0]                        
    ____________________________________________________________________________________________________
    dna/activation_1 (Activation)    (None, 991, 128)      0           dna/convolution1d_1[0][0]        
    ____________________________________________________________________________________________________
    dna/maxpooling1d_1 (MaxPooling1D (None, 247, 128)      0           dna/activation_1[0][0]           
    ____________________________________________________________________________________________________
    dna/convolution1d_2 (Convolution (None, 245, 256)      98560       dna/maxpooling1d_1[0][0]         
    ____________________________________________________________________________________________________
    dna/activation_2 (Activation)    (None, 245, 256)      0           dna/convolution1d_2[0][0]        
    ____________________________________________________________________________________________________
    dna/maxpooling1d_2 (MaxPooling1D (None, 122, 256)      0           dna/activation_2[0][0]           
    ____________________________________________________________________________________________________
    cpg/state (InputLayer)           (None, 5, 50)         0                                            
    ____________________________________________________________________________________________________
    cpg/dist (InputLayer)            (None, 5, 50)         0                                            
    ____________________________________________________________________________________________________
    dna/flatten_1 (Flatten)          (None, 31232)         0           dna/maxpooling1d_2[0][0]         
    ____________________________________________________________________________________________________
    cpg/merge_1 (Merge)              (None, 5, 100)        0           cpg/state[0][0]                  
                                                                       cpg/dist[0][0]                   
    ____________________________________________________________________________________________________
    dna/dense_1 (Dense)              (None, 128)           3997824     dna/flatten_1[0][0]              
    ____________________________________________________________________________________________________
    cpg/timedistributed_1 (TimeDistr (None, 5, 256)        25856       cpg/merge_1[0][0]                
    ____________________________________________________________________________________________________
    dna/activation_3 (Activation)    (None, 128)           0           dna/dense_1[0][0]                
    ____________________________________________________________________________________________________
    cpg/bidirectional_1 (Bidirection (None, 512)           787968      cpg/timedistributed_1[0][0]      
    ____________________________________________________________________________________________________
    dna/dropout_1 (Dropout)          (None, 128)           0           dna/activation_3[0][0]           
    ____________________________________________________________________________________________________
    cpg/dropout_1 (Dropout)          (None, 512)           0           cpg/bidirectional_1[0][0]        
    ____________________________________________________________________________________________________
    merge_1 (Merge)                  (None, 640)           0           dna/dropout_1[0][0]              
                                                                       cpg/dropout_1[0][0]              
    ____________________________________________________________________________________________________
    joint/dense_1 (Dense)            (None, 512)           328192      merge_1[0][0]                    
    ____________________________________________________________________________________________________
    joint/activation_1 (Activation)  (None, 512)           0           joint/dense_1[0][0]              
    ____________________________________________________________________________________________________
    joint/dropout_1 (Dropout)        (None, 512)           0           joint/activation_1[0][0]         
    ____________________________________________________________________________________________________
    joint/dense_2 (Dense)            (None, 512)           262656      joint/dropout_1[0][0]            
    ____________________________________________________________________________________________________
    joint/activation_2 (Activation)  (None, 512)           0           joint/dense_2[0][0]              
    ____________________________________________________________________________________________________
    joint/dropout_2 (Dropout)        (None, 512)           0           joint/activation_2[0][0]         
    ____________________________________________________________________________________________________
    cpg/BS27_1_SER (Dense)           (None, 1)             513         joint/dropout_2[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_3_SER (Dense)           (None, 1)             513         joint/dropout_2[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_5_SER (Dense)           (None, 1)             513         joint/dropout_2[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_6_SER (Dense)           (None, 1)             513         joint/dropout_2[0][0]            
    ____________________________________________________________________________________________________
    cpg/BS27_8_SER (Dense)           (None, 1)             513         joint/dropout_2[0][0]            
    ====================================================================================================
    Total params: 5509381
    ____________________________________________________________________________________________________
    INFO (2017-02-27 16:46:18,716): Computing output statistics ...
    Output statistics:
              name | nb_tot | nb_obs | frac_obs | mean |  var
    ---------------------------------------------------------
    cpg/BS27_1_SER |   1000 |    392 |     0.39 | 0.86 | 0.12
    cpg/BS27_3_SER |   1000 |    290 |     0.29 | 0.92 | 0.08
    cpg/BS27_5_SER |   1000 |    434 |     0.43 | 0.70 | 0.21
    cpg/BS27_6_SER |   1000 |    249 |     0.25 | 0.68 | 0.22
    cpg/BS27_8_SER |   1000 |    408 |     0.41 | 0.84 | 0.13
    
    Class weights:
    cpg/BS27_1_SER | cpg/BS27_3_SER | cpg/BS27_5_SER | cpg/BS27_6_SER | cpg/BS27_8_SER
    ----------------------------------------------------------------------------------
            0=0.86 |         0=0.92 |         0=0.70 |         0=0.68 |         0=0.84
            1=0.14 |         1=0.08 |         1=0.30 |         1=0.32 |         1=0.16
    
    INFO (2017-02-27 16:46:19,038): Loading data ...
    INFO (2017-02-27 16:46:19,040): Initializing callbacks ...
    INFO (2017-02-27 16:46:19,041): Training model ...
    
    Training samples: 1000
    Validation samples: 1000
    Epochs: 1
    Learning rate: 0.0001
    ====================================================================================================
    Epoch 1/1
    ====================================================================================================
    done (%) | time |   loss |    acc | cpg/BS27_8_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_5_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_1_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        12.8 |  0.0 | 7.8632 | 0.5661 |              0.1080 |              0.1378 |              0.0736 |              0.1799 |              0.1088 |             0.5000 |             0.5000 |             0.6800 |             0.4839 |             0.6667
        25.6 |  0.1 | 7.8120 | 0.5186 |              0.0963 |              0.1209 |              0.0635 |              0.1823 |              0.1174 |             0.5035 |             0.5000 |             0.6594 |             0.3982 |             0.5316
        38.4 |  0.1 | 7.7578 | 0.4955 |              0.0863 |              0.1165 |              0.0593 |              0.1725 |              0.1146 |             0.4785 |             0.4496 |             0.6919 |             0.3937 |             0.4636
        51.2 |  0.1 | 7.7305 | 0.5034 |              0.0877 |              0.1164 |              0.0585 |              0.1719 |              0.1102 |             0.4991 |             0.4709 |             0.7050 |             0.4064 |             0.4359
        64.0 |  0.2 | 7.7080 | 0.5021 |              0.0885 |              0.1119 |              0.0632 |              0.1757 |              0.1059 |             0.5059 |             0.4601 |             0.7181 |             0.4225 |             0.4037
        76.8 |  0.2 | 7.6902 | 0.5043 |              0.0919 |              0.1107 |              0.0652 |              0.1736 |              0.1089 |             0.5011 |             0.4667 |             0.7084 |             0.4378 |             0.4074
        89.6 |  0.2 | 7.6635 | 0.4983 |              0.0919 |              0.1090 |              0.0641 |              0.1731 |              0.1081 |             0.4957 |             0.4734 |             0.6836 |             0.4315 |             0.4069
       100.0 |  0.2 | 7.6457 | 0.5008 |              0.0927 |              0.1096 |              0.0646 |              0.1726 |              0.1078 |             0.5050 |             0.4806 |             0.6646 |             0.4467 |             0.4071
    Epoch 00000: val_loss improved from inf to 1.60208, saving model to ./models/dna_cpg/model_weights_val.h5
    
     split |   loss |    acc | cpg/BS27_6_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_5_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_1_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
     train | 7.6457 | 0.5008 |              0.1096 |              0.1726 |              0.0927 |              0.0646 |              0.1078 |             0.5050 |             0.4467 |             0.4071 |             0.4806 |             0.6646
       val | 1.6021 | 0.1671 |              0.2589 |              0.4091 |              0.1663 |              0.3807 |              0.3872 |             0.2467 |             0.1313 |             0.1507 |             0.2468 |             0.0600
    ====================================================================================================
    
    Training set performance:
      loss |    acc | cpg/BS27_6_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_1_SER_loss | cpg/BS27_5_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_1_SER_acc | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    7.6457 | 0.5008 |              0.1096 |              0.1726 |              0.0927 |              0.0646 |              0.1078 |             0.5050 |             0.4467 |             0.4071 |             0.4806 |             0.6646
    
    Validation set performance:
      loss |    acc | cpg/BS27_1_SER_loss | cpg/BS27_6_SER_loss | cpg/BS27_8_SER_loss | cpg/BS27_3_SER_loss | cpg/BS27_5_SER_loss | cpg/BS27_6_SER_acc | cpg/BS27_8_SER_acc | cpg/BS27_3_SER_acc | cpg/BS27_5_SER_acc | cpg/BS27_1_SER_acc
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    1.6021 | 0.1671 |              0.3872 |              0.2589 |              0.1663 |              0.3807 |              0.4091 |             0.2468 |             0.0600 |             0.1313 |             0.2467 |             0.1507
    INFO (2017-02-27 16:46:47,282): Done!


`--dna_module` and `--cpg_module` point to the output training directory of the DNA and CpG module, respectively, which contains their specification and weights:


```bash
ls $models_dir/dna
```

    events.out.tfevents.1488213772.lawrence model.json
    lc_train.csv                            model_weights_train.h5
    lc_val.csv                              model_weights_val.h5
    model.h5


`model.json` is the specification of the trained model, `model_weights_train.h5` the weights with the best performance on the training set, and `model_weights_val.h5` the weights with the best performance on the validation set. `--dna_model ./dna` is equivalent to using `--dna_model ./dna/model.json ./dna/model_weights_val.h5`, i.e. the validation weights will be used. The training weights can be used by `--dna_model ./dna/model.json ./dna/model_weights_train.h5` 

In the command above, we used `--joint_only` to only train the paramters of the joint module without training the pre-trained DNA and CpG module. Although the `--joint_only` arguments reduces training time, you might obtain better results by also fine-tuning the paramters of the DNA and CpG module without using `--joint_only`:

## Monitoring training progress

To check if your model is training correctly, you should monitor the training and validation loss. DeepCpG prints the loss and performance metrics for each output to the console as you can see from the previous commands. `loss` is the loss on the training set, `val_loss` the loss on the validation set, and `cpg/X_acc`, is, for example, the accuracy for output cell X. DeepCpG also stores these metrics in `X.csv` in the training output directory.

Both the training loss and validation loss should continually decrease until saturation. If at some point the validation loss starts to increase while the training loss is still decreasing, your model is overfitting the training set and you should stop training. DeepCpG will automatically stop training if the validation loss does not increase over the number of epochs that is specified by `--early_stopping` (by default 5). If your model is overfitting already after few epochs, your training set might be to small, and you could try to regularize your model model by choosing a higher value for `--dropout` or `--l2_decay`.

If your training loss fluctuates or increases, then you should decrease the learning rate. For more information on interpreting learning curves I recommend this tutorial.

To stop training before reaching the number of epochs specified by `--nb_epoch`, you can create a stop file (default name `STOP`) in the training output directory with `touch STOP`. See also **X**.

Watching numeric console outputs is not particular user friendly. [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) provides a more convenient and visually appealing way to mointor training. You can use TensorBoard provided that you are using the Tensorflow backend (**see X**). Simply go to the training output directory and run `tensorboard --logdir .`.

## Configuring the Keras backend

DeepCpG use the [Keras](https://keras.io) deep learning library, which supports [Theano](http://deeplearning.net/software/theano/) or [Tensorflow](https://www.tensorflow.org/) as backend. While Theano has long been the dominant deep learning library, Tensorflow is more suited for parallelizing computations on multiple GPUs and CPUs, and provides [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to interactively monitor training.

You can configure the backend by setting the `backend` attribute in `~/.keras/keras.json` to `tensorflow` or `theano`. Alternatively you can set the environemnt variable `KERAS_BACKEND='tensorflow'` to use Tensorflow, or `KERAS_BACKEND='theano'` to use Theano.

You can find more information about Keras backends [here](https://keras.io/backend/).

## Optimizing hyper-parameters

DeepCpG has differernt hyper-parameters, such as the learning rate, dropout rate, or module architectures. Although the performance of DeepCpG is relatively robust to different hyper-parameters, you can tweak performances by trying out different parameter combinations. For example, you could train different models with different paramters on a subset of your data, select the parameters with the highest performance on the validation set, and then train the full model.

The following hyper-parameters are most important (default values shown):
1. Learning rate: `--learning_rate 0.0001`
2. Dropout rate: `--dropout 0.0`
3. DNA model architecture: `--dna_model CnnL2h128`
4. Joint model architecture: `--joint_model JointL2h512`
5. CpG model architecture: `--cpg_model RnnL1`
6. L2 weight decay: `--l2_decay 0.0001`

The learning rate defines how agressively model parameters are updated during training. If the training loss changes only slowly (**see X**), you could try increasing the learning rate. If your model is overfitting of if the training loss fluctuates, you should decrease the learning rate. Reasonable values are 0.001, 0.0005, 0.0001, 0.00001, or values in between.

The dropout rate defines how strongly your model is regularized. If you have only few data and your model is overfitting, then you should increase the dropout rate. Reasonable values are, e.g., 0.0, 0.2, 0.4.

DeepCpG provides different architectures for the DNA, CpG, and joint module. Architectures are more or less complex, depending on how many layers and neurons say have. More complex model might yield better performances, but take longer to train and might overfit your data. See **X** for more information about different architectures.

L2 weight decay is an alternative to dropout for regularizing model training. If your model is overfitting, you might try 0.001, or 0.005.

## Deciding how long to train / Controlling training
The arguments `--nb_epoch` and `--early_stopping` control how long models are trained. 

`--nb_epoch` defines the maximum number of training epochs (default 30). After one epoch, the model has seen the entire training set once. The time per epoch hence depends on the size of the training set, but also on the complexity of the model that you are training and the hardware of your machine. On a large dataset, you have to train for fewer epochs than on a small dataset, since your model will have seen already a lot of training samples after one epoch. For training on about 3,000,000 samples, good default values are 20 for the DNA and CpG module, and 10 for the joint module.

Early stopping stops training if the loss on the validation set did not improve after the number of epochs that is specified by `--early_stopping` (default 5). If you are training without specifying a validation set with `--val_files`, early stopping will be deactivated.

`--max_time` sets the maximum training time in hours. This guarantees that training terminates after a certain amount of time regardless of the `--nb_epoch` or `--early_stopping` argument.

`--stop_file` defines the path of a file that, if it exists, stop training after the end of the current epoch. This is useful if you are monitoring training and want to terminate training manually as soon as the training loss starts to saturate regardless of `--nb_epoch` or `--early_stopping`. For example, when using `--stop_file ./train/STOP`, you can create an empty file with `touch ./train/STOP` to stop training at the end of the current epoch.



## Testing training

`dcpg_train.py` provides different arguments that allow to briefly test training before training the full model for a about a day.

`--nb_train_sample` and `--nb_val_sample` specify the number of training and validation samples. When using `--nb_train_sample 500`, the training loss should briefly decay and your model should start overfitting.

`--nb_output` and `--output_names` define the maximum number and the name of model outputs. For example, `--nb_output 3` will train only on the first three outputs, and `--output_names cpg/.*SER.*` only on outputs that include 'SER' in their name.

Analogously, `--nb_replicate` and `--replicate_name` define the number and name of cells that are used as input to the CpG module. `--nb_replicate 3` will only use observed methylation states from the first three cells, and allows to briefly test the CpG module.

`--dna_wlen` specifies the size of DNA sequence windows that will be used as input to the DNA module. For example, `--dna_wlen 101` will train only on windows of size 101, instead of using the full window length that was specified when creating data files with `dcpg_data.py`.

Analogously, `--cpg_wlen` specifies the sum of the number of observed CpG sites to the left and the right of the target CpG site for training the CpG module. For example, `--cpg_wlen 10` will use 5 observed CpG sites to the left and to the right.


## Fine-tuning
`dcpg_train.py` provides different arguments that allow to selectively train only some components of a model. 

With `--fine_tune`, only the output layer will be trained. As the name implies, this argument is useful for fine-tuning a pre-trained model.

`--train_models` specifies which modules are trained. For example, `--train_models joint` will train the joint module, but not the DNA and CpG module. `--train_models cpg joint` will train the CpG and joint module, but not the DNA module.

`--trainable` and `--not_trainable` allow including and excluding certain layers. For example, `--not_trainable '.*' --trainable 'dna/.*_2'` will only train the second layers of the DNA module.


`--freeze_filter` excludes the first convolutional layer of the DNA module from training.
