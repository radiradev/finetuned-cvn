# DUNE CVN

DUNE CVN [code](modules/dune_cvn.py).

# Environment

```
python==3.6
tensorflow==2.1.0
numpy==1.17.3
pickle=4.0
sklearn=0.22.1
```

# Usage

```
python test.py
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --batch_size 	       |	10          |Batch size
| --model         |        'saved_model/model.json'          |JSON model
| --weights         |        'saved_model/weights.h5'          |HDF5 pre-trained model weights
| --dataset         |        'dataset'          |Dataset path
| --partition         |        'dataset/partition.p'          |Pickled partition
| --shuffle         |        False          |Shuffle partition
| --print_model         |        False          |Print model summary
| --output_file         |        'output/results.txt'          |Output file
