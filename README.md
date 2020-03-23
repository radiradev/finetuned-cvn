# DUNE CVN

DUNE CVN [code](modules/dune_cvn.py).

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
