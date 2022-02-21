# jet-inn
INN for density-based anomaly search in particle jets.

## dependences
* [pytorch](https://pytorch.org/) - for the neural network
* [scikit-learn](https://scikit-learn.org) - to calculate ROC-curves
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org) - to generate random orthogonal transformations
* [pandas](https://github.com/pandas-dev/pandas) - for data import
* [pytables](https://www.pytables.org/index.html) - for data import
* [matplotlib](https://matplotlib.org/) - for plotting
* [tqdm](https://github.com/tqdm/tqdm) - for the progress bar
* [energyflow](https://energyflow.network/) - to calculate EFP's (only for preprocessing)

## run

To try out the network, you can use the [top tagging dataset](https://arxiv.org/pdf/1902.09914.pdf), for example. You can download the test dataset [here](https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6), which is large enough for training and testing. If the file was saved under ```~/Downloads/test.h5```, you can train the network with the following commands.

```bash
# install dependences
python3 -m pip install -r requirements.txt

# preprocessing
python3 prepare_data/split.py ~/Downloads/test.h5
python3 prepare_data/cons2efps.py

# training
python3 src/main.py params/top.json
python3 src/main.py params/qcd.json

# ploting
python3 src/plot.py
```

For questions/comments about the code contact: buss@thphys.uni-heidelberg.de

---

This code was written for the paper:

**What’s Anomalous in LHC Jets?**<br/>
https://arxiv.org/abs/2202.00686<br/>
*Thorsten Buss, Barry M. Dillon, Thorben Finke, Michael Krämer, Alessandro Morandini, Alexander Mück, Ivan Oleksiyuk and Tilman Plehn*
