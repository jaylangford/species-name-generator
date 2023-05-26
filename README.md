species-name-generator demonstrates the ability of recurrent neural networks to generate text. Using Python and pytorch, we can train the neural network on publicly available species name data to generate new, latin-sounding, [binomial names](https://en.wikipedia.org/wiki/Binomial_nomenclature).

Some example outputs (cherry-picked):

- meubeditaecusodus violagzarrcopthobi
- kyphyca oleuuzia
- biosinta difum
- craonufnus hiculiazarjeeuquiuli
- gyrxebiumax cpotuinffouypelum
- xubiwalias fylus
- dodeynngxunureus kacuphai
- noxtadbtlifatenseha lontaga
- iskra phiori
- bakutelin petizisporoporemciolactfev

The model design closely resembles the one in [pytorch-charRNN](https://github.com/mcleonard/pytorch-charRNN). 

Inspiration for this project came from [banned-license-plates](ttps://github.com/jnolis/banned-license-plates/tree/main).

Data come from the National Library of Medicine's National Center for Biotechnology Information. Download the data [here](https://ftp.ncbi.nih.gov/pub/taxonomy/).

To prepare the data for training:

```{console}
python src/name_gen/data.py --name file/with/name/data --node file/with/node/data
```

To train the model:

```{console}
python src/name_gen/train_model.py
```

Finally, to run the model:

```{console}
python src/name_gen/run_model.py models/name_of_model_file
```
