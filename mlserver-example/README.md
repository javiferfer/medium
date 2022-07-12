# Serving Python Machine Learning Models With Ease

Link: https://github.com/edshee/mlserver-example

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── tests          <- Unit and integration tests
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── settings.json      <- This contains the configuration for the server itself


Ref: https://pravarmahajan.github.io/fashion/


Ever trained a new model and just wanted to use it through an API straight away? Sometimes you don't want to bother writing Flask code or containerizing your model and running it in Docker. If that sounds like you, you definitely want to check out [MLServer](https://github.com/seldonio/mlserver). It's a python based inference server that [recently went GA](https://www.seldon.io/introducing-mlserver) and what's really neat about it is that it's a highly-performant server designed for production environments too. That means that, by serving models locally, you are running in the exact same environment as they will be in when they get to production. 

This blog walks you through how to use MLServer by using a couple of image models as examples...

## Dataset

The dataset we're going to work with is the [Fashion MNIST dataset](https://www.kaggle.com/zalando-research/fashionmnist). It contains 70,000 images of clothing in greyscale 28x28 pixels across 10 different classes (top, dress, coat, trouser etc...). 

*If you want to reproduce the code from this blog, make sure you download the files and extract them in to a folder named `data`. They have been omitted from the github repo because they are quite large.*

```
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```

## Training the Scikit-learn Model

First up, we're going to train a support vector machine (SVM) model using the [scikit-learn](lhttps://scikit-learn.org/stable/) framework. We'll then save the model to a file named `Fashion-MNIST.joblib`.

Note: The SVM algorithm is not particularly well suited to large datasets because of it's quadratic nature. The model in this example will, depending on your hardware, take a couple of minutes to train.

## Serving the Scikit-learn Model

Ok, so we've now got a saved model file `Fashion-MNIST.joblib`. Let's take a look at how we can serve that using MLServer...

First up, we need to install MLServer.

`pip install mlserver`

The additional runtimes are optional but make life really easy when serving models, we'll install the Scikit-Learn and XGBoost ones too

`pip install mlserver-sklearn mlserver-xgboost`

*You can find details on all of the inference runtimes [here](https://mlserver.readthedocs.io/en/latest/runtimes/index.html#included-inference-runtimes)*

Once we've done that, all we need to do is add two configuration files:
- `settings.json` - This contains the configuration for the server itself.
- `model-settings.json` - As the name suggests, this file contains configuration for the model we want to run. 

The `name` parameter should be self-explanatory. It gives MLServer a unique identifier which is particularly useful when serving multiple models (we'll come to that in a bit). The `implementation` defines which pre-built server, if any, to use. It is heavily coupled to the machine learning framework used to train your model. In our case we trained the model using scikit-learn so we're going to use the scikit-learn implementation for MLServer. For model `parameters` we just need to provide the location of our model file as well as a version number.

That's it, two small config files and we're ready to serve our model using the command:

`mlserver start .` 

Boom, we've now got our model running on a production-ready server locally. It's now ready to accept requests over HTTP and gRPC (default ports `8080` and `8081` respectively).

## Testing the Model

Now that our model is up and running. Let's send some requests to see it in action.

To make predictions on our model, we need to send a POST request to the following URL:

`http://localhost:8080/v2/models/<MODEL_NAME>/versions/<VERSION>/infer`

That means to access our scikit-learn model that we trained earlier, we need to replace the `MODEL_NAME` with `fashion-sklearn` and `VERSION` with `v1`. 

The code below shows how to import the test data, make a request to the model server and then compare the result with the actual label:

When running the `test.py` code above we get the following response from MLServer:

```json
{
  "model_name": "fashion-sklearn",
  "model_version": "v1",
  "id": "31c3fa70-2e56-49b1-bcec-294452dbe73c",
  "parameters": null,
  "outputs": [
    {
      "name": "predict",
      "shape": [
        1
      ],
      "datatype": "INT64",
      "parameters": null,
      "data": [
        0
      ]
    }
  ]
}
```

You'll notice that MLServer has generated a request id and automatically added metadata about the model and version that was used to serve our request. Capturing this kind of metadata is super important once our model gets to production; it allows us to log every request for audit and troubleshooting purposes. 

You might also notice that MLServer has returned an array for `outputs`. In our request we only sent one row of data but MLServer also handles batch requests and returns them together. You can even use a technique called [adaptive batching](https://mlserver.readthedocs.io/en/latest/user-guide/adaptive-batching.html) to optimise the way multiple requests are handled in production environments. 

In our example above, the model's prediction can be found in `outputs[0].data` which shows that the model has labeled this sample with the category `0` (The value 0 corresponds to the category `t-shirt/top`). The true label for that sample was a `0` too so the model got this prediction correct!

## Training the XGBoost Model

Now that we've seen how to create and serve a single model using MLServer, let's take a look at how we'd handle multiple models trained in different frameworks. 

We'll be using the same Fashion MNIST dataset but, this time, we'll train an [XGBoost](https://xgboost.readthedocs.io/en/stable/) model instead.

The code above, used to train the XGBoost model, is similar to the code we used earlier to train the scikit-learn model but this time our model has been saved in an XGBoost-compatible format as `Fashion_MNIST.json`.

## Serving Multiple Models

One of the cool things about MLServer is that it supports [multi-model serving](https://mlserver.readthedocs.io/en/latest/examples/mms/README.html). This means that you don't have to create or run a new server for each ML model you want to deploy. Using the models we built above, we'll use this feature to serve them both at once.

When MLServer starts up, it will search the directory (and any subdirectories) for `model-settings.json` files. If you've got multiple `model-settings.json` files then it'll automatically serve them all. 

*Note: you still only need a single `settings.json` (server config) file in the root directory*

Here's a breakdown of my directory structure for reference:

```bash
.
├── data
│   ├── fashion-mnist_test.csv
│   └── fashion-mnist_train.csv
├── models
│   ├── sklearn
│   │   ├── Fashion_MNIST.joblib
│   │   ├── model-settings.json
│   │   ├── test.py
│   │   └── train.py
│   └── xgboost
│       ├── Fashion_MNIST.json
│       ├── model-settings.json
│       ├── test.py
│       └── train.py
├── README.md
├── settings.json
└── test_models.py
```

Notice that there are two `model-settings.json` files - one for the scikit-learn model and one for the XGBoost model. 

We can now just run `mlserver start .` and it will start handling requests for both models.

```bash
[mlserver] INFO - Loaded model 'fashion-sklearn' succesfully.
[mlserver] INFO - Loaded model 'fashion-xgboost' succesfully.
```

## Testing Accuracy of Multiple Models

With both models now up and running on MLServer, we can use the samples from our test set to validate how accurate each of our models is. 

The following code sends a batch request (containing the full test set) to each of the models and then compares the predictions received to the true labels. Doing this across the whole test set gives us a reasonably good measure for each model's accuracy, which gets printed at the end.

The results show that the XGBoost model slightly outperforms the SVM scikit-learn one:

```
Model Accuracy for fashion-xgboost: 0.8953
Model Accuracy for fashion-sklearn: 0.864
```

## Summary

Hopefully by now you've gained an understanding of how easy it is to serve models using [MLServer](https://mlserver.readthedocs.io/en/latest/index.html). For further info it's worth reading the [docs](https://mlserver.readthedocs.io/en/latest/index.html) and taking a look at the [examples for different frameworks](https://mlserver.readthedocs.io/en/latest/examples/index.html). 

For [MLFlow](https://mlflow.org/) users you can now serve [models directly in MLFlow using MLServer](https://www.mlflow.org/docs/latest/models.html#serving-with-mlserver-experimental) and if you're a [Kubernetes](https://kubernetes.io/) user you should definitely check out [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/index.html) - an open source tool that deploys models to Kubernetes (it uses MLServer under the covers). 

All of the code from this example can be found [here](https://github.com/edshee/mlserver-example).


## Docker

Create a dockerfile with the following command:
```
mlserver dockerfile .
```

This creates an image with the name python-mlserver. The mlserver build subcommand will search for any Conda environment file (i.e. named either as environment.
yaml or conda.yaml) and / or any requirements.txt present in your root folder. These can be used to tell MLServer
what Python environment is required in the final Docker image.
```
docker build -t python-mlserver .
```

To see the image created, at least in Mac, go to the Docker Desktop

Run the image
```
docker run -it -p 8080:8080 python-mlserver
```
