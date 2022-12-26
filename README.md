# Book Genre Classification based on book description
##  Model
1) first we need to analysis the data which is the most important part
   1) data is prepared in csv format which i read i by pandas
   2) columns are provided as "['index', 'title', 'genre', 'summary']"
   3) 'genre' column will be used as label
   4) 'summary' column will be our feature
   5) the distribution of 'genre' column shows the imbalance data
   6) as imbalanced , accuracy is not a good metric and we will check precision, recall ,f1score, confusion matrix
   7) descriptions also include non-alphabet characters which should be cleaned both for train and test data
   8) nltk library , word2vec for word preprocessing
### data distribution
    thriller        1023
    fantasy         876
    science        647
    history        600
    horror         600
    crime          500
    romance        111
    psychology     100
    sports         100
    travel         100

2) then, the next step is model architecture:
    1) as the supervised data is provided (sequence -label), this problem is considered as sequence modeling
   2)  as the it is a Many-to-1 architecture as recurrent problems
   3) as the input features are long, to avoid vanishing gradient problem, gated architectures like GRU (faster) and LSTM are preferred
   4) Attention and Transformer(Attention without RNN) also can be applied
   5) final layer is a Dense layer with softmax activation since the problem as multiclass classification
   6) as a pretrained model, we can use Roberta from HuggingFace with pytorch code. it is based on transformer architecture, which is also used for text classification(modified version of BERT)
3) if confront with overfit, try regularization techniques 
4) Since the number of model parameters are more than the number of training samples, we will encounter overfitting. regularization techniques can be applied (Dropout, early stopping, WeightDecay, ...)
##
### The structure of my code(tensorflow-keras) is as follows:
    Congig
        Config.py   : as config reader from the json file
        config.json : contains the parameters to set, like learning rate, batch size, etc
    Data    
        data.csv    : our dataset
    modules:
        callbacks.py  : list of callbacks that are called in training loop/or model fit
        datareader.py : read data as pandas dataframe, extract and preprocess features and labels, split train and validation set
        evaluation.py : metric for evaluation (customization)
        model.py      : the architecture of model
    train.py          : model training and saving
    inference.py      : for testing
## training procedure:
 Training loss(for one  batch) at step 0:2.3178

 Training loss(for one  batch) at step 200:0.0169

 Training loss(for one  batch) at step 400:0.0010

## Docker
1) we need three files:
   1) Dockerfile (basically for one service)
   2) docker-compose.yaml (for serving several services)[for our problem is just on service: django app]
   3) requirements.txt  #[packages to be installed(mentioned in Dockerfile) specially : django]
2) run these commands in the above directory:
   1) docker-compose build
   2) docker-compose run --rm app django-admin startproject DjangoProject .
      1) run is for both building and running
      2) after the DjangoProject is added to current directory, we can add model predicition to it (Go to step Django)
   2) docker-compose up


## Django
as the command 'django-admin startproject DjangoProject' is already commited by docker compose, we skip this step.
we can try to different ways:

1) without using startapp a new app, and just
   1) create a views.py inside DjangoProject directory
   2) modify urls.py , settings.py(add DjangoProject to INSTALLED_APP)
   3) create a 'templates' directory in DjangoProject,add index.html and code the templates
2) with using startapp
   1) after python manage.py startapp DjangoApp , a new folder for app will be created
   2) inside this App folder, mkdir another folder called: models , and save the checkpoint of the model inside this directory
   3) add DjangoApp to INSTALLED_APP in settings.py
   4) calling the model checkpoint is happening inside apps.py , where we create a class
   5) prediction codes is commited in views.py
