# EmotionModelAutomatisation

### How do I get set up? ###

The following section describes how to run the service locally.

* virtualenv venv
* source venv/bin/activate (Under windows run  $ venv/Scripts/activate.bat)
* pip install -r requirements.txt
* python manage.py runserver
* navigate to [localhost](http://127.0.0.1:8000/)

### Connection with database

```
python manage.py migrate
```
### Running the webservice
 
To run this project use this command:
```
python manage.py runserver
```

### In order to run this download Word2Vec and Glove Vectors, since the file sizes are very large:

[Glove] (https://nlp.stanford.edu/projects/glove/)
