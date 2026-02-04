from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=3000)
    gender= models.CharField(max_length=30)

class predicting_bike_usage(models.Model):

    Fid= models.CharField(max_length=3000)
    trip_id= models.CharField(max_length=3000)
    starttime= models.CharField(max_length=3000)
    stoptime= models.CharField(max_length=3000)
    bikeid= models.CharField(max_length=3000)
    tripduration= models.CharField(max_length=3000)
    from_station_id= models.CharField(max_length=3000)
    from_station_name= models.CharField(max_length=3000)
    to_station_id= models.CharField(max_length=3000)
    to_station_name= models.CharField(max_length=3000)
    usertype= models.CharField(max_length=3000)
    gender= models.CharField(max_length=3000)
    birthyear= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



