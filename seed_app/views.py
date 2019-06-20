from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
# Create your views here.

out_dict = {'Average': 0,
            'Bad': 1,
            'Excellent': 2,
            'Good': 3,
            'Worst': 4,
            }

li = list(out_dict.keys())

def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        m = str(filename)
        K.clear_session()
        im = Image.open("{}/".format(settings.MEDIA_ROOT) + m)
        j = im.resize((256, 256),)
        l = "predicted.jpg"
        j.save("{}/".format(settings.MEDIA_ROOT) + l)
        file_url = fs.url(l)
        mod = load_model('seed_app/model.hdf5', compile=False)
        img = image.load_img(myfile, target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = mod.predict(x)
        d = preds.flatten()
        j = d.max()
        for index, item in enumerate(d):
            if item == j:
                result = li[index]
                return render(request, "index.html", {
                                 'result': result, 'file_url': file_url })

    return render(request, "index.html")
