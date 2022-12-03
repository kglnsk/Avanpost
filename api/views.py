import os
import shutil

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.parsers import FormParser, MultiPartParser


from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import CategoryModel, PhotoModel
from .serializers import CategorySerializer, PhotoSerializer
from icrawler.builtin import GoogleImageCrawler

import torch
import clip
from PIL import Image

host_url = "https://web-production-0241.up.railway.app"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
THRESHOLD = 0.5
categories = ['snowboard','skateboard','truck','car','train','horse','lawnmower','ski','snowmobile','dump truck', 'van']


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
THRESHOLD = 0.5
categories = ['snowboard','skateboard','truck','car','train','horse','lawnmower','ski','snowmobile','dump truck', 'van']

def predict_image_from_path(path):
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    text = clip.tokenize(categories).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
        logits_per_image, logits_per_text = model(image, text)
        probs = list(logits_per_image.softmax(dim=-1).cpu().numpy().flatten())
        labels = []
        for i,prob in enumerate(probs):
            if prob>=THRESHOLD:
                labels.append(categories[i])
            
    return labels




def find_photos(category, num):
    paths = []
    crawl = GoogleImageCrawler(storage={'root_dir': f'/home/kirill/outsource_project/AvanpostHak/mediafiles/images/{category}'})
    crawl.crawl(keyword=category, max_num=num)
    # paths = [f'mediafiles/images/{category}/{f"{i + "0" * }"}' for i in range(0, num + 1)]
    for i in range(1, num + 1):
        path = f'mediafiles/images/{category}/'
        photo = "0" * (6 - len(str(i))) + f"{i}" + ".jpg"
        paths.append(path + photo)

    return paths


@api_view(['POST'])
def start_neuron(request):
    if request.method == "POST":
        print(request.data)
        img_urls = []

@api_view(['GET', 'POST'])
def save_photo(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == "GET":
        photo_path = "/home/kirill/outsource_project/AvanpostHak/mediafiles/images/tests"
        # Сюда вставлять нейронку, которая будет проверятся на тестовых фотках
        # photo_path - директория где находятся все фотки
        answer_dict = {}
        for filename in os.listdir(photo_path):
            if ".jpg" in filename.lower():
                labels = predict_image_from_path(os.path.join(filename,photo_path))
                answer_dict[filename] = labels
        
        try:
            shutil.rmtree(photo_path)
        except:
            pass
        return HttpResponse(answer_dict)


    if request.method == 'POST':
        serializer = PhotoSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()

    return HttpResponse("Not ok")


@api_view(['GET', 'POST'])
def take_category(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        snippets = CategoryModel.objects.all()
        serializer = CategorySerializer(snippets, many=True)
        for object in serializer.data:
            object['image_url'] = host_url + object['image_url']
        results = {"categories" : serializer.data}
        return Response(results)

    elif request.method == 'POST':
        serializer = CategorySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            category = serializer.data['category']

            paths = find_photos(category, 5)
            # Сюда вставалять нейронку category - это категория в формате строк Пример: 'bus',
            # paths - это путь до картинок, Пример: mediafiles/images/{category}/0000001
    return HttpResponse("ok")

