import pickle
import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Load the model and scaling info
MODEL_PATH = os.path.join(settings.BASE_DIR, 'predict/models/shallot.pkl')
SCALING_INFO_PATH = os.path.join(settings.BASE_DIR, 'predict/models/scaling_info_shallot.pkl')

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(SCALING_INFO_PATH, 'rb') as scaling_file:
    scaling_info = pickle.load(scaling_file)

@api_view(['POST'])
def predict(request):
    data = request.data
    x1 = data.get('rainfall')
    x2 = data.get('production')

    # Normalize inputs using original min-max values from scaling_info
    normal_curah = (x1 - scaling_info['curah_hujan_min']) / (scaling_info['curah_hujan_max'] - scaling_info['curah_hujan_min'])
    normal_produksi = (x2 - scaling_info['produksi_min']) / (scaling_info['produksi_max'] - scaling_info['produksi_min'])

    x = [[normal_curah, normal_produksi]]
    prediction = model.predict(x)

    return Response({"predict": prediction.tolist()})
