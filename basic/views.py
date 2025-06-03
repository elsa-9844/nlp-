from django.shortcuts import render
from .fine_tune import fine_tune_model  # Import fine-tuning function
import os
import pandas as pd

def about(request):
    if request.method == "POST":
        model = request.POST.get('model')  # Get selected model
        dataset_option = request.POST.get('dataset-option')  # Get dataset option (existing/custom)
        dataset = request.POST.get('dataset')  # Get dataset name if existing
        custom_file = request.FILES.get('custom-file')  # Get the uploaded file if custom

        # Handling file upload
        if custom_file:
            file_path = os.path.join("uploads", custom_file.name)  # Define path
            #with open(file_path, "wb+") as destination:
                #for chunk in custom_file.chunks():
                    #destination.write(chunk)

            # Load CSV if custom dataset
            custom_data = pd.read_csv(r"C:\Users\dell8\OneDrive\Documents\onedrive\Desktop\folder\add.csv")
        else:
            custom_data = None

        # Call fine-tuning function
        fine_tune_model(model, dataset_option, dataset, custom_data)

        return render(request, 'smb.html', {
            'model': model,
            'dataset_option': dataset_option,
            'dataset': dataset,
            'custom_file': custom_file
        })

    return render(request, 'smb.html')
