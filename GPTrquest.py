
import base64
from openai import OpenAI 
import json
import os
from openai import AzureOpenAI
from configparser import ConfigParser, ExtendedInterpolation
import httpx
import base64
from openai import OpenAI
import os
from PIL import Image
import matplotlib.pyplot as plt


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_img(img,name='name'):
    # Calculate the new width based on the aspect ratio
    new_height = 512
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = img.resize((new_width, new_height))

    # Save or show the resized image
    resized_image.save(f'{name}.jpg')  # Replace with desired save path
    plt.imshow(resized_image)
    plt.axis('off')
    plt.show()

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def send_message(messages, engine, max_response_tokens=500):
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=0.3,
        max_tokens=max_response_tokens,
        top_p=0.2,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

if __name__ == '__main__':

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('config-v1.x.ini', 'UTF-8')

    http_client = httpx.Client(verify='petrobras-ca-root.pem')

    client = AzureOpenAI(
        api_key=config['OPENAI']['OPENAI_API_KEY'],  
        api_version=config['OPENAI']['OPENAI_API_VERSION'],
        #azure_endpoint=config['OPENAI']['OPENAI_API_BASE'],
        base_url=config['OPENAI']['AZURE_OPENAI_BASE_URL'],
        http_client=http_client
    )

    dataset = json.load(open('/nethome/recpinfo/users/fibz/data/dataset/nwpu/val.json', 'r'))
    i = 0
    main_path = '/nethome/recpinfo/users/fibz/data/dataset/nwpu/images/{}'.format(dataset[i]['image_name']).replace('\\', '/')
    captions = dataset[i]['captions']
    main = Image.open(main_path)
    resize_img(main,'main')
    base64_main = encode_image('main.jpg')
    messages=[
            {"role": "system", "content": 'voce é um assistente '},
            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": 'descreva a imagem'
                 },
                 {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_main}"    
                     }
                 }
             ]}
        ]

    response = send_message(messages, engine='gpt-4o-2024-08-06', max_response_tokens=500)
    print(response)


