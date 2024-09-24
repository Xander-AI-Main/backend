import fitz
import google.generativeai as genai
import os
import uuid

api_key = "AIzaSyC93XxpL8z7dz4UjNBvECFYaobAOQre0Bk"
genai.configure(api_key=api_key)

class ChatbotPDF:
    def __init__(self, dataset_url, userId, task):
        self.dataset_url = dataset_url
        self.pdf_dir = 'pdfs'
        self.pdf_name = self.dataset_url.split('/')[-1]
        self.pdf_path = os.path.join(self.pdf_dir, self.pdf_name)
        self.userId = userId
        self._id = str(uuid.uuid4())
        self.task = task

    # def extract(self):
    # self.pdf_name = self.dataset_url.split('/')[-1]
    #     self.pdf_path = os.path.join(self.pdf_dir, self.pdf_name)
    #     pdf_document = fitz.open(self.pdf_path)

    #     text = ""
    #     for page_num in range(pdf_document.page_count):
    #         page = pdf_document.load_page(page_num)
    #         text += page.get_text()

    #     pdf_document.close()

    #     return text

    def execute(self):
        # if not os.path.exists(self.pdf_path):
        #     raise FileNotFoundError(f"The file {self.pdf_name} does not exist in the directory {self.pdf_dir}")
        
        # text = self.extract()

        # model = genai.GenerativeModel("gemini-1.5-flash")
        # response = model.generate_content(f"Context: {text} Answer the following question in less than 100 words no matter what and if the answer doesnt exist in the context, simple reply with answer not available: {self.question}")

        # print(response.text)
        interference_code = ""

#         api_code_python = f'''
# import requests
# import json

# url = "https://apiv3.xanderco.in/core/interference/" 

# data = {{
#     "data": "Your input text",
#     "modelId": '{self._id}',
#     "userId": '{self.userId}',
# }}

# try:
#     response = requests.post(url, json=data)

#     if response.status_code == 200:
#         # Print the response JSON
#         print("Response:")
#         print(json.dumps(response.json(), indent=2))
#     else:
#         print(f"Error: {{response.status_code}}")
#         print(response.text)
# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {{e}}")
# '''

        api_code_python = f'''
<script src="https://chatbot.xanderco.in/chatbot-loader.js?modelId={self._id}&userId={self.userId}"></script>
        '''

        api_code_js = f'''
const url = "https://apiv3.xanderco.in/core/interference/";

const data = {{
    data: "Your text here",
    modelId: '{self._id}',
    userId: '{self.userId}',
}};

const headers = {{
    'Content-Type': 'application/json',
}};

fetch(url, {{
    method: 'POST',
    headers: headers,
    body: JSON.stringify(data)
}})
.then(response => response.json().then(data => {{
    if (response.ok) {{
        console.log("Response:");
        console.log(JSON.stringify(data, null, 2));
    }} else {{
        console.error(`Error: ${{response.status}}`);
        console.error(data);
    }}
}}))
.catch(error => {{
    console.error(`An error occurred: ${{error}}`);
}});
'''
        model_obj = {
            "modelUrl": "",
            "helpers": [],
            "id": self._id,
            "architecture": "Language Models",
            "hyperparameters": {},
            "size": 0,
            "interferenceCode": interference_code,
            "datasetUrl": self.dataset_url,
            "codes": [
                    {"python": api_code_python},
                    {"javascript": api_code_js}
            ],
            "task": self.task,
        }

        return model_obj